"""AI Analyst — LLM-powered deep analysis of screener candidates via Ollama."""

import json
import logging
import threading
import urllib.request
from datetime import datetime
from typing import Optional

import pandas as pd

from config.settings import (
    AI_MODEL,
    AI_CONFIDENCE_THRESHOLD,
    ALLOW_SHORT_SELLING,
    OLLAMA_HOST,
    MIN_RISK_REWARD_RATIO,
)
from core.models import Signal, Action, TradeType

logger = logging.getLogger(__name__)

# Token usage tracking (guarded by _token_lock for thread safety)
_daily_token_usage = {"input": 0, "output": 0, "date": None}
_token_lock = threading.Lock()


def _reset_daily_usage_if_needed() -> None:
    today = datetime.now().date().isoformat()
    if _daily_token_usage["date"] != today:
        _daily_token_usage["input"] = 0
        _daily_token_usage["output"] = 0
        _daily_token_usage["date"] = today


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT = """You are a disciplined stock trader making real money decisions. Analyze this stock using a strict checklist approach.

## Stock: {ticker} ({exchange})

## Recent Price Action (last 5 days)
{price_action}

## Technical Indicators
{indicators}

## News Headlines
{news}

## Macro/Political Headlines
{macro_news}

## Decision Checklist — evaluate each before deciding:

1. TREND: Is the stock in a clear uptrend (for buy) or downtrend (for sell)? Are moving averages aligned (MA5 > MA10 > MA20 for uptrend)?
2. MOMENTUM: Is momentum confirming the move? Check RSI direction and MACD alignment. Reject if RSI is already extreme (>80 for buy, <20 for sell).
3. VOLUME: Is there volume confirmation? Moves without volume are unreliable.
4. RISK/REWARD: Is the reward at least 1.5x the risk? Calculate: (take_profit - entry) / (entry - stop_loss) >= 1.5
5. NEWS SENTIMENT: Do recent headlines support or contradict the technical signal?
6. ANTI-CHASE RULE: Has the stock already moved more than 5% in the signal direction recently? If yes, you missed the move — say "hold".
7. MACRO/POLITICAL RISK: Do current political, regulatory, or macroeconomic conditions (elections, trade wars, sanctions, Fed policy, sector regulation) create risk or opportunity for this stock? Consider how macro headlines might override or reinforce the technical picture.

## Response Format
Return a JSON object with these exact fields:
- "action": {actions}
- "confidence": integer 0-100
- "entry_price": specific entry price (float)
- "stop_loss": stop-loss price (float) — place below recent support for buys, above resistance for sells
- "take_profit": take-profit price (float) — must give at least 1.5:1 reward/risk
- "trade_type": "day" (close before market end) or "swing" (hold overnight)
- "reasoning": 2-3 sentences covering your checklist assessment

## Discipline Rules
- Default to "hold" unless at least 5 of 7 checklist items are clearly favorable
- Never chase: if the stock already ran >5% toward the signal, say "hold"
- Confidence above 65 only when trend + momentum + volume all align and macro environment is not hostile
- Be honest about uncertainty — a confident "hold" is better than a shaky "buy"
- Set stop-loss at a technical level (support/resistance), not an arbitrary percentage{short_rule}"""


_ACTIONS_WITH_SHORTS = '"buy", "sell", or "hold"'
_ACTIONS_NO_SHORTS = '"buy" or "hold"'
_SHORT_RULE_DISABLED = (
    "\n- This system does not short stocks. Never recommend 'sell' — only 'buy' or 'hold'."
)


def _build_prompt(
    ticker: str,
    exchange: str,
    df: pd.DataFrame,
    indicator_values: dict,
    news: list[str],
    macro_news: list[str] | None = None,
) -> str:
    """Build the analysis prompt with all context."""
    # Recent price action
    recent = df.tail(5)
    price_lines = []
    for idx, row in recent.iterrows():
        date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
        price_lines.append(
            f"  {date_str}: O={row['open']:.2f} H={row['high']:.2f} "
            f"L={row['low']:.2f} C={row['close']:.2f} V={row['volume']:,.0f}"
        )
    price_action = "\n".join(price_lines) if price_lines else "No recent data"

    # Indicators
    if indicator_values:
        indicators = "\n".join(f"  - {k}: {v}" for k, v in indicator_values.items())
    else:
        indicators = "  No indicator data available"

    # News — sanitize headlines to mitigate prompt injection from external sources
    def _sanitize_headline(h: str) -> str:
        return h[:200].replace("\n", " ").replace("##", "").replace("---", "").strip()

    if news:
        news_text = "\n".join(f"  - {_sanitize_headline(h)}" for h in news[:5])
    else:
        news_text = "  No recent news available"

    # Macro/political news
    if macro_news:
        macro_text = "\n".join(f"  - {_sanitize_headline(h)}" for h in macro_news[:5])
    else:
        macro_text = "  No macro/political headlines available"

    actions = _ACTIONS_WITH_SHORTS if ALLOW_SHORT_SELLING else _ACTIONS_NO_SHORTS
    short_rule = "" if ALLOW_SHORT_SELLING else _SHORT_RULE_DISABLED

    return ANALYSIS_PROMPT.format(
        ticker=ticker,
        exchange=exchange,
        price_action=price_action,
        indicators=indicators,
        news=news_text,
        macro_news=macro_text,
        actions=actions,
        short_rule=short_rule,
    )


# ---------------------------------------------------------------------------
# Ollama LLM call
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str) -> Optional[dict]:
    """Call a local model via Ollama HTTP API with JSON output."""
    payload = json.dumps({
        "model": AI_MODEL,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"num_predict": 1024},
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    response = urllib.request.urlopen(req, timeout=1800)
    result = json.loads(response.read())

    # Check for errors BEFORE tracking usage — error responses should
    # not count toward daily token limits
    if "error" in result:
        logger.warning("Ollama error: %s", result["error"])
        return None

    # Track usage only on successful responses
    with _token_lock:
        _reset_daily_usage_if_needed()
        if "prompt_eval_count" in result:
            _daily_token_usage["input"] += result["prompt_eval_count"]
        if "eval_count" in result:
            _daily_token_usage["output"] += result["eval_count"]

    duration = result.get("total_duration", 0) / 1e9
    logger.info("Ollama response in %.1fs (%s)", duration, AI_MODEL)

    try:
        return json.loads(result["response"])
    except (KeyError, json.JSONDecodeError) as e:
        logger.warning("Ollama response malformed (missing 'response' key or invalid JSON): %s", e)
        return None


def _call_llm(prompt: str, max_retries: int = 3) -> Optional[dict]:
    """Call Ollama with retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            result = _call_ollama(prompt)

            if result and _validate_response(result):
                return result
            else:
                logger.warning("Invalid LLM response on attempt %d: %s", attempt, result)

        except Exception as e:
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt, max_retries, e)

    return None


def _validate_response(data: dict) -> bool:
    """Validate LLM response has all required fields with valid values."""
    required = ["action", "confidence", "entry_price", "stop_loss", "take_profit", "reasoning", "trade_type"]
    missing = [f for f in required if f not in data]
    if missing:
        logger.warning("LLM response missing fields: %s. Keys present: %s", missing, list(data.keys()))
        return False

    if data["action"] not in ("buy", "sell", "hold"):
        logger.warning("LLM response invalid action: %r", data["action"])
        return False
    # The short-selling gate lives in risk.check_short_selling, which knows
    # whether the user holds the stock. Rejecting every SELL here would block
    # AI-driven exits on held longs (a legitimate signal shape). The prompt
    # also steers the LLM away from "sell" when shorts are disabled; this
    # validator used to fight the prompt redundantly.
    if data.get("trade_type") not in ("day", "swing"):
        logger.warning("LLM response invalid trade_type: %r", data.get("trade_type"))
        return False
    if not isinstance(data["confidence"], (int, float)) or not (0 <= data["confidence"] <= 100):
        logger.warning("LLM response invalid confidence: %r", data["confidence"])
        return False
    # Price fields are only required for buy/sell — holds legitimately have no prices
    if data["action"] in ("buy", "sell"):
        for price_field in ("entry_price", "stop_loss", "take_profit"):
            if not isinstance(data[price_field], (int, float)) or data[price_field] <= 0:
                logger.warning("LLM response invalid %s: %r", price_field, data[price_field])
                return False

        # Validate price relationships
        entry = data["entry_price"]
        sl = data["stop_loss"]
        tp = data["take_profit"]

        if data["action"] == "buy":
            if sl >= entry:
                logger.warning("BUY stop_loss %.2f must be below entry %.2f", sl, entry)
                return False
            if tp <= entry:
                logger.warning("BUY take_profit %.2f must be above entry %.2f", tp, entry)
                return False
            risk = entry - sl
            reward = tp - entry
            if risk > 0 and reward / risk < MIN_RISK_REWARD_RATIO:
                logger.warning("BUY R:R %.2f below minimum %.2f", reward / risk, MIN_RISK_REWARD_RATIO)
                return False
        elif data["action"] == "sell":
            if sl <= entry:
                logger.warning("SELL stop_loss %.2f must be above entry %.2f", sl, entry)
                return False
            if tp >= entry:
                logger.warning("SELL take_profit %.2f must be below entry %.2f", tp, entry)
                return False
            risk = sl - entry
            reward = entry - tp
            if risk > 0 and reward / risk < MIN_RISK_REWARD_RATIO:
                logger.warning("SELL R:R %.2f below minimum %.2f", reward / risk, MIN_RISK_REWARD_RATIO)
                return False

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_candidate(
    ticker: str,
    exchange: str,
    df: pd.DataFrame,
    indicator_values: dict,
    news: list[str],
    macro_news: list[str] | None = None,
) -> Optional[Signal]:
    """Analyze a single screener candidate with the LLM.

    Pure function — receives all data, doesn't fetch anything.

    Returns a Signal if confidence >= threshold, else None.
    """
    prompt = _build_prompt(ticker, exchange, df, indicator_values, news, macro_news=macro_news)
    result = _call_llm(prompt)

    if not result:
        logger.warning("No valid LLM response for %s", ticker)
        return None

    if result["confidence"] < AI_CONFIDENCE_THRESHOLD:
        logger.info(
            "Skipping %s: confidence %d < threshold %d",
            ticker, result["confidence"], AI_CONFIDENCE_THRESHOLD,
        )
        return None

    action_map = {"buy": Action.BUY, "sell": Action.SELL, "hold": Action.HOLD}
    action = action_map.get(result["action"], Action.HOLD)

    if action == Action.HOLD:
        return None

    trade_type_map = {"day": TradeType.DAY, "swing": TradeType.SWING}
    trade_type = trade_type_map.get(result.get("trade_type", "day"), TradeType.DAY)

    return Signal(
        ticker=ticker,
        action=action,
        confidence=float(result["confidence"]),
        entry_price=float(result["entry_price"]),
        stop_loss=float(result["stop_loss"]),
        take_profit=float(result["take_profit"]),
        reasoning=result.get("reasoning", ""),
        source="ai",
        exchange=exchange,
        trade_type=trade_type,
        indicator_values=indicator_values,
    )


def analyze_batch(
    candidates: list[dict],
    on_signal=None,
    on_progress=None,
    macro_news: list[str] | None = None,
) -> list[Signal]:
    """Analyze a batch of screener candidates sequentially.

    Args:
        candidates: List of dicts with keys:
            ticker, exchange, df, indicator_values, news
        on_signal: Optional callback called immediately when a signal is approved.
        on_progress: Optional callback(current, total) called after each candidate.
        macro_news: Broad market/political headlines shared across all candidates.

    Returns list of approved Signal objects.
    """
    signals = []
    total = len(candidates)

    for i, c in enumerate(candidates, 1):
        try:
            ticker = c["ticker"]
        except KeyError as e:
            logger.error("Skipping malformed candidate (missing key %s)", e)
            continue
        logger.info("Analyzing candidate %d/%d: %s", i, total, ticker)
        try:
            signal = analyze_candidate(
                ticker=ticker,
                exchange=c["exchange"],
                df=c["df"],
                indicator_values=c.get("indicator_values", {}),
                news=c.get("news", []),
                macro_news=macro_news,
            )
        except (KeyError, TypeError) as e:
            logger.error("Skipping candidate %s (missing key %s)", ticker, e)
            if on_progress:
                on_progress(i, total)
            continue
        if signal:
            signals.append(signal)
            logger.info(
                "AI approved %s: %s confidence=%d",
                signal.ticker, signal.action.value, signal.confidence,
            )
            if on_signal:
                on_signal(signal)

        if on_progress:
            on_progress(i, total)

    logger.info(
        "AI analysis complete: %d/%d candidates approved",
        len(signals), len(candidates),
    )
    return signals


def get_daily_token_usage() -> dict:
    """Return today's token usage stats."""
    with _token_lock:
        _reset_daily_usage_if_needed()
        return dict(_daily_token_usage)

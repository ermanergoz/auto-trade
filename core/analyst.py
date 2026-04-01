"""AI Analyst — LLM-powered deep analysis of screener candidates via Ollama."""

import json
import logging
import urllib.request
from datetime import datetime
from typing import Optional

import pandas as pd

from config.settings import AI_MODEL, AI_CONFIDENCE_THRESHOLD, OLLAMA_HOST
from core.models import Signal, Action, TradeType

logger = logging.getLogger(__name__)

# Token usage tracking
_daily_token_usage = {"input": 0, "output": 0, "date": None}


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

## Decision Checklist — evaluate each before deciding:

1. TREND: Is the stock in a clear uptrend (for buy) or downtrend (for sell)? Are moving averages aligned (MA5 > MA10 > MA20 for uptrend)?
2. MOMENTUM: Is momentum confirming the move? Check RSI direction and MACD alignment. Reject if RSI is already extreme (>80 for buy, <20 for sell).
3. VOLUME: Is there volume confirmation? Moves without volume are unreliable.
4. RISK/REWARD: Is the reward at least 1.5x the risk? Calculate: (take_profit - entry) / (entry - stop_loss) >= 1.5
5. NEWS SENTIMENT: Do recent headlines support or contradict the technical signal?
6. ANTI-CHASE RULE: Has the stock already moved more than 5% in the signal direction recently? If yes, you missed the move — say "hold".

## Response Format
Return a JSON object with these exact fields:
- "action": "buy", "sell", or "hold"
- "confidence": integer 0-100
- "entry_price": specific entry price (float)
- "stop_loss": stop-loss price (float) — place below recent support for buys, above resistance for sells
- "take_profit": take-profit price (float) — must give at least 1.5:1 reward/risk
- "trade_type": "day" (close before market end) or "swing" (hold overnight)
- "reasoning": 2-3 sentences covering your checklist assessment

## Discipline Rules
- Default to "hold" unless at least 4 of 6 checklist items are clearly favorable
- Never chase: if the stock already ran >5% toward the signal, say "hold"
- Confidence above 70 only when trend + momentum + volume all align
- Be honest about uncertainty — a confident "hold" is better than a shaky "buy"
- Set stop-loss at a technical level (support/resistance), not an arbitrary percentage"""


def _build_prompt(
    ticker: str,
    exchange: str,
    df: pd.DataFrame,
    indicator_values: dict,
    news: list[str],
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

    # News
    if news:
        news_text = "\n".join(f"  - {h}" for h in news[:5])
    else:
        news_text = "  No recent news available"

    return ANALYSIS_PROMPT.format(
        ticker=ticker,
        exchange=exchange,
        price_action=price_action,
        indicators=indicators,
        news=news_text,
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

    response = urllib.request.urlopen(req, timeout=600)
    result = json.loads(response.read())

    # Track usage
    _reset_daily_usage_if_needed()
    if "prompt_eval_count" in result:
        _daily_token_usage["input"] += result["prompt_eval_count"]
    if "eval_count" in result:
        _daily_token_usage["output"] += result["eval_count"]

    duration = result.get("total_duration", 0) / 1e9
    logger.info("Ollama response in %.1fs (%s)", duration, AI_MODEL)

    return json.loads(result["response"])


def _call_llm(prompt: str, max_retries: int = 3) -> Optional[dict]:
    """Call Ollama with retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            result = _call_ollama(prompt)

            if result and _validate_response(result):
                return result
            else:
                logger.warning("Invalid LLM response on attempt %d", attempt)

        except Exception as e:
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt, max_retries, e)

    return None


def _validate_response(data: dict) -> bool:
    """Validate LLM response has all required fields with valid values."""
    required = ["action", "confidence", "entry_price", "stop_loss", "take_profit", "reasoning"]
    for field in required:
        if field not in data:
            return False

    if data["action"] not in ("buy", "sell", "hold"):
        return False
    if not isinstance(data["confidence"], (int, float)) or not (0 <= data["confidence"] <= 100):
        return False
    if data["entry_price"] <= 0 or data["stop_loss"] <= 0 or data["take_profit"] <= 0:
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
) -> Optional[Signal]:
    """Analyze a single screener candidate with the LLM.

    Pure function — receives all data, doesn't fetch anything.

    Returns a Signal if confidence >= threshold, else None.
    """
    prompt = _build_prompt(ticker, exchange, df, indicator_values, news)
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
) -> list[Signal]:
    """Analyze a batch of screener candidates sequentially.

    Args:
        candidates: List of dicts with keys:
            ticker, exchange, df, indicator_values, news

    Returns list of approved Signal objects.
    """
    signals = []

    for i, c in enumerate(candidates, 1):
        logger.info("Analyzing candidate %d/%d: %s", i, len(candidates), c["ticker"])
        signal = analyze_candidate(
            ticker=c["ticker"],
            exchange=c["exchange"],
            df=c["df"],
            indicator_values=c.get("indicator_values", {}),
            news=c.get("news", []),
        )
        if signal:
            signals.append(signal)
            logger.info(
                "AI approved %s: %s confidence=%d",
                signal.ticker, signal.action.value, signal.confidence,
            )

    logger.info(
        "AI analysis complete: %d/%d candidates approved",
        len(signals), len(candidates),
    )
    return signals


def get_daily_token_usage() -> dict:
    """Return today's token usage stats."""
    _reset_daily_usage_if_needed()
    return dict(_daily_token_usage)

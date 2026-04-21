"""AI Analyst — LLM-powered deep analysis of screener candidates.

Routing: Gemini (primary) -> Ollama (fallback). See _call_llm for the router
and _gemini_exhausted for the process-lifetime short-circuit that mirrors the
Tavily->YFinance exhaustion pattern in core/data.py.
"""

import json
import logging
import threading
import urllib.error
import urllib.request
from datetime import datetime
from typing import Optional

import pandas as pd

from config.settings import (
    AI_MODEL,
    AI_PROVIDER,
    AI_CONFIDENCE_THRESHOLD,
    ALLOW_SHORT_SELLING,
    GEMINI_API_KEY,
    GEMINI_HOST,
    GEMINI_MODEL,
    OLLAMA_HOST,
    MIN_RISK_REWARD_RATIO,
)
from core.models import Signal, Action, TradeType

logger = logging.getLogger(__name__)

# Per-provider token usage (guarded by _token_lock). Shape changed from flat
# {input,output,date} on 2026-04-21 when Gemini became primary — no external
# callers depended on the old shape (repo-wide grep of get_daily_token_usage).
_daily_token_usage = {
    "gemini": {"input": 0, "output": 0},
    "ollama": {"input": 0, "output": 0},
    "date": None,
}
_token_lock = threading.Lock()

# Process-lifetime flag — once Gemini signals plan/quota exhaustion or an
# unrecoverable auth failure, skip it for the rest of the process and go
# straight to Ollama. Mirrors _tavily_exhausted in core/data.py. Resets on
# process restart (covers "user topped up credits and restarted").
_gemini_exhausted = threading.Event()

# Substring markers (case-insensitive) that classify a Gemini error body as
# PERMANENT exhaustion. Tighter than the Tavily list because Gemini emits
# "Quota exceeded per minute" for transient per-minute rate limits — a bare
# "quota" marker would latch the flag on recoverable conditions.
_GEMINI_EXHAUSTION_MARKERS = (
    "prepayment credits are depleted",
    "usage limit",
    "limit: 0",
    "free tier limit",
    "quota metric exceeded",
)


def _is_permanent_gemini_exhaustion(body_text: str) -> bool:
    lowered = (body_text or "").lower()
    return any(marker in lowered for marker in _GEMINI_EXHAUSTION_MARKERS)


def _reset_daily_usage_if_needed() -> None:
    today = datetime.now().date().isoformat()
    if _daily_token_usage["date"] != today:
        for provider in ("gemini", "ollama"):
            _daily_token_usage[provider]["input"] = 0
            _daily_token_usage[provider]["output"] = 0
        _daily_token_usage["date"] = today


def _record_tokens(provider: str, input_tokens: int, output_tokens: int) -> None:
    """Thread-safe per-provider token accounting."""
    with _token_lock:
        _reset_daily_usage_if_needed()
        _daily_token_usage[provider]["input"] += int(input_tokens or 0)
        _daily_token_usage[provider]["output"] += int(output_tokens or 0)


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

    _record_tokens(
        "ollama",
        result.get("prompt_eval_count", 0),
        result.get("eval_count", 0),
    )

    duration = result.get("total_duration", 0) / 1e9
    logger.info("Ollama response in %.1fs (%s)", duration, AI_MODEL)

    try:
        return json.loads(result["response"])
    except (KeyError, json.JSONDecodeError) as e:
        logger.warning("Ollama response malformed (missing 'response' key or invalid JSON): %s", e)
        return None


# ---------------------------------------------------------------------------
# Gemini LLM call
# ---------------------------------------------------------------------------

class _GeminiTransportError(Exception):
    """Raised by _call_gemini on HTTP / network failures so the router can
    fall back to Ollama immediately without wasting Gemini retries. Content-level
    failures (malformed envelope, invalid structured JSON) return None instead,
    letting the router retry Gemini up to max_retries for the content issue."""


def _call_gemini(prompt: str) -> Optional[dict]:
    """Call Gemini generateContent. Returns parsed JSON dict.

    Exhaustion signals that latch _gemini_exhausted for the rest of the process:
      - HTTP 401/403 (invalid key / permissions)
      - HTTP 429 whose body matches _GEMINI_EXHAUSTION_MARKERS (credits depleted, etc.)

    Any transport failure (5xx, transient 429, network error, exhausted auth)
    raises _GeminiTransportError so the router can fall straight through to
    Ollama. Content issues (bad envelope, unparseable inner JSON) return None
    so the router can retry the provider.
    """
    if not GEMINI_API_KEY or _gemini_exhausted.is_set():
        raise _GeminiTransportError("gemini unavailable (no key or exhausted)")

    url = f"{GEMINI_HOST}/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "maxOutputTokens": 1024,
            # thinkingBudget:0 disables thinking for 2.5-series models (avoids
            # latency blowup); ignored by 2.0 models.
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }).encode()

    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"},
    )

    try:
        response = urllib.request.urlopen(req, timeout=120)
        raw = response.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        if e.code in (401, 403):
            logger.warning("Gemini auth failed (%d) — latching exhausted flag: %s", e.code, body[:200])
            _gemini_exhausted.set()
            raise _GeminiTransportError(f"auth {e.code}") from e
        if e.code == 429 and _is_permanent_gemini_exhaustion(body):
            logger.warning("Gemini plan exhausted — short-circuiting for process lifetime: %s", body[:200])
            _gemini_exhausted.set()
            raise _GeminiTransportError("429 exhausted") from e
        logger.warning("Gemini transient HTTP %d: %s", e.code, body[:200])
        raise _GeminiTransportError(f"http {e.code}") from e
    except urllib.error.URLError as e:
        logger.warning("Gemini network error: %s", e)
        raise _GeminiTransportError(f"network: {e}") from e

    try:
        envelope = json.loads(raw)
        text = envelope["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(text)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
        logger.warning("Gemini response malformed: %s", e)
        return None

    usage = envelope.get("usageMetadata") or {}
    _record_tokens(
        "gemini",
        usage.get("promptTokenCount", 0),
        usage.get("candidatesTokenCount", 0),
    )
    logger.info("Gemini response OK (%s)", GEMINI_MODEL)
    return parsed


# ---------------------------------------------------------------------------
# Provider router
# ---------------------------------------------------------------------------

def _call_llm(prompt: str, max_retries: int = 3) -> Optional[dict]:
    """Route to Gemini first, fall back to Ollama.

    Gemini is attempted only when AI_PROVIDER == "gemini", a key is set, and the
    exhaustion flag has not been latched. Failure handling within the Gemini
    leg splits by cause:

      * Transport failure (5xx, network, auth, credits depleted)
        -> _GeminiTransportError -> immediate fallback to Ollama; matches
           user intent ("returns an error -> fallback") and avoids burning
           3x the timeout on a stateless server error.
      * Content failure (unparseable envelope, validator rejects the JSON)
        -> None -> retry Gemini up to max_retries, because a re-prompt can
           yield a different, parseable response.

    Ollama keeps its legacy 3-attempt retry loop.
    """
    use_gemini = (
        AI_PROVIDER == "gemini"
        and bool(GEMINI_API_KEY)
        and not _gemini_exhausted.is_set()
    )

    if use_gemini:
        for attempt in range(1, max_retries + 1):
            try:
                result = _call_gemini(prompt)
            except _GeminiTransportError as e:
                logger.warning("Gemini transport failure — falling back to Ollama: %s", e)
                break
            except Exception as e:
                logger.warning("Gemini unexpected error — falling back to Ollama: %s", e)
                break
            if result and _validate_response(result):
                return result
            logger.warning("Gemini invalid response on attempt %d/%d", attempt, max_retries)

    for attempt in range(1, max_retries + 1):
        try:
            result = _call_ollama(prompt)
            if result and _validate_response(result):
                return result
            logger.warning("Ollama invalid response on attempt %d: %s", attempt, result)
        except Exception as e:
            logger.warning("Ollama call failed (attempt %d/%d): %s", attempt, max_retries, e)

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
    """Return today's per-provider token usage stats.

    Shape: {"gemini": {"input": N, "output": N}, "ollama": {"input": N, "output": N}, "date": "YYYY-MM-DD"}.
    """
    with _token_lock:
        _reset_daily_usage_if_needed()
        return {
            "gemini": dict(_daily_token_usage["gemini"]),
            "ollama": dict(_daily_token_usage["ollama"]),
            "date": _daily_token_usage["date"],
        }

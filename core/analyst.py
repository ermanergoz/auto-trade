"""AI Analyst — LLM-powered deep analysis of screener candidates.

Routing: Gemini (primary) -> Ollama (fallback). See _call_llm for the router
and _gemini_exhausted for the process-lifetime short-circuit that mirrors the
Tavily->YFinance exhaustion pattern in core/data.py.
"""

import json
import logging
import threading
import time
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
    GEMINI_API_KEYS,
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
# PERMANENT exhaustion (RPD/quota/credits). Tighter than the Tavily list
# because Gemini emits "Quota exceeded ... requests per minute" for transient
# RPM rate limits — a bare "quota" marker would latch the flag on recoverable
# conditions and block the rotation from advancing to a sibling key.
_GEMINI_EXHAUSTION_MARKERS = (
    "prepayment credits are depleted",
    "usage limit",
    "limit: 0",
    "free tier limit",
    "quota metric exceeded",
    # 2026-04-27 production: tonight's RPD-exhaustion body. Caught by neither
    # of the older markers, so the per-key flag never latched and every
    # candidate burned a wasted Gemini round-trip before falling to Ollama.
    "exceeded your current quota",
    # Google's RPD-specific phrasing returned alongside the
    # GenerateRequestsPerDayPerProjectPerModel-FreeTier metric.
    "requests per day",
)


def _is_permanent_gemini_exhaustion(body_text: str) -> bool:
    lowered = (body_text or "").lower()
    return any(marker in lowered for marker in _GEMINI_EXHAUSTION_MARKERS)


# ---------------------------------------------------------------------------
# Multi-key rotation state — see _call_gemini below.
#
# 3 free-tier keys × 1,000 RPD = 3,000 RPD pool, which fits the bot's ~2,688
# calls/day at the current 15-min scan cadence. Rotation also raises the
# burst RPM ceiling to 45 (3 × 15), comfortably above ~28-candidate scans.
# ---------------------------------------------------------------------------
_gemini_keys: list[str] = list(GEMINI_API_KEYS)  # patchable for tests
_gemini_key_lock = threading.Lock()
_gemini_key_index: int = 0
# Per-key permanent-exhaustion flags. Keys are added lazily on first failure.
# Cleared on process restart (same lifetime as _gemini_exhausted).
_gemini_key_exhausted: dict[str, threading.Event] = {}

# Sleep between the cross-key first-pass and the retry pass when EVERY key
# returns a transient RPM 429 in a single call attempt. Bounded to one sleep
# per call. Set to 30s — Gemini RPM windows are 60s, so a 30s wait gives a
# real chance for the bucket to drain without doubling the candidate's wall
# time. Single-key deployments skip this sleep (no sibling to recover).
_GEMINI_RPM_RETRY_SLEEP = 30


class _GeminiKeyExhausted(Exception):
    """Internal signal: this specific key is permanently exhausted (RPD/auth).
    Mark the per-key flag and try the next key in the rotation."""


class _GeminiKeyTransient(Exception):
    """Internal signal: this key returned a transient RPM 429.
    Try the next key without latching anything."""


def _ensure_key_event(key: str) -> threading.Event:
    """Get-or-create the per-key exhaustion flag (thread-safe)."""
    with _gemini_key_lock:
        ev = _gemini_key_exhausted.get(key)
        if ev is None:
            ev = threading.Event()
            _gemini_key_exhausted[key] = ev
        return ev


def _active_gemini_keys() -> list[str]:
    """Current key list, freshly resolved so test patches take effect.

    Precedence: ``_gemini_keys`` (multi-key list) wins when non-empty.
    Otherwise fall back to ``GEMINI_API_KEY`` (legacy single-key path) so
    existing tests that ``patch("core.analyst.GEMINI_API_KEY", "dummy")``
    keep working unchanged.

    Filters out keys whose RPD-exhaustion flag is set.
    """
    if _gemini_keys:
        keys = list(_gemini_keys)
    elif GEMINI_API_KEY:
        keys = [GEMINI_API_KEY]
    else:
        return []
    return [k for k in keys if not _ensure_key_event(k).is_set()]


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


def _gemini_response_schema() -> dict:
    """JSON schema for Gemini structured output.

    Forces every required field including trade_type — observed in production
    on 2026-04-21 that Gemini 2.5 Flash-Lite was silently omitting trade_type,
    triggering validator-driven retries that burned our free-tier RPM budget.
    The schema makes the field mandatory at the model level.
    """
    action_enum = ["buy", "sell", "hold"] if ALLOW_SHORT_SELLING else ["buy", "hold"]
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": action_enum},
            "confidence": {"type": "integer"},
            "entry_price": {"type": "number"},
            "stop_loss": {"type": "number"},
            "take_profit": {"type": "number"},
            "trade_type": {"type": "string", "enum": ["day", "swing"]},
            "reasoning": {"type": "string"},
        },
        "required": [
            "action", "confidence", "entry_price",
            "stop_loss", "take_profit", "trade_type", "reasoning",
        ],
    }


def _call_gemini_with_key(prompt: str, key: str) -> Optional[dict]:
    """Single Gemini attempt against ONE specific key.

    Maps HTTP 401/403 and RPD-marker 429 to ``_GeminiKeyExhausted`` so the
    rotation can mark the key dead and advance to the next.
    Maps non-marker 429 (RPM) to ``_GeminiKeyTransient`` so the rotation
    can advance without latching anything.
    Other failures (5xx, network) raise ``_GeminiTransportError`` directly
    — those aren't key-specific, so trying a sibling won't help.
    Content errors (bad envelope) return None so the outer router can retry.
    """
    url = f"{GEMINI_HOST}/v1beta/models/{GEMINI_MODEL}:generateContent?key={key}"
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": _gemini_response_schema(),
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
            logger.warning("Gemini auth failed (%d) — marking key exhausted: %s", e.code, body[:200])
            raise _GeminiKeyExhausted(f"auth {e.code}") from e
        if e.code == 429:
            if _is_permanent_gemini_exhaustion(body):
                logger.warning("Gemini RPD/quota exhausted for this key: %s", body[:200])
                raise _GeminiKeyExhausted("429 RPD") from e
            logger.warning("Gemini RPM 429 — will try next key: %s", body[:200])
            raise _GeminiKeyTransient("429 RPM") from e
        logger.warning("Gemini HTTP %d: %s", e.code, body[:200])
        raise _GeminiTransportError(f"http {e.code}") from e
    except urllib.error.URLError as e:
        logger.warning("Gemini network error: %s", e)
        raise _GeminiTransportError(f"network: {e}") from e
    except TimeoutError as e:
        # socket-level read timeout from inside ssl.recv_into. urllib does
        # not wrap this in URLError when the read times out mid-handshake,
        # so the bare TimeoutError would otherwise propagate up and kill
        # the bot (analyst path AND universe sector-classifier path).
        # Treat as transport failure so the router falls back to Ollama.
        logger.warning("Gemini read timeout: %s", e)
        raise _GeminiTransportError(f"timeout: {e}") from e

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


def _call_gemini(prompt: str) -> Optional[dict]:
    """Multi-key rotation wrapper around _call_gemini_with_key.

    Each call advances a round-robin cursor by one position so successive
    candidates spread their requests across the configured keys. Within a
    single call:

      - RPD/auth (permanent) on a key marks ONLY that key's flag and tries
        the next key. When every key is RPD-exhausted, latch the global
        ``_gemini_exhausted`` and raise; the router then short-circuits to
        Ollama for the rest of the process.
      - RPM (transient) on a key advances to the next key without marking
        anything. If every key 429s in pass 0 *and* there is more than one
        key, sleep ``_GEMINI_RPM_RETRY_SLEEP`` once and try a second pass.
        Single-key deployments skip the retry pass — sleeping mid-call when
        there is no sibling to recover is pure latency.

    Content errors (bad envelope, unparseable JSON) return ``None`` so the
    outer router can retry the same provider with a fresh prompt sample.
    """
    global _gemini_key_index

    if _gemini_exhausted.is_set():
        raise _GeminiTransportError("gemini unavailable (all keys exhausted)")

    initial_keys = _active_gemini_keys()
    if not initial_keys:
        _gemini_exhausted.set()
        raise _GeminiTransportError("gemini unavailable (no active keys)")

    # Reserve a starting cursor for this call and advance for the next call.
    with _gemini_key_lock:
        start_cursor = _gemini_key_index
        _gemini_key_index = (start_cursor + 1) % max(len(initial_keys), 1)

    multi_key = len(initial_keys) > 1
    passes = 2 if multi_key else 1

    for pass_num in range(passes):
        if pass_num == 1:
            # Cross-key RPM stampede: every key 429-RPM'd in pass 0. Sleep
            # once and try a second pass. Bounded to one sleep per call.
            time.sleep(_GEMINI_RPM_RETRY_SLEEP)
            logger.info(
                "Cross-key RPM stampede — slept %ds, retrying rotation",
                _GEMINI_RPM_RETRY_SLEEP,
            )

        keys = _active_gemini_keys()
        if not keys:
            _gemini_exhausted.set()
            raise _GeminiTransportError("all keys exhausted")

        # Reorder starting from this call's reserved cursor (modulo the
        # current active-key list — some keys may have been latched in
        # an earlier iteration of this same call).
        offset = start_cursor % len(keys)
        order = keys[offset:] + keys[:offset]

        for key in order:
            try:
                return _call_gemini_with_key(prompt, key)
            except _GeminiKeyExhausted:
                _ensure_key_event(key).set()
                if not _active_gemini_keys():
                    _gemini_exhausted.set()
                    raise _GeminiTransportError("all keys exhausted") from None
                continue
            except _GeminiKeyTransient:
                continue
        # End of this pass — fall through to retry pass if multi_key.

    raise _GeminiTransportError("all keys 429 RPM after retry pass")


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
            rr = reward / risk if risk > 0 else 0
            if risk > 0 and round(rr, 2) < MIN_RISK_REWARD_RATIO:
                logger.warning("BUY R:R %.2f below minimum %.2f", rr, MIN_RISK_REWARD_RATIO)
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
            rr = reward / risk if risk > 0 else 0
            if risk > 0 and round(rr, 2) < MIN_RISK_REWARD_RATIO:
                logger.warning("SELL R:R %.2f below minimum %.2f", rr, MIN_RISK_REWARD_RATIO)
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

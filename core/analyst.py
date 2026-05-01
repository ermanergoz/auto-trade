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
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from config.settings import (
    AI_MODEL,
    AI_PROVIDER,
    AI_CONFIDENCE_THRESHOLD,
    ALLOW_SHORT_SELLING,
    GEMINI_API_KEYS,
    GEMINI_HOST,
    GEMINI_MODEL,
    LLM_TRAFFIC_LOG_ENABLED,
    LOG_DIR,
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
# Multi-key rotation state — see _call_gemini_payload below.
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

    Reads `_gemini_keys` (the live rotation list, populated from
    `GEMINI_API_KEYS` at module load) and filters out keys whose
    RPD-exhaustion flag is set.
    """
    return [k for k in _gemini_keys if not _ensure_key_event(k).is_set()]


# ---------------------------------------------------------------------------
# Diagnostic JSONL traffic log — one record per LLM round-trip.
#
# Captures full prompt + parsed response + raw body + error classification so
# the confidence-distribution analysis (scripts/analyze_confidence.py) can
# correlate Gemini-vs-Ollama outcomes with the actual reasoning text. Best-
# effort by design: a write failure must NEVER break the analyst path.
# ---------------------------------------------------------------------------
_traffic_lock = threading.Lock()


def _log_llm_traffic(record: dict) -> None:
    """Append one LLM round-trip record to logs/llm_traffic_YYYY-MM-DD.jsonl.

    Disabled when LLM_TRAFFIC_LOG_ENABLED is False. Any IO failure (disk
    full, permission denied, LOG_DIR is a regular file) is swallowed —
    diagnostic logging must never block a trade decision.
    """
    if not LLM_TRAFFIC_LOG_ENABLED:
        return
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = LOG_DIR / f"llm_traffic_{datetime.now().date().isoformat()}.jsonl"
        line = json.dumps(record, default=str)
        with _traffic_lock:
            with open(path, "a") as f:
                f.write(line + "\n")
    except Exception as e:
        # Don't even log this at WARNING — the analyst is in the hot path
        # and we don't want a disk failure to spam the trader log every
        # candidate. Debug is enough for forensic post-mortem.
        logger.debug("Traffic log write failed: %s", e)


def _key_suffix(key: str) -> str:
    """Last 4 chars of a Gemini key — enough to identify which key was
    used in a multi-key rotation without leaking the secret."""
    return f"...{key[-4:]}" if key and len(key) >= 4 else "<short>"


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

ANALYSIS_PROMPT = """You are a disciplined stock trader making real money decisions. Your job is to vote on whether to enter this trade. Trade levels (entry, stop-loss, take-profit) are set deterministically by the technical screener — you do not pick them. Focus on whether the overall setup justifies entering.

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
4. NEWS SENTIMENT: Do recent headlines support or contradict the technical signal?
5. ANTI-CHASE RULE: Has the stock already moved more than 5% in the signal direction recently? If yes, you missed the move — say "hold".
6. MACRO/POLITICAL RISK: Do current political, regulatory, or macroeconomic conditions (elections, trade wars, sanctions, Fed policy, sector regulation) create risk or opportunity for this stock? Consider how macro headlines might override or reinforce the technical picture.

## Response Format
Return a JSON object with these exact fields:
- "action": {actions}
- "confidence": integer 0-100
- "trade_type": "day" (close before market end) or "swing" (hold overnight)
- "reasoning": 2-3 sentences covering your checklist assessment

## Discipline Rules
- Default to "hold" unless at least 4 of 6 checklist items are clearly favorable
- Never chase: if the stock already ran >5% toward the signal, say "hold"
- Confidence above 65 only when trend + momentum + volume all align and macro environment is not hostile
- Be honest about uncertainty — a confident "hold" is better than a shaky "buy"{short_rule}"""


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

def _call_ollama(prompt: str, ctx: Optional[dict] = None) -> Optional[dict]:
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

    ctx = ctx or {}
    started = time.monotonic()

    def _record(error: Optional[str], response: Optional[dict] = None,
                response_raw: Optional[str] = None,
                tokens: Optional[dict] = None) -> None:
        _log_llm_traffic({
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "provider": "ollama",
            "kind": ctx.get("kind"),
            "ticker": ctx.get("ticker"),
            "model": AI_MODEL,
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "tokens": tokens,
            "prompt": prompt,
            "response": response,
            "response_raw": response_raw,
            "error": error,
        })

    try:
        response = urllib.request.urlopen(req, timeout=1800)
        raw = response.read()
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        _record(error=f"network: {e}")
        raise

    raw_text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        _record(error="malformed_envelope", response_raw=raw_text)
        logger.warning("Ollama response not JSON: %s", e)
        return None

    # Check for errors BEFORE tracking usage — error responses should
    # not count toward daily token limits
    if "error" in result:
        logger.warning("Ollama error: %s", result["error"])
        _record(error="ollama_error", response_raw=raw_text)
        return None

    in_tokens = result.get("prompt_eval_count", 0) or 0
    out_tokens = result.get("eval_count", 0) or 0
    _record_tokens("ollama", in_tokens, out_tokens)

    duration = result.get("total_duration", 0) / 1e9
    logger.info("Ollama response in %.1fs (%s)", duration, AI_MODEL)

    try:
        parsed = json.loads(result["response"])
    except (KeyError, json.JSONDecodeError) as e:
        logger.warning("Ollama response malformed (missing 'response' key or invalid JSON): %s", e)
        _record(error="malformed_response_field", response_raw=raw_text)
        return None

    _record(
        error=None, response=parsed, response_raw=raw_text,
        tokens={"input": int(in_tokens), "output": int(out_tokens)},
    )
    return parsed


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

    The LLM only votes (buy/hold) + tags confidence/trade_type/reasoning. Trade
    levels are set deterministically by the screener (ATR-based) and carried
    through analyze_candidate; asking the LLM for them invited hallucinated
    chart-readings to propagate into entry/stop/take-profit picks.
    """
    action_enum = ["buy", "sell", "hold"] if ALLOW_SHORT_SELLING else ["buy", "hold"]
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": action_enum},
            "confidence": {"type": "integer"},
            "trade_type": {"type": "string", "enum": ["day", "swing"]},
            "reasoning": {"type": "string"},
        },
        "required": ["action", "confidence", "trade_type", "reasoning"],
    }


def _post_gemini_payload(
    payload: dict, key: str, *, timeout: int = 120,
    ctx: Optional[dict] = None, key_idx: Optional[int] = None,
) -> Optional[dict]:
    """Single Gemini POST against ONE specific key. Generic over payload.

    Used by the rotation in ``_call_gemini_payload`` for both the analyst's
    trading prompt and the universe's sector-classification prompt — only
    the payload (and timeout) differ between them.

    Maps HTTP 401/403 and RPD-marker 429 to ``_GeminiKeyExhausted`` so the
    rotation can mark the key dead and advance to the next.
    Maps non-marker 429 (RPM) to ``_GeminiKeyTransient`` so the rotation
    can advance without latching anything.
    Other failures (5xx, network) raise ``_GeminiTransportError`` directly
    — those aren't key-specific, so trying a sibling won't help.
    Content errors (bad envelope) return None so the outer router can retry.

    ``ctx`` carries optional caller context (ticker, kind="trading"/"sector")
    for the diagnostic JSONL traffic log; ``key_idx`` is the rotation index
    of this key, also for diagnostics.
    """
    url = f"{GEMINI_HOST}/v1beta/models/{GEMINI_MODEL}:generateContent?key={key}"
    encoded = json.dumps(payload).encode()
    prompt_text = _extract_prompt_text(payload)
    ctx = ctx or {}
    started = time.monotonic()

    def _record(error: Optional[str], response: Optional[dict] = None,
                response_raw: Optional[str] = None,
                tokens: Optional[dict] = None) -> None:
        _log_llm_traffic({
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "provider": "gemini",
            "kind": ctx.get("kind"),
            "ticker": ctx.get("ticker"),
            "key_idx": key_idx,
            "key_suffix": _key_suffix(key),
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "tokens": tokens,
            "prompt": prompt_text,
            "response": response,
            "response_raw": response_raw,
            "error": error,
        })

    req = urllib.request.Request(
        url, data=encoded, headers={"Content-Type": "application/json"},
    )

    try:
        response = urllib.request.urlopen(req, timeout=timeout)
        raw = response.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        if e.code in (401, 403):
            logger.warning("Gemini auth failed (%d) — marking key exhausted: %s", e.code, body[:200])
            _record(error=f"auth_{e.code}", response_raw=body)
            raise _GeminiKeyExhausted(f"auth {e.code}") from e
        if e.code == 429:
            if _is_permanent_gemini_exhaustion(body):
                logger.warning("Gemini RPD/quota exhausted for this key: %s", body[:200])
                _record(error="rpd_quota", response_raw=body)
                raise _GeminiKeyExhausted("429 RPD") from e
            logger.warning("Gemini RPM 429 — will try next key: %s", body[:200])
            _record(error="rpm_429", response_raw=body)
            raise _GeminiKeyTransient("429 RPM") from e
        logger.warning("Gemini HTTP %d: %s", e.code, body[:200])
        _record(error=f"http_{e.code}", response_raw=body)
        raise _GeminiTransportError(f"http {e.code}") from e
    except urllib.error.URLError as e:
        logger.warning("Gemini network error: %s", e)
        _record(error=f"network: {e}")
        raise _GeminiTransportError(f"network: {e}") from e
    except TimeoutError as e:
        # socket-level read timeout from inside ssl.recv_into. urllib does
        # not wrap this in URLError when the urlopen call itself times out
        # mid-handshake/read, so the bare TimeoutError would otherwise
        # propagate up and kill the bot mid-scan. Treat as transport error
        # so the router falls back to Ollama.
        logger.warning("Gemini read timeout: %s", e)
        _record(error=f"timeout: {e}")
        raise _GeminiTransportError(f"timeout: {e}") from e

    raw_text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
    try:
        envelope = json.loads(raw)
        text = envelope["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(text)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
        logger.warning("Gemini response malformed: %s", e)
        _record(error="malformed_envelope", response_raw=raw_text)
        return None

    usage = envelope.get("usageMetadata") or {}
    in_tokens = usage.get("promptTokenCount", 0) or 0
    out_tokens = usage.get("candidatesTokenCount", 0) or 0
    _record_tokens("gemini", in_tokens, out_tokens)
    _record(
        error=None, response=parsed, response_raw=raw_text,
        tokens={"input": int(in_tokens), "output": int(out_tokens)},
    )
    logger.info("Gemini response OK (%s)", GEMINI_MODEL)
    return parsed


def _extract_prompt_text(payload: dict) -> str:
    """Pull the user-prompt text out of a Gemini generateContent payload."""
    try:
        return payload["contents"][0]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return ""


def _call_gemini_payload(
    payload: dict, *, timeout: int = 120, ctx: Optional[dict] = None,
) -> Optional[dict]:
    """Multi-key rotation wrapper around ``_post_gemini_payload``.

    Generic entry point for any module that needs to call Gemini through
    the shared rotation/exhaustion infrastructure. The analyst-side
    wrapper ``_call_gemini`` builds the trading payload and delegates here;
    ``core.universe._classify_sector_gemini`` passes its own sector-prompt
    payload (with a shorter timeout).

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

        for i, key in enumerate(order):
            # key_idx is the position in the FULL configured list (not the
            # filtered active list) so the JSONL log records a stable index
            # even as keys get latched within a single call.
            try:
                key_idx = _gemini_keys.index(key)
            except ValueError:
                key_idx = (offset + i) % len(_gemini_keys) if _gemini_keys else None
            try:
                return _post_gemini_payload(
                    payload, key, timeout=timeout, ctx=ctx, key_idx=key_idx,
                )
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


def _call_gemini(prompt: str, ctx: Optional[dict] = None) -> Optional[dict]:
    """Analyst-side Gemini call: build trading payload, dispatch through rotation."""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": _gemini_response_schema(),
            "maxOutputTokens": 1024,
            # thinkingBudget:0 disables thinking for 2.5-series models (avoids
            # latency blowup); ignored by 2.0 models.
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    return _call_gemini_payload(payload, timeout=120, ctx=ctx)


# ---------------------------------------------------------------------------
# Provider router
# ---------------------------------------------------------------------------

def _call_llm(prompt: str, max_retries: int = 3, ctx: Optional[dict] = None) -> Optional[dict]:
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
        and bool(_active_gemini_keys())
        and not _gemini_exhausted.is_set()
    )

    if use_gemini:
        for attempt in range(1, max_retries + 1):
            try:
                result = _call_gemini(prompt, ctx=ctx)
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
            result = _call_ollama(prompt, ctx=ctx)
            if result and _validate_response(result):
                return result
            logger.warning("Ollama invalid response on attempt %d: %s", attempt, result)
        except Exception as e:
            logger.warning("Ollama call failed (attempt %d/%d): %s", attempt, max_retries, e)

    return None


def _validate_response(data: dict) -> bool:
    """Validate LLM response has all required fields with valid values.

    The LLM contract is buy/hold + confidence + trade_type + reasoning. Trade
    levels (entry_price, stop_loss, take_profit) come from the screener's
    deterministic ATR computation in core/screener.py — the LLM never picks
    them. Price-relationship and R:R checks live in core/risk.py, where they
    operate on the screener-set levels.
    """
    required = ["action", "confidence", "reasoning", "trade_type"]
    missing = [f for f in required if f not in data]
    if missing:
        logger.warning("LLM response missing fields: %s. Keys present: %s", missing, list(data.keys()))
        return False

    if data["action"] not in ("buy", "sell", "hold"):
        logger.warning("LLM response invalid action: %r", data["action"])
        return False
    # The short-selling gate lives in risk.check_short_selling, which knows
    # whether the user holds the stock. Rejecting every SELL here would block
    # AI-driven exits on held longs (a legitimate signal shape).
    if data.get("trade_type") not in ("day", "swing"):
        logger.warning("LLM response invalid trade_type: %r", data.get("trade_type"))
        return False
    if not isinstance(data["confidence"], (int, float)) or not (0 <= data["confidence"] <= 100):
        logger.warning("LLM response invalid confidence: %r", data["confidence"])
        return False

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_candidate(
    screener_signal: Signal,
    df: pd.DataFrame,
    news: list[str],
    macro_news: list[str] | None = None,
) -> Optional[Signal]:
    """Analyze a single screener candidate with the LLM.

    Pure function — receives all data, doesn't fetch anything.

    The LLM is restricted to a buy/hold vote plus confidence/trade_type/reasoning.
    Trade levels (entry_price, stop_loss, take_profit) are carried through from
    the screener's deterministic ATR computation, NOT picked by the LLM. This
    blocks hallucinated chart-readings from propagating into the bracket order.

    Returns a Signal if action is buy/sell and confidence >= threshold, else None.
    """
    ticker = screener_signal.ticker
    exchange = screener_signal.exchange
    prompt = _build_prompt(
        ticker, exchange, df, screener_signal.indicator_values, news, macro_news=macro_news,
    )
    result = _call_llm(prompt, ctx={"ticker": ticker, "kind": "trading"})

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
        entry_price=screener_signal.entry_price,
        stop_loss=screener_signal.stop_loss,
        take_profit=screener_signal.take_profit,
        reasoning=result.get("reasoning", ""),
        source="ai",
        exchange=exchange,
        trade_type=trade_type,
        indicator_values=screener_signal.indicator_values,
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
            screener_signal (Signal), df, news
        on_signal: Optional callback called immediately when a signal is approved.
        on_progress: Optional callback(current, total) called after each candidate.
        macro_news: Broad market/political headlines shared across all candidates.

    Returns list of approved Signal objects.
    """
    signals = []
    total = len(candidates)

    for i, c in enumerate(candidates, 1):
        try:
            screener_signal = c["screener_signal"]
        except KeyError as e:
            logger.error("Skipping malformed candidate (missing key %s)", e)
            continue
        ticker = screener_signal.ticker
        logger.info("Analyzing candidate %d/%d: %s", i, total, ticker)
        try:
            signal = analyze_candidate(
                screener_signal=screener_signal,
                df=c["df"],
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

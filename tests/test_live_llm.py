"""Live integration tests for the LLM router.

These tests call the REAL Gemini and Ollama endpoints. They are OPT-IN because
CPU Ollama inference is slow (~7 min for the full file) and Gemini calls cost
real tokens/quota — neither belongs in the default `pytest` run.

Enable by setting RUN_LIVE_LLM_TESTS=1 in the environment:

    RUN_LIVE_LLM_TESTS=1 pytest tests/test_live_llm.py -v

What they verify:
  - Gemini produces a validating response given a real prompt.
  - Ollama produces a validating response given a real prompt.
  - The router (_call_llm) prefers Gemini when available.
  - The router falls back to Ollama when the _gemini_exhausted flag is latched.
"""

import os
import urllib.request
import urllib.error

import pandas as pd
import pytest

from config.settings import AI_PROVIDER, GEMINI_API_KEY, OLLAMA_HOST


LIVE_ENABLED = os.getenv("RUN_LIVE_LLM_TESTS") == "1"


def _ollama_reachable() -> bool:
    """Return True if the local Ollama daemon answers on /api/tags."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=3) as r:
            return r.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return False


skip_no_gemini = pytest.mark.skipif(
    not LIVE_ENABLED or not GEMINI_API_KEY or AI_PROVIDER != "gemini",
    reason="set RUN_LIVE_LLM_TESTS=1 and configure GEMINI_API_KEY / AI_PROVIDER=gemini",
)

skip_no_ollama = pytest.mark.skipif(
    not LIVE_ENABLED or not _ollama_reachable(),
    reason="set RUN_LIVE_LLM_TESTS=1 and ensure Ollama daemon is running",
)


def _make_df(n: int = 30) -> pd.DataFrame:
    """Realistic-looking OHLCV so the prompt-builder has something to work with."""
    dates = pd.date_range("2026-03-01", periods=n, freq="D")
    return pd.DataFrame({
        "open":   [100 + i * 0.3 for i in range(n)],
        "high":   [101 + i * 0.3 for i in range(n)],
        "low":    [ 99 + i * 0.3 for i in range(n)],
        "close":  [100.5 + i * 0.3 for i in range(n)],
        "volume": [1_000_000] * n,
    }, index=dates)


def _analysis_prompt() -> str:
    """Build a real analysis prompt so we exercise the same code path used in production."""
    from core.analyst import _build_prompt
    return _build_prompt(
        ticker="TEST",
        exchange="SMART",
        df=_make_df(),
        indicator_values={"RSI": 42.3, "MACD": 0.15, "MA5": 105.2, "MA20": 103.1},
        news=["Test company announces strong quarterly results"],
        macro_news=["Fed holds rates steady"],
    )


@pytest.fixture
def reset_analyst_state():
    """Clear the Gemini exhaustion flag and per-provider token counters before and after."""
    import core.analyst as _a
    _a._gemini_exhausted.clear()
    for provider in ("gemini", "ollama"):
        _a._daily_token_usage[provider]["input"] = 0
        _a._daily_token_usage[provider]["output"] = 0
    yield
    _a._gemini_exhausted.clear()


@skip_no_gemini
def test_live_gemini_returns_validating_response(reset_analyst_state):
    """Gemini produces a response whose structure passes _validate_response."""
    from core.analyst import _call_gemini, _validate_response, get_daily_token_usage

    result = _call_gemini(_analysis_prompt())

    assert result is not None, "Gemini returned None (content failure)"
    assert _validate_response(result), f"Gemini response failed validator: {result}"
    assert result["action"] in ("buy", "sell", "hold")
    assert 0 <= result["confidence"] <= 100

    usage = get_daily_token_usage()
    assert usage["gemini"]["input"] > 0, "Gemini input tokens not recorded"
    assert usage["gemini"]["output"] > 0, "Gemini output tokens not recorded"
    assert usage["ollama"]["input"] == 0, "Ollama should not have been called"


@skip_no_ollama
def test_live_ollama_returns_validating_response(reset_analyst_state):
    """Ollama (whatever model is configured) produces a validating response."""
    from core.analyst import _call_ollama, _validate_response, get_daily_token_usage

    result = _call_ollama(_analysis_prompt())

    assert result is not None, "Ollama returned None — likely model not pulled or malformed JSON"
    assert _validate_response(result), f"Ollama response failed validator: {result}"
    assert result["action"] in ("buy", "sell", "hold")

    usage = get_daily_token_usage()
    assert usage["ollama"]["input"] > 0, "Ollama input tokens not recorded"


@skip_no_gemini
def test_live_router_prefers_gemini(reset_analyst_state):
    """When Gemini is available, _call_llm uses it and does NOT call Ollama."""
    from core.analyst import _call_llm, _validate_response, get_daily_token_usage

    result = _call_llm(_analysis_prompt())

    assert result is not None, "Router returned None — both providers failed"
    assert _validate_response(result)

    usage = get_daily_token_usage()
    assert usage["gemini"]["input"] > 0, "Gemini was not called"
    assert usage["ollama"]["input"] == 0, (
        f"Router incorrectly fell back to Ollama — gemini={usage['gemini']}, ollama={usage['ollama']}"
    )


@skip_no_ollama
def test_live_router_falls_back_to_ollama_when_gemini_exhausted(reset_analyst_state):
    """With _gemini_exhausted latched, _call_llm skips Gemini entirely and uses Ollama."""
    import core.analyst as _a
    from core.analyst import _call_llm, _validate_response, get_daily_token_usage

    _a._gemini_exhausted.set()

    result = _call_llm(_analysis_prompt())

    assert result is not None, "Router returned None when Gemini exhausted — Ollama fallback broken"
    assert _validate_response(result)

    usage = get_daily_token_usage()
    assert usage["gemini"]["input"] == 0, "Gemini was called despite exhausted flag"
    assert usage["ollama"]["input"] > 0, "Ollama fallback did not fire"

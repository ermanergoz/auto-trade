"""Tests for core/analyst.py."""

import json
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from core.models import Action, Signal
from core.analyst import (
    _build_prompt,
    _validate_response,
    analyze_candidate,
)


def _make_df(n=30):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "open": [100 + i * 0.5 for i in range(n)],
        "high": [101 + i * 0.5 for i in range(n)],
        "low": [99 + i * 0.5 for i in range(n)],
        "close": [100.5 + i * 0.5 for i in range(n)],
        "volume": [1_000_000] * n,
    }, index=dates)


def _make_screener_signal(
    ticker="AAPL",
    exchange="SMART",
    entry=150.0,
    sl=145.0,
    tp=160.0,
    indicators=None,
) -> Signal:
    """Build a screener-source Signal fixture with deterministic ATR-style levels."""
    return Signal(
        ticker=ticker,
        action=Action.BUY,
        confidence=70.0,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        reasoning="screener: triggered indicators",
        source="screener",
        exchange=exchange,
        indicator_values=indicators or {},
    )


class TestPromptBuilding:
    def test_builds_prompt(self):
        df = _make_df()
        prompt = _build_prompt("AAPL", "SMART", df, {"RSI": 25.5}, ["Apple beats earnings"])
        assert "AAPL" in prompt
        assert "RSI" in prompt
        assert "Apple beats earnings" in prompt
        assert "SMART" in prompt

    def test_empty_news(self):
        df = _make_df()
        prompt = _build_prompt("MSFT", "SMART", df, {}, [])
        assert "No recent news" in prompt

    def test_empty_indicators(self):
        df = _make_df()
        prompt = _build_prompt("MSFT", "SMART", df, {}, [])
        assert "No indicator data" in prompt

    def test_macro_news_in_prompt(self):
        df = _make_df()
        prompt = _build_prompt(
            "AAPL", "SMART", df, {"RSI": 25.5},
            ["Apple beats earnings"],
            macro_news=["Fed holds rates steady", "US-China trade talks resume"],
        )
        assert "Fed holds rates steady" in prompt
        assert "US-China trade talks resume" in prompt
        assert "Macro/Political" in prompt

    def test_empty_macro_news(self):
        df = _make_df()
        prompt = _build_prompt("MSFT", "SMART", df, {}, [], macro_news=[])
        assert "No macro/political headlines" in prompt

    def test_none_macro_news_backward_compat(self):
        df = _make_df()
        prompt = _build_prompt("MSFT", "SMART", df, {}, [])
        assert "No macro/political headlines" in prompt

    def test_macro_checklist_item_present(self):
        df = _make_df()
        prompt = _build_prompt("AAPL", "SMART", df, {}, [])
        assert "MACRO/POLITICAL RISK" in prompt
        # Checklist now has 6 items (R:R item removed — LLM no longer picks levels)
        assert "4 of 6" in prompt


class TestValidation:
    def test_valid_response(self):
        data = {
            "action": "buy",
            "confidence": 80,
            "trade_type": "day",
            "reasoning": "Strong bullish pattern",
        }
        assert _validate_response(data) is True

    def test_missing_field(self):
        data = {"action": "buy", "confidence": 80}
        assert _validate_response(data) is False

    def test_invalid_action(self):
        data = {
            "action": "yolo",
            "confidence": 80,
            "trade_type": "day",
            "reasoning": "test",
        }
        assert _validate_response(data) is False

    def test_invalid_trade_type(self):
        data = {
            "action": "buy",
            "confidence": 80,
            "trade_type": "overnight",
            "reasoning": "test",
        }
        assert _validate_response(data) is False

    def test_confidence_out_of_range(self):
        data = {
            "action": "buy",
            "confidence": 150,
            "trade_type": "day",
            "reasoning": "test",
        }
        assert _validate_response(data) is False

    def test_buy_does_not_require_prices(self):
        """LLM no longer picks entry/stop/TP — validator must accept buy/hold
        responses that omit those fields entirely. Levels come from the
        screener's deterministic ATR computation, carried through analyze_candidate.
        """
        data = {
            "action": "buy",
            "confidence": 80,
            "trade_type": "day",
            "reasoning": "Strong bullish setup with macro tailwind",
        }
        assert _validate_response(data) is True


class TestShortSellingGate:
    """The ANALYSIS_PROMPT and validator must gate on ALLOW_SHORT_SELLING.

    When shorts are disabled, the LLM should only be offered 'buy' or 'hold',
    and any SELL that slips through is rejected without a retry.
    """

    def _valid_sell(self):
        return {
            "action": "sell",
            "confidence": 80,
            "trade_type": "day",
            "reasoning": "Bearish setup",
        }

    def _valid_buy(self):
        return {
            "action": "buy",
            "confidence": 80,
            "trade_type": "day",
            "reasoning": "Bullish setup",
        }

    def _valid_hold(self):
        return {
            "action": "hold",
            "confidence": 70,
            "trade_type": "day",
            "reasoning": "Wait",
        }

    # ---- Prompt gating ----

    @patch("core.analyst.ALLOW_SHORT_SELLING", False)
    def test_prompt_excludes_sell_when_shorts_disabled(self):
        df = _make_df()
        prompt = _build_prompt("AAPL", "SMART", df, {"RSI": 50}, ["news"])
        # Dropped "sell" from the action list
        assert '"sell"' not in prompt
        assert '"buy"' in prompt and '"hold"' in prompt
        # Discipline rule makes the constraint explicit
        assert "does not short stocks" in prompt.lower() or "no shorts" in prompt.lower() \
            or "never recommend 'sell'" in prompt.lower()

    @patch("core.analyst.ALLOW_SHORT_SELLING", True)
    def test_prompt_includes_sell_when_shorts_enabled(self):
        df = _make_df()
        prompt = _build_prompt("AAPL", "SMART", df, {"RSI": 50}, ["news"])
        assert '"buy"' in prompt
        assert '"sell"' in prompt
        assert '"hold"' in prompt

    # ---- Validator gating ----

    @patch("core.analyst.ALLOW_SHORT_SELLING", False)
    def test_validator_accepts_sell_when_shorts_disabled(self):
        """The validator is structural only — the short-selling gate lives in
        risk.check_short_selling, which can distinguish a short-open (blocked)
        from closing a held long (allowed). Rejecting SELL here would prevent
        AI-driven exits on held positions even when shorts are disabled.
        """
        assert _validate_response(self._valid_sell()) is True

    @patch("core.analyst.ALLOW_SHORT_SELLING", True)
    def test_validator_accepts_sell_when_shorts_enabled(self):
        assert _validate_response(self._valid_sell()) is True

    @patch("core.analyst.ALLOW_SHORT_SELLING", False)
    def test_validator_accepts_buy_when_shorts_disabled(self):
        assert _validate_response(self._valid_buy()) is True

    @patch("core.analyst.ALLOW_SHORT_SELLING", True)
    def test_validator_accepts_buy_when_shorts_enabled(self):
        assert _validate_response(self._valid_buy()) is True

    @patch("core.analyst.ALLOW_SHORT_SELLING", False)
    def test_validator_accepts_hold_when_shorts_disabled(self):
        assert _validate_response(self._valid_hold()) is True

    @patch("core.analyst.ALLOW_SHORT_SELLING", True)
    def test_validator_accepts_hold_when_shorts_enabled(self):
        assert _validate_response(self._valid_hold()) is True


class TestAnalyzeCandidate:
    @patch("core.analyst._call_llm")
    def test_returns_signal_on_high_confidence(self, mock_llm):
        mock_llm.return_value = {
            "action": "buy",
            "confidence": 85,
            "trade_type": "day",
            "reasoning": "Strong bullish divergence",
        }
        df = _make_df()
        screener_signal = _make_screener_signal(ticker="AAPL", indicators={"RSI": 28})
        signal = analyze_candidate(
            screener_signal=screener_signal, df=df, news=["Good earnings"],
        )
        assert signal is not None
        assert signal.ticker == "AAPL"
        assert signal.action == Action.BUY
        assert signal.confidence == 85
        assert signal.source == "ai"

    @patch("core.analyst._call_llm")
    def test_filters_low_confidence(self, mock_llm):
        mock_llm.return_value = {
            "action": "buy",
            "confidence": 50,
            "trade_type": "day",
            "reasoning": "Weak signal",
        }
        df = _make_df()
        screener_signal = _make_screener_signal()
        signal = analyze_candidate(screener_signal=screener_signal, df=df, news=[])
        assert signal is None

    @patch("core.analyst._call_llm")
    def test_filters_hold(self, mock_llm):
        mock_llm.return_value = {
            "action": "hold",
            "confidence": 90,
            "trade_type": "day",
            "reasoning": "Unclear direction",
        }
        df = _make_df()
        screener_signal = _make_screener_signal()
        signal = analyze_candidate(screener_signal=screener_signal, df=df, news=[])
        assert signal is None

    @patch("core.analyst._call_llm")
    def test_handles_llm_failure(self, mock_llm):
        mock_llm.return_value = None
        df = _make_df()
        screener_signal = _make_screener_signal()
        signal = analyze_candidate(screener_signal=screener_signal, df=df, news=[])
        assert signal is None

    @patch("core.analyst._call_llm")
    def test_macro_news_passed_to_prompt(self, mock_llm):
        mock_llm.return_value = {
            "action": "buy",
            "confidence": 85,
            "trade_type": "day",
            "reasoning": "Strong with favorable macro",
        }
        df = _make_df()
        screener_signal = _make_screener_signal(ticker="AAPL", indicators={"RSI": 28})
        signal = analyze_candidate(
            screener_signal=screener_signal,
            df=df,
            news=["Good earnings"],
            macro_news=["Fed cuts rates"],
        )
        assert signal is not None
        prompt_arg = mock_llm.call_args[0][0]
        assert "Fed cuts rates" in prompt_arg

    @patch("core.analyst._call_llm")
    def test_uses_screener_levels_not_llm_picks(self, mock_llm):
        """analyze_candidate must copy entry_price/stop_loss/take_profit from
        the screener Signal, ignoring whatever the LLM might suggest. Hallucinated
        chart-readings cannot propagate into the bracket order this way.
        """
        # LLM returns only the new contract — no price fields at all
        mock_llm.return_value = {
            "action": "buy",
            "confidence": 85,
            "trade_type": "swing",
            "reasoning": "Macro tailwind plus positive earnings catalyst",
        }
        screener_signal = _make_screener_signal(
            ticker="AAPL",
            entry=200.0,
            sl=180.0,
            tp=240.0,
            indicators={"RSI": 28, "MA5": 198.0, "MA10": 195.0, "MA20": 190.0},
        )
        df = _make_df()
        signal = analyze_candidate(
            screener_signal=screener_signal,
            df=df,
            news=["Apple beats estimates"],
        )
        assert signal is not None
        # Levels come from the screener, not the LLM
        assert signal.entry_price == 200.0
        assert signal.stop_loss == 180.0
        assert signal.take_profit == 240.0
        # LLM still drives action/confidence/trade_type/reasoning
        assert signal.action == Action.BUY
        assert signal.confidence == 85
        assert "Macro tailwind" in signal.reasoning
        # Source remains "ai" because the buy decision is LLM-driven
        assert signal.source == "ai"
        # Ticker/exchange/indicators threaded through from screener_signal
        assert signal.ticker == "AAPL"
        assert signal.exchange == "SMART"
        assert signal.indicator_values["RSI"] == 28


class TestOllamaTimeout:
    """Verify Ollama timeout is reasonable (not 600s)."""

    @patch("core.analyst.urllib.request.urlopen")
    def test_timeout_is_1800s(self, mock_urlopen):
        """Ollama timeout must be 1800s to allow slow local models to complete."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "response": '{"action": "hold", "confidence": 50}',
            "prompt_eval_count": 100,
            "eval_count": 50,
            "total_duration": 1_000_000_000,
        }).encode()
        mock_urlopen.return_value = mock_response

        from core.analyst import _call_ollama
        _call_ollama("test prompt")

        _, kwargs = mock_urlopen.call_args
        assert kwargs.get("timeout") == 1800, (
            f"Ollama timeout should be 1800s, got {kwargs.get('timeout')}"
        )


# ---------------------------------------------------------------------------
# Gemini-primary + Ollama-fallback provider routing
# ---------------------------------------------------------------------------

def _valid_llm_response_text() -> str:
    """Canonical valid JSON response used by both provider mocks.

    Trade levels are NOT in the LLM contract anymore — they come from the
    screener's deterministic ATR computation in core/screener.py.
    """
    return json.dumps({
        "action": "buy",
        "confidence": 85,
        "trade_type": "day",
        "reasoning": "Strong bullish pattern",
    })


def _gemini_success_body(json_text: str, prompt_tokens: int = 100, output_tokens: int = 50) -> bytes:
    """Shape of a successful Gemini generateContent response."""
    return json.dumps({
        "candidates": [{"content": {"parts": [{"text": json_text}]}}],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": output_tokens,
        },
    }).encode()


def _ollama_success_body(json_text: str, prompt_tokens: int = 100, output_tokens: int = 50) -> bytes:
    """Shape of a successful Ollama /api/generate response."""
    return json.dumps({
        "response": json_text,
        "prompt_eval_count": prompt_tokens,
        "eval_count": output_tokens,
        "total_duration": 1_000_000_000,
    }).encode()


def _mock_http_response(body: bytes) -> MagicMock:
    m = MagicMock()
    m.read.return_value = body
    return m


@pytest.fixture
def reset_analyst_state():
    """Clear process-wide Gemini state and zero per-provider token counters.

    Autouse-style in every provider-routing / token test — state must not leak
    across cases because _gemini_exhausted, _gemini_keys, _gemini_key_index,
    _gemini_key_exhausted, and _daily_token_usage are all module globals.

    Critically: also clears `_gemini_keys` and `_gemini_key_exhausted` so the
    user's real ``.env`` cannot leak a real key into tests. Each test opts
    in to a populated rotation list explicitly via
    ``patch("core.analyst._gemini_keys", [...])``.
    """
    from core import analyst as _a
    _a._gemini_exhausted.clear()
    _a._gemini_keys = []
    _a._gemini_key_index = 0
    _a._gemini_key_exhausted.clear()
    _a._daily_token_usage["gemini"]["input"] = 0
    _a._daily_token_usage["gemini"]["output"] = 0
    _a._daily_token_usage["ollama"]["input"] = 0
    _a._daily_token_usage["ollama"]["output"] = 0
    _a._daily_token_usage["date"] = None
    yield
    _a._gemini_exhausted.clear()
    _a._gemini_keys = []
    _a._gemini_key_index = 0
    _a._gemini_key_exhausted.clear()


class TestProviderConfigImports:
    """Config surface and analyst internals exist for the router to wire up."""

    def test_settings_exposes_provider_switches(self):
        from config import settings
        assert hasattr(settings, "AI_PROVIDER")
        assert hasattr(settings, "GEMINI_API_KEYS")
        assert hasattr(settings, "GEMINI_MODEL")
        assert hasattr(settings, "GEMINI_HOST")
        # Default provider is gemini (user wants Gemini-first); Ollama is the fallback.
        assert settings.AI_PROVIDER == "gemini" or settings.AI_PROVIDER == "ollama"
        # A sensible default model for the Gemini key class confirmed working.
        assert settings.GEMINI_MODEL

    def test_analyst_exposes_gemini_internals(self):
        from core import analyst
        assert hasattr(analyst, "_gemini_exhausted")
        assert hasattr(analyst, "_call_gemini")
        assert hasattr(analyst, "_GEMINI_EXHAUSTION_MARKERS")
        assert hasattr(analyst, "_is_permanent_gemini_exhaustion")
        assert hasattr(analyst, "_record_tokens")


class TestTokenUsagePerProvider:
    """Token counters are tracked per provider so we can see which provider did the work."""

    def test_default_shape_has_per_provider_keys(self, reset_analyst_state):
        from core.analyst import get_daily_token_usage
        usage = get_daily_token_usage()
        assert "gemini" in usage
        assert "ollama" in usage
        assert "date" in usage
        assert usage["gemini"] == {"input": 0, "output": 0}
        assert usage["ollama"] == {"input": 0, "output": 0}

    def test_ollama_success_increments_ollama_only(self, reset_analyst_state):
        from core.analyst import _call_ollama, get_daily_token_usage
        with patch("core.analyst.urllib.request.urlopen") as mu:
            mu.return_value = _mock_http_response(_ollama_success_body(
                _valid_llm_response_text(), prompt_tokens=111, output_tokens=77,
            ))
            _call_ollama("prompt")
        usage = get_daily_token_usage()
        assert usage["ollama"] == {"input": 111, "output": 77}
        assert usage["gemini"] == {"input": 0, "output": 0}

    def test_counters_reset_on_new_day(self, reset_analyst_state):
        from core import analyst
        from core.analyst import get_daily_token_usage
        analyst._daily_token_usage["gemini"]["input"] = 500
        analyst._daily_token_usage["ollama"]["output"] = 200
        analyst._daily_token_usage["date"] = "1999-01-01"  # stale
        usage = get_daily_token_usage()
        assert usage["gemini"] == {"input": 0, "output": 0}
        assert usage["ollama"] == {"input": 0, "output": 0}
        assert usage["date"] != "1999-01-01"


class TestGeminiCallContract:
    """_call_gemini classifies HTTP errors, parses success, records tokens, never retries internally."""

    def _make_http_error(self, code: int, body: bytes, reason: str = ""):
        import io
        import urllib.error
        return urllib.error.HTTPError(
            url="https://generativelanguage.googleapis.com/fake",
            code=code,
            msg=reason or "error",
            hdrs=None,
            fp=io.BytesIO(body),
        )

    def test_gemini_success_parses_nested_json_and_records_tokens(self, reset_analyst_state):
        from core.analyst import _call_gemini, get_daily_token_usage
        with patch("core.analyst.urllib.request.urlopen") as mu, \
             patch("core.analyst._gemini_keys", ["dummy"]):
            mu.return_value = _mock_http_response(_gemini_success_body(
                _valid_llm_response_text(), prompt_tokens=120, output_tokens=80,
            ))
            result = _call_gemini("prompt")
        assert result is not None
        assert result["action"] == "buy"
        assert result["confidence"] == 85
        usage = get_daily_token_usage()
        assert usage["gemini"] == {"input": 120, "output": 80}
        assert usage["ollama"] == {"input": 0, "output": 0}

    def test_gemini_missing_api_key_raises_transport_error_without_http(self, reset_analyst_state):
        from core.analyst import _call_gemini, _GeminiTransportError
        with patch("core.analyst.urllib.request.urlopen") as mu, \
             patch("core.analyst._gemini_keys", []):
            with pytest.raises(_GeminiTransportError):
                _call_gemini("prompt")
        assert mu.call_count == 0

    def test_gemini_401_sets_exhausted_flag_and_raises(self, reset_analyst_state):
        from core import analyst
        from core.analyst import _call_gemini, _GeminiTransportError
        with patch("core.analyst.urllib.request.urlopen") as mu, \
             patch("core.analyst._gemini_keys", ["dummy"]):
            mu.side_effect = self._make_http_error(401, b'{"error":"invalid key"}', "Unauthorized")
            with pytest.raises(_GeminiTransportError):
                _call_gemini("prompt")
        assert analyst._gemini_exhausted.is_set() is True

    def test_gemini_403_sets_exhausted_flag_and_raises(self, reset_analyst_state):
        from core import analyst
        from core.analyst import _call_gemini, _GeminiTransportError
        with patch("core.analyst.urllib.request.urlopen") as mu, \
             patch("core.analyst._gemini_keys", ["dummy"]):
            mu.side_effect = self._make_http_error(403, b'{"error":"forbidden"}', "Forbidden")
            with pytest.raises(_GeminiTransportError):
                _call_gemini("prompt")
        assert analyst._gemini_exhausted.is_set() is True

    def test_gemini_429_credits_depleted_sets_flag_and_raises(self, reset_analyst_state):
        from core import analyst
        from core.analyst import _call_gemini, _GeminiTransportError
        body = b'{"error":{"message":"Your prepayment credits are depleted."}}'
        with patch("core.analyst.urllib.request.urlopen") as mu, \
             patch("core.analyst._gemini_keys", ["dummy"]):
            mu.side_effect = self._make_http_error(429, body, "Too Many Requests")
            with pytest.raises(_GeminiTransportError):
                _call_gemini("prompt")
        assert analyst._gemini_exhausted.is_set() is True

    def test_gemini_429_transient_rate_limit_raises_without_latching_flag(self, reset_analyst_state):
        from core import analyst
        from core.analyst import _call_gemini, _GeminiTransportError
        body = b'{"error":{"message":"Quota exceeded per minute; retry in 30s"}}'
        with patch("core.analyst.urllib.request.urlopen") as mu, \
             patch("core.analyst._gemini_keys", ["dummy"]):
            mu.side_effect = self._make_http_error(429, body, "Too Many Requests")
            with pytest.raises(_GeminiTransportError):
                _call_gemini("prompt")
        assert analyst._gemini_exhausted.is_set() is False

    def test_gemini_503_raises_transport_error_without_latching(self, reset_analyst_state):
        from core import analyst
        from core.analyst import _call_gemini, _GeminiTransportError
        with patch("core.analyst.urllib.request.urlopen") as mu, \
             patch("core.analyst._gemini_keys", ["dummy"]):
            mu.side_effect = self._make_http_error(503, b"service unavailable", "Service Unavailable")
            with pytest.raises(_GeminiTransportError):
                _call_gemini("prompt")
        assert analyst._gemini_exhausted.is_set() is False

    def test_gemini_network_error_raises_transport_error_without_latching(self, reset_analyst_state):
        import urllib.error
        from core import analyst
        from core.analyst import _call_gemini, _GeminiTransportError
        with patch("core.analyst.urllib.request.urlopen") as mu, \
             patch("core.analyst._gemini_keys", ["dummy"]):
            mu.side_effect = urllib.error.URLError("connection refused")
            with pytest.raises(_GeminiTransportError):
                _call_gemini("prompt")
        assert analyst._gemini_exhausted.is_set() is False

    def test_gemini_socket_timeout_raises_transport_error_not_bare_TimeoutError(self, reset_analyst_state):
        """A read timeout from inside ssl/socket must not bubble out raw.

        Production crash 2026-05-01: urllib.request.urlopen() succeeded but
        the subsequent read() raised `TimeoutError: The read operation
        timed out` from ssl.recv_into. _post_gemini_payload caught
        HTTPError + URLError but not TimeoutError, so the bare exception
        propagated up through _call_gemini_payload → universe sector
        classifier → build_universe → run_scan_cycle, killing the bot
        mid-scan. Must be classified as transport failure so the router
        falls back to Ollama (and doesn't latch _gemini_exhausted).
        """
        from core import analyst
        from core.analyst import _call_gemini, _GeminiTransportError
        with patch("core.analyst.urllib.request.urlopen") as mu, \
             patch("core.analyst._gemini_keys", ["dummy"]):
            mu.side_effect = TimeoutError("The read operation timed out")
            with pytest.raises(_GeminiTransportError):
                _call_gemini("prompt")
        assert analyst._gemini_exhausted.is_set() is False

    def test_gemini_malformed_envelope_returns_none_for_router_retry(self, reset_analyst_state):
        """Content errors (bad envelope) return None so the router can retry Gemini."""
        from core import analyst
        from core.analyst import _call_gemini
        with patch("core.analyst.urllib.request.urlopen") as mu, \
             patch("core.analyst._gemini_keys", ["dummy"]):
            mu.return_value = _mock_http_response(b"<html>Gateway Timeout</html>")
            result = _call_gemini("prompt")
        assert result is None
        assert analyst._gemini_exhausted.is_set() is False

    def test_gemini_short_circuits_when_already_exhausted(self, reset_analyst_state):
        from core import analyst
        from core.analyst import _call_gemini, _GeminiTransportError
        analyst._gemini_exhausted.set()
        with patch("core.analyst.urllib.request.urlopen") as mu, \
             patch("core.analyst._gemini_keys", ["dummy"]):
            with pytest.raises(_GeminiTransportError):
                _call_gemini("prompt")
        assert mu.call_count == 0

    # ---- responseSchema contract ----
    #
    # Seen in production 2026-04-21: Gemini 2.5 Flash-Lite omitted trade_type
    # on the majority of calls, causing validator retries that burned our
    # free-tier RPM budget. Fix: send a responseSchema so the model is forced
    # to emit every required field.

    def _extract_payload(self, mock_urlopen):
        """Parse the JSON body of the Request passed into urlopen."""
        request = mock_urlopen.call_args[0][0]
        return json.loads(request.data.decode())

    def test_gemini_payload_includes_response_schema(self, reset_analyst_state):
        """generationConfig must carry a responseSchema with required fields.

        The LLM contract is buy/hold + confidence + trade_type + reasoning.
        Trade levels are screener-deterministic (core/screener.py) and not
        requested from the model, so entry_price/stop_loss/take_profit are
        intentionally absent from the schema.
        """
        from core.analyst import _call_gemini
        with patch("core.analyst.urllib.request.urlopen") as mu, \
             patch("core.analyst._gemini_keys", ["dummy"]):
            mu.return_value = _mock_http_response(_gemini_success_body(
                _valid_llm_response_text(),
            ))
            _call_gemini("prompt")

        payload = self._extract_payload(mu)
        gc = payload["generationConfig"]
        assert "responseSchema" in gc, "Gemini payload missing responseSchema"
        schema = gc["responseSchema"]
        assert schema["type"].lower() == "object"
        required = schema.get("required", [])
        for field in ("action", "confidence", "trade_type", "reasoning"):
            assert field in required, f"{field} not in schema.required: {required}"
        # Price fields must NOT be in the schema — the LLM no longer picks levels
        for field in ("entry_price", "stop_loss", "take_profit"):
            assert field not in required, (
                f"{field} should be absent from schema after the analyst-veto refactor"
            )
            assert field not in schema.get("properties", {}), (
                f"{field} should be absent from schema.properties"
            )

    def test_gemini_schema_trade_type_enum_is_day_or_swing(self, reset_analyst_state):
        from core.analyst import _call_gemini
        with patch("core.analyst.urllib.request.urlopen") as mu, \
             patch("core.analyst._gemini_keys", ["dummy"]):
            mu.return_value = _mock_http_response(_gemini_success_body(
                _valid_llm_response_text(),
            ))
            _call_gemini("prompt")

        payload = self._extract_payload(mu)
        tt = payload["generationConfig"]["responseSchema"]["properties"]["trade_type"]
        assert set(tt["enum"]) == {"day", "swing"}

    def test_gemini_schema_action_enum_excludes_sell_when_shorts_disabled(
        self, reset_analyst_state,
    ):
        """When shorts are disabled, the schema must not offer 'sell' as an action."""
        from core.analyst import _call_gemini
        with patch("core.analyst.urllib.request.urlopen") as mu, \
             patch("core.analyst._gemini_keys", ["dummy"]), \
             patch("core.analyst.ALLOW_SHORT_SELLING", False):
            mu.return_value = _mock_http_response(_gemini_success_body(
                _valid_llm_response_text(),
            ))
            _call_gemini("prompt")

        payload = self._extract_payload(mu)
        action_enum = payload["generationConfig"]["responseSchema"]["properties"]["action"]["enum"]
        assert "sell" not in action_enum
        assert set(action_enum) == {"buy", "hold"}

    def test_gemini_schema_action_enum_includes_sell_when_shorts_enabled(
        self, reset_analyst_state,
    ):
        from core.analyst import _call_gemini
        with patch("core.analyst.urllib.request.urlopen") as mu, \
             patch("core.analyst._gemini_keys", ["dummy"]), \
             patch("core.analyst.ALLOW_SHORT_SELLING", True):
            mu.return_value = _mock_http_response(_gemini_success_body(
                _valid_llm_response_text(),
            ))
            _call_gemini("prompt")

        payload = self._extract_payload(mu)
        action_enum = payload["generationConfig"]["responseSchema"]["properties"]["action"]["enum"]
        assert set(action_enum) == {"buy", "sell", "hold"}


class TestGeminiExhaustionMarkers:
    """Marker-based classification of permanent (RPD/quota/auth) vs transient
    (RPM/per-minute) Gemini 429 errors.

    The 2026-04-27 production bug: tonight's response body
    "You exceeded your current quota" matched no marker, so the per-key
    exhaustion flag never latched and every candidate burned a wasted
    Gemini round-trip before falling to the slow Ollama path.

    The fix is to extend `_GEMINI_EXHAUSTION_MARKERS` to capture RPD/quota
    phrasing — but NOT to also catch per-minute (RPM) phrasing, which is a
    transient rate limit and should let the next key in the rotation try.
    """

    def test_existing_credits_depleted_marker_still_matches(self):
        from core.analyst import _is_permanent_gemini_exhaustion
        assert _is_permanent_gemini_exhaustion(
            "Your prepayment credits are depleted."
        ) is True

    def test_existing_free_tier_limit_marker_still_matches(self):
        from core.analyst import _is_permanent_gemini_exhaustion
        assert _is_permanent_gemini_exhaustion(
            "Free tier limit exceeded."
        ) is True

    def test_rpd_message_exceeded_your_current_quota_matches(self):
        """The exact 2026-04-27 production error must classify as permanent."""
        from core.analyst import _is_permanent_gemini_exhaustion
        body = (
            '{"error":{"code":429,"message":"You exceeded your current quota, '
            'please check your plan and billing details."}}'
        )
        assert _is_permanent_gemini_exhaustion(body) is True

    def test_rpd_phrasing_requests_per_day_matches(self):
        """Google's RPD-specific phrasing must classify as permanent."""
        from core.analyst import _is_permanent_gemini_exhaustion
        body = (
            "Quota exceeded for quota metric 'GenerateRequestsPerDayPerProjectPerModel-FreeTier' "
            "and limit 'requests per day' for service ..."
        )
        assert _is_permanent_gemini_exhaustion(body) is True

    def test_rpm_per_minute_message_does_NOT_match(self):
        """Per-minute rate limit is recoverable; must not latch."""
        from core.analyst import _is_permanent_gemini_exhaustion
        body = (
            "Quota exceeded for quota metric 'requests' and limit "
            "'requests per minute' for service ..."
        )
        assert _is_permanent_gemini_exhaustion(body) is False

    def test_rpm_alternate_phrasing_does_NOT_match(self):
        from core.analyst import _is_permanent_gemini_exhaustion
        assert _is_permanent_gemini_exhaustion(
            "Rate limit exceeded; retry in 30s"
        ) is False

    def test_empty_body_does_not_match(self):
        from core.analyst import _is_permanent_gemini_exhaustion
        assert _is_permanent_gemini_exhaustion("") is False
        assert _is_permanent_gemini_exhaustion(None) is False

    def test_case_insensitive(self):
        from core.analyst import _is_permanent_gemini_exhaustion
        assert _is_permanent_gemini_exhaustion(
            "YOU EXCEEDED YOUR CURRENT QUOTA"
        ) is True


class TestGeminiAPIKeysParser:
    """config/settings.py exposes a pure parser for the GEMINI_API_KEYS env var.

    Comma-separated, whitespace-trimmed, empty segments dropped. A single
    key is just a one-element rotation list — same code path as multi-key.
    """

    def test_parses_comma_separated(self):
        from config.settings import _parse_gemini_keys
        assert _parse_gemini_keys("k1,k2,k3") == ["k1", "k2", "k3"]

    def test_strips_whitespace(self):
        from config.settings import _parse_gemini_keys
        assert _parse_gemini_keys("k1, k2 ,  k3 ") == ["k1", "k2", "k3"]

    def test_filters_empty_segments(self):
        from config.settings import _parse_gemini_keys
        assert _parse_gemini_keys("k1,,k2,") == ["k1", "k2"]

    def test_single_key_returns_one_element_list(self):
        from config.settings import _parse_gemini_keys
        assert _parse_gemini_keys("just_one_key") == ["just_one_key"]

    def test_empty_returns_empty_list(self):
        from config.settings import _parse_gemini_keys
        assert _parse_gemini_keys("") == []


class TestGeminiKeyRotation:
    """Multi-key rotation:

      - Round-robin per call: each successive _call_gemini uses the next key.
      - RPD/auth 429 marks ONLY that key's exhausted flag and advances to
        the next key. The global _gemini_exhausted is set only when ALL keys
        are exhausted, which is what gates the Ollama fallback in _call_llm.
      - RPM 429 (transient) advances to the next key WITHOUT latching any
        flag — that key may recover on its next attempt.
      - Cross-key RPM stampede (every key 429s in pass 0) sleeps once and
        retries. After the second pass also fails, raise without latching.
      - Single-key (one entry in GEMINI_API_KEYS) keeps the pre-rotation
        behavior — no extra sleep, no retry pass.
    """

    @staticmethod
    def _make_http_error(code, body, reason="error"):
        import io
        import urllib.error
        return urllib.error.HTTPError(
            url="https://generativelanguage.googleapis.com/fake",
            code=code, msg=reason, hdrs=None, fp=io.BytesIO(body),
        )

    @staticmethod
    def _success_response():
        return _mock_http_response(_gemini_success_body(_valid_llm_response_text()))

    @staticmethod
    def _key_in_request(call_args) -> str:
        """Extract the ?key=... value from a urlopen Request call."""
        req = call_args[0][0]
        url = req.full_url
        if "key=" not in url:
            return ""
        return url.split("key=")[-1].split("&")[0]

    def test_round_robin_advances_per_call(self, reset_analyst_state):
        """3 keys, 3 successive calls — each call uses the next key in order."""
        from core.analyst import _call_gemini
        with patch("core.analyst._gemini_keys", ["KEY_A", "KEY_B", "KEY_C"]), \
             patch("core.analyst.urllib.request.urlopen") as mu:
            mu.return_value = self._success_response()
            for _ in range(3):
                _call_gemini("prompt")
                # urlopen consumes the response body, reset for the next call
                mu.return_value = self._success_response()
        keys_used = [self._key_in_request(c) for c in mu.call_args_list]
        assert keys_used == ["KEY_A", "KEY_B", "KEY_C"], (
            f"Round-robin must advance one step per call. Got {keys_used}"
        )

    def test_rpd_429_marks_only_that_key_and_falls_through(self, reset_analyst_state):
        """key A returns RPD-marker 429 → mark A → try B → return success."""
        from core import analyst
        from core.analyst import _call_gemini
        rpd_body = (
            b'{"error":{"code":429,"message":"You exceeded your current quota."}}'
        )
        with patch("core.analyst._gemini_keys", ["KEY_A", "KEY_B"]), \
             patch("core.analyst.urllib.request.urlopen") as mu:
            mu.side_effect = [
                self._make_http_error(429, rpd_body, "Too Many Requests"),
                self._success_response(),
            ]
            result = _call_gemini("prompt")
        assert result is not None and result.get("action") == "buy"
        assert analyst._gemini_key_exhausted.get("KEY_A").is_set() is True
        # KEY_B must not be marked
        flag_b = analyst._gemini_key_exhausted.get("KEY_B")
        assert flag_b is None or not flag_b.is_set()
        # Global flag must not be set when at least one key remains
        assert analyst._gemini_exhausted.is_set() is False

    def test_all_keys_rpd_exhausted_latches_global_flag(self, reset_analyst_state):
        """When every key reports RPD-marker 429, latch the global flag so the
        router falls back to Ollama for the rest of the process.
        """
        from core import analyst
        from core.analyst import _call_gemini, _GeminiTransportError
        rpd_body = (
            b'{"error":{"code":429,"message":"You exceeded your current quota."}}'
        )
        with patch("core.analyst._gemini_keys", ["KEY_A", "KEY_B"]), \
             patch("core.analyst.urllib.request.urlopen") as mu:
            mu.side_effect = [
                self._make_http_error(429, rpd_body, "Too Many Requests"),
                self._make_http_error(429, rpd_body, "Too Many Requests"),
            ]
            with pytest.raises(_GeminiTransportError):
                _call_gemini("prompt")
        assert analyst._gemini_exhausted.is_set() is True

    def test_rpm_429_tries_next_key_without_latching(self, reset_analyst_state):
        """key A returns RPM 429 (transient) → advance → key B succeeds.
        No flags should be latched; both keys remain available for next call.
        """
        from core import analyst
        from core.analyst import _call_gemini
        rpm_body = (
            b'{"error":{"code":429,"message":"Quota exceeded for quota metric '
            b"'requests' and limit 'requests per minute'.\"}}"
        )
        with patch("core.analyst._gemini_keys", ["KEY_A", "KEY_B"]), \
             patch("core.analyst.urllib.request.urlopen") as mu, \
             patch("core.analyst.time.sleep") as ms:
            mu.side_effect = [
                self._make_http_error(429, rpm_body, "Too Many Requests"),
                self._success_response(),
            ]
            result = _call_gemini("prompt")
        assert result is not None and result.get("action") == "buy"
        # No flag latched on either key
        for k in ("KEY_A", "KEY_B"):
            flag = analyst._gemini_key_exhausted.get(k)
            assert flag is None or not flag.is_set(), (
                f"{k} flag should not be set after a transient RPM 429"
            )
        assert analyst._gemini_exhausted.is_set() is False
        # No sleep needed when the next key in the rotation succeeded
        ms.assert_not_called()

    def test_cross_key_rpm_stampede_sleeps_then_retries(self, reset_analyst_state):
        """Both keys 429-RPM in pass 0 → sleep once → both 429-RPM in pass 1
        → raise transport error; no flags latched (RPM is recoverable, just
        slower than the candidate's allotted wall time).
        """
        from core import analyst
        from core.analyst import _call_gemini, _GeminiTransportError
        rpm_body = (
            b'{"error":{"code":429,"message":"requests per minute"}}'
        )
        with patch("core.analyst._gemini_keys", ["KEY_A", "KEY_B"]), \
             patch("core.analyst.urllib.request.urlopen") as mu, \
             patch("core.analyst.time.sleep") as ms:
            # Pass 0: A → 429, B → 429.  Pass 1 (after sleep): A → 429, B → 429.
            mu.side_effect = [
                self._make_http_error(429, rpm_body, "Too Many Requests"),
                self._make_http_error(429, rpm_body, "Too Many Requests"),
                self._make_http_error(429, rpm_body, "Too Many Requests"),
                self._make_http_error(429, rpm_body, "Too Many Requests"),
            ]
            with pytest.raises(_GeminiTransportError):
                _call_gemini("prompt")
        # Exactly one sleep between the two passes
        ms.assert_called_once()
        sleep_seconds = ms.call_args[0][0]
        assert sleep_seconds >= 10, (
            f"Cross-key RPM stampede should sleep >=10s, got {sleep_seconds}"
        )
        # Flag must NOT latch on transient RPM, even after retry pass failed
        assert analyst._gemini_exhausted.is_set() is False

    def test_single_key_skips_retry_pass_to_avoid_30s_sleep(self, reset_analyst_state):
        """When only one key is configured, the RPM-stampede retry pass MUST
        be skipped — sleeping 30s mid-call when there's no alternate key to
        try is pure latency. Maintains pre-rotation single-key behavior.
        """
        from core import analyst
        from core.analyst import _call_gemini, _GeminiTransportError
        rpm_body = b'{"error":{"message":"requests per minute"}}'
        with patch("core.analyst._gemini_keys", ["ONLY_KEY"]), \
             patch("core.analyst.urllib.request.urlopen") as mu, \
             patch("core.analyst.time.sleep") as ms:
            mu.side_effect = self._make_http_error(429, rpm_body, "Too Many Requests")
            with pytest.raises(_GeminiTransportError):
                _call_gemini("prompt")
        # Exactly ONE HTTP attempt, no sleep
        assert mu.call_count == 1
        ms.assert_not_called()
        assert analyst._gemini_exhausted.is_set() is False


class TestLLMProviderRouting:
    """_call_llm routes: Gemini first (if enabled), Ollama as fallback."""

    def test_gemini_success_returns_signal_without_touching_ollama(self, reset_analyst_state):
        from core.analyst import analyze_candidate
        df = _make_df()
        with patch("core.analyst.AI_PROVIDER", "gemini"), \
             patch("core.analyst._gemini_keys", ["dummy"]), \
             patch("core.analyst._call_gemini") as mock_gemini, \
             patch("core.analyst._call_ollama") as mock_ollama:
            mock_gemini.return_value = json.loads(_valid_llm_response_text())
            signal = analyze_candidate(_make_screener_signal(ticker="AAPL", indicators={"RSI": 28}), df, ["news"])
        assert signal is not None
        assert signal.ticker == "AAPL"
        assert mock_ollama.call_count == 0
        assert mock_gemini.call_count >= 1

    def test_ai_provider_ollama_never_calls_gemini(self, reset_analyst_state):
        from core.analyst import analyze_candidate
        df = _make_df()
        with patch("core.analyst.AI_PROVIDER", "ollama"), \
             patch("core.analyst._gemini_keys", ["real-looking-key"]), \
             patch("core.analyst._call_gemini") as mock_gemini, \
             patch("core.analyst._call_ollama") as mock_ollama:
            mock_ollama.return_value = json.loads(_valid_llm_response_text())
            signal = analyze_candidate(_make_screener_signal(ticker="AAPL", indicators={"RSI": 28}), df, ["news"])
        assert signal is not None
        assert mock_gemini.call_count == 0
        assert mock_ollama.call_count >= 1

    def test_missing_gemini_api_key_goes_straight_to_ollama_no_gemini_http(self, reset_analyst_state):
        from core.analyst import analyze_candidate
        df = _make_df()
        with patch("core.analyst.AI_PROVIDER", "gemini"), \
             patch("core.analyst._gemini_keys", []), \
             patch("core.analyst.urllib.request.urlopen") as mu:
            mu.return_value = _mock_http_response(_ollama_success_body(_valid_llm_response_text()))
            signal = analyze_candidate(_make_screener_signal(ticker="AAPL", indicators={"RSI": 28}), df, ["news"])
        assert signal is not None
        urls = [str(call.args[0].full_url) for call in mu.call_args_list]
        assert not any("generativelanguage" in u for u in urls)

    def test_gemini_transient_503_falls_back_to_ollama_this_call(self, reset_analyst_state):
        from core import analyst
        from core.analyst import analyze_candidate
        df = _make_df()
        import io
        import urllib.error
        http503 = urllib.error.HTTPError(
            url="https://generativelanguage.googleapis.com/x",
            code=503, msg="high demand", hdrs=None,
            fp=io.BytesIO(b"service unavailable"),
        )
        with patch("core.analyst.AI_PROVIDER", "gemini"), \
             patch("core.analyst._gemini_keys", ["dummy"]), \
             patch("core.analyst.urllib.request.urlopen") as mu:
            mu.side_effect = [
                http503,
                _mock_http_response(_ollama_success_body(_valid_llm_response_text())),
            ]
            signal = analyze_candidate(_make_screener_signal(ticker="AAPL", indicators={"RSI": 28}), df, ["news"])
        assert signal is not None
        assert analyst._gemini_exhausted.is_set() is False
        # First call Gemini, second call Ollama.
        urls = [str(call.args[0].full_url) for call in mu.call_args_list]
        assert "generativelanguage" in urls[0]
        assert "11434" in urls[1] or "localhost" in urls[1]

    def test_gemini_permanent_exhaustion_latches_flag_and_skips_gemini_next_call(self, reset_analyst_state):
        from core import analyst
        from core.analyst import analyze_candidate
        df = _make_df()
        import io
        import urllib.error
        http429_permanent = urllib.error.HTTPError(
            url="https://generativelanguage.googleapis.com/x",
            code=429, msg="Too Many Requests", hdrs=None,
            fp=io.BytesIO(b'{"error":{"message":"Your prepayment credits are depleted."}}'),
        )
        with patch("core.analyst.AI_PROVIDER", "gemini"), \
             patch("core.analyst._gemini_keys", ["dummy"]), \
             patch("core.analyst.urllib.request.urlopen") as mu:
            mu.side_effect = [
                http429_permanent,
                _mock_http_response(_ollama_success_body(_valid_llm_response_text())),
            ]
            signal = analyze_candidate(_make_screener_signal(ticker="AAPL", indicators={"RSI": 28}), df, ["news"])
        assert signal is not None
        assert analyst._gemini_exhausted.is_set() is True

        # Second call: Gemini must NOT be hit — flag is latched.
        with patch("core.analyst.AI_PROVIDER", "gemini"), \
             patch("core.analyst._gemini_keys", ["dummy"]), \
             patch("core.analyst.urllib.request.urlopen") as mu2:
            mu2.return_value = _mock_http_response(_ollama_success_body(_valid_llm_response_text()))
            signal2 = analyze_candidate(_make_screener_signal(ticker="MSFT", indicators={"RSI": 28}), df, ["news"])
        assert signal2 is not None
        urls2 = [str(call.args[0].full_url) for call in mu2.call_args_list]
        assert not any("generativelanguage" in u for u in urls2)

    def test_gemini_transient_failure_does_not_latch_flag_retries_next_call(self, reset_analyst_state):
        """After a transient Gemini 503 falls back to Ollama once, the NEXT call retries Gemini."""
        from core import analyst
        from core.analyst import analyze_candidate
        df = _make_df()
        import io
        import urllib.error

        # Call 1: Gemini 503 → Ollama success.
        http503 = urllib.error.HTTPError(
            url="https://generativelanguage.googleapis.com/x",
            code=503, msg="high demand", hdrs=None,
            fp=io.BytesIO(b"busy"),
        )
        with patch("core.analyst.AI_PROVIDER", "gemini"), \
             patch("core.analyst._gemini_keys", ["dummy"]), \
             patch("core.analyst.urllib.request.urlopen") as mu:
            mu.side_effect = [
                http503,
                _mock_http_response(_ollama_success_body(_valid_llm_response_text())),
            ]
            s1 = analyze_candidate(_make_screener_signal(ticker="AAPL", indicators={"RSI": 28}), df, ["news"])
        assert s1 is not None
        assert analyst._gemini_exhausted.is_set() is False

        # Call 2: Gemini succeeds on first try, Ollama must not be touched.
        with patch("core.analyst.AI_PROVIDER", "gemini"), \
             patch("core.analyst._gemini_keys", ["dummy"]), \
             patch("core.analyst.urllib.request.urlopen") as mu2:
            mu2.return_value = _mock_http_response(_gemini_success_body(_valid_llm_response_text()))
            s2 = analyze_candidate(_make_screener_signal(ticker="MSFT", indicators={"RSI": 28}), df, ["news"])
        assert s2 is not None
        urls2 = [str(call.args[0].full_url) for call in mu2.call_args_list]
        assert any("generativelanguage" in u for u in urls2)
        assert not any("11434" in u for u in urls2)

    def test_gemini_invalid_response_retries_then_falls_back(self, reset_analyst_state):
        """When _call_gemini returns None 3 times (retry limit), Ollama takes over."""
        from core.analyst import analyze_candidate
        df = _make_df()
        with patch("core.analyst.AI_PROVIDER", "gemini"), \
             patch("core.analyst._gemini_keys", ["dummy"]), \
             patch("core.analyst._call_gemini") as mock_gemini, \
             patch("core.analyst._call_ollama") as mock_ollama:
            mock_gemini.return_value = None  # simulates malformed JSON / validation fail
            mock_ollama.return_value = json.loads(_valid_llm_response_text())
            signal = analyze_candidate(_make_screener_signal(ticker="AAPL", indicators={"RSI": 28}), df, ["news"])
        assert signal is not None
        assert mock_gemini.call_count == 3
        assert mock_ollama.call_count == 1

    def test_mixed_calls_accumulate_token_counters_per_provider(self, reset_analyst_state):
        """Gemini success + Gemini-503-then-Ollama + Gemini success → per-provider totals correct."""
        from core.analyst import analyze_candidate, get_daily_token_usage
        df = _make_df()
        import io
        import urllib.error
        http503 = urllib.error.HTTPError(
            url="https://generativelanguage.googleapis.com/x",
            code=503, msg="high demand", hdrs=None,
            fp=io.BytesIO(b"busy"),
        )
        with patch("core.analyst.AI_PROVIDER", "gemini"), \
             patch("core.analyst._gemini_keys", ["dummy"]), \
             patch("core.analyst.urllib.request.urlopen") as mu:
            mu.side_effect = [
                _mock_http_response(_gemini_success_body(_valid_llm_response_text(), 110, 70)),
                http503,
                _mock_http_response(_ollama_success_body(_valid_llm_response_text(), 200, 90)),
                _mock_http_response(_gemini_success_body(_valid_llm_response_text(), 130, 80)),
            ]
            analyze_candidate(_make_screener_signal(ticker="AAPL", indicators={"RSI": 28}), df, ["news"])
            analyze_candidate(_make_screener_signal(ticker="MSFT", indicators={"RSI": 28}), df, ["news"])
            analyze_candidate(_make_screener_signal(ticker="GOOG", indicators={"RSI": 28}), df, ["news"])
        usage = get_daily_token_usage()
        assert usage["gemini"] == {"input": 110 + 130, "output": 70 + 80}
        assert usage["ollama"] == {"input": 200, "output": 90}


class TestPluralKeyOnlyRouting:
    """Regression for the 2026-04-28 production bug.

    Symptom: an entire 14-hour run never made a single Gemini call —
    every analyst response was Ollama, every sector lookup logged
    "0 via Gemini, 13 via Ollama". The user's `.env` only set the
    plural rotation list (`GEMINI_API_KEYS=k1,k2,k3`), not the legacy
    singular `GEMINI_API_KEY`.

    Root cause: `_call_llm` gated the Gemini path on
    `bool(GEMINI_API_KEY)` (singular), which was empty, so the router
    skipped Gemini regardless of the plural list. The multi-key
    rotation that `_call_gemini` implemented was unreachable code.

    Contract after the fix: the router gate must read the same
    source of truth that the rotation does — the configured key
    list, exposed via `_active_gemini_keys()`. With the plural list
    populated and the singular empty, Gemini must be tried first.
    """

    def test_router_takes_gemini_path_when_only_plural_list_is_set(
        self, reset_analyst_state,
    ):
        from core.analyst import analyze_candidate
        df = _make_df()
        # `_gemini_keys` is the live rotation list (populated from
        # GEMINI_API_KEYS at module load); patching it directly expresses
        # "user has configured keys via GEMINI_API_KEYS".
        with patch("core.analyst.AI_PROVIDER", "gemini"), \
             patch("core.analyst._gemini_keys", ["KEY_FROM_PLURAL_LIST"]), \
             patch("core.analyst._call_gemini") as mock_gemini, \
             patch("core.analyst._call_ollama") as mock_ollama:
            mock_gemini.return_value = json.loads(_valid_llm_response_text())
            signal = analyze_candidate(_make_screener_signal(ticker="AAPL", indicators={"RSI": 28}), df, ["news"])
        assert signal is not None, "Router must produce a signal"
        assert mock_gemini.call_count >= 1, (
            "Router must enter the Gemini path when GEMINI_API_KEYS has keys"
        )
        assert mock_ollama.call_count == 0, (
            "Ollama must NOT be called when Gemini succeeds"
        )


class TestLLMTrafficLog:
    """Diagnostic JSONL traffic log: every Gemini and Ollama round-trip
    appends one record to logs/llm_traffic_YYYY-MM-DD.jsonl.

    Lets us correlate the confidence-distribution analysis with the actual
    prompts and responses — including failures (auth, RPM, malformed
    envelope) so we can see exactly what each provider returned and why.
    """

    def _read_jsonl(self, path):
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_gemini_success_writes_record_with_prompt_response_tokens(
        self, reset_analyst_state, tmp_path,
    ):
        from core.analyst import _call_gemini
        with patch("core.analyst.LOG_DIR", tmp_path), \
             patch("core.analyst.LLM_TRAFFIC_LOG_ENABLED", True), \
             patch("core.analyst._gemini_keys", ["KEY_FROM_TEST"]), \
             patch("core.analyst.urllib.request.urlopen") as mu:
            mu.return_value = _mock_http_response(_gemini_success_body(
                _valid_llm_response_text(), prompt_tokens=120, output_tokens=80,
            ))
            _call_gemini("the actual prompt body sent to gemini")

        files = list(tmp_path.glob("llm_traffic_*.jsonl"))
        assert len(files) == 1, f"expected 1 traffic log file, got {files}"
        records = self._read_jsonl(files[0])
        assert len(records) == 1, "exactly one record per call"
        rec = records[0]
        assert rec["provider"] == "gemini"
        assert rec["error"] is None
        assert rec["prompt"] == "the actual prompt body sent to gemini"
        assert rec["response"]["action"] == "buy"
        assert rec["response"]["confidence"] == 85
        assert rec["tokens"] == {"input": 120, "output": 80}
        assert isinstance(rec["elapsed_ms"], (int, float))
        assert "ts" in rec

    def test_gemini_429_rpm_writes_error_record_with_body(
        self, reset_analyst_state, tmp_path,
    ):
        """RPM 429 must produce a record so we can see exactly what Gemini said."""
        import io
        import urllib.error
        from core.analyst import _call_gemini, _GeminiTransportError
        rpm_body = b'{"error":{"message":"requests per minute"}}'
        with patch("core.analyst.LOG_DIR", tmp_path), \
             patch("core.analyst.LLM_TRAFFIC_LOG_ENABLED", True), \
             patch("core.analyst._gemini_keys", ["ONLY_KEY"]), \
             patch("core.analyst.urllib.request.urlopen") as mu:
            mu.side_effect = urllib.error.HTTPError(
                url="x", code=429, msg="rate", hdrs=None, fp=io.BytesIO(rpm_body),
            )
            with pytest.raises(_GeminiTransportError):
                _call_gemini("prompt")

        files = list(tmp_path.glob("llm_traffic_*.jsonl"))
        records = self._read_jsonl(files[0])
        assert len(records) == 1
        rec = records[0]
        assert rec["provider"] == "gemini"
        assert rec["error"] == "rpm_429"
        assert "requests per minute" in rec["response_raw"]
        assert rec["response"] is None

    def test_gemini_401_writes_error_record(self, reset_analyst_state, tmp_path):
        import io
        import urllib.error
        from core.analyst import _call_gemini, _GeminiTransportError
        with patch("core.analyst.LOG_DIR", tmp_path), \
             patch("core.analyst.LLM_TRAFFIC_LOG_ENABLED", True), \
             patch("core.analyst._gemini_keys", ["BAD_KEY"]), \
             patch("core.analyst.urllib.request.urlopen") as mu:
            mu.side_effect = urllib.error.HTTPError(
                url="x", code=401, msg="unauth",
                hdrs=None, fp=io.BytesIO(b"API key not valid"),
            )
            with pytest.raises(_GeminiTransportError):
                _call_gemini("prompt")

        files = list(tmp_path.glob("llm_traffic_*.jsonl"))
        records = self._read_jsonl(files[0])
        assert len(records) == 1
        assert records[0]["error"] == "auth_401"

    def test_gemini_malformed_envelope_writes_record_with_raw_body(
        self, reset_analyst_state, tmp_path,
    ):
        """Bad envelope (HTML error page, etc.) must capture the raw body
        so we can see what came back instead of the JSON we expected."""
        from core.analyst import _call_gemini
        with patch("core.analyst.LOG_DIR", tmp_path), \
             patch("core.analyst.LLM_TRAFFIC_LOG_ENABLED", True), \
             patch("core.analyst._gemini_keys", ["KEY"]), \
             patch("core.analyst.urllib.request.urlopen") as mu:
            mu.return_value = _mock_http_response(b"<html>504 Gateway Timeout</html>")
            _call_gemini("prompt")

        files = list(tmp_path.glob("llm_traffic_*.jsonl"))
        records = self._read_jsonl(files[0])
        assert len(records) == 1
        rec = records[0]
        assert rec["error"] == "malformed_envelope"
        assert "504 Gateway Timeout" in rec["response_raw"]

    def test_ollama_success_writes_record(self, reset_analyst_state, tmp_path):
        from core.analyst import _call_ollama
        with patch("core.analyst.LOG_DIR", tmp_path), \
             patch("core.analyst.LLM_TRAFFIC_LOG_ENABLED", True), \
             patch("core.analyst.urllib.request.urlopen") as mu:
            mu.return_value = _mock_http_response(_ollama_success_body(
                _valid_llm_response_text(), prompt_tokens=200, output_tokens=90,
            ))
            _call_ollama("the ollama prompt")

        files = list(tmp_path.glob("llm_traffic_*.jsonl"))
        records = self._read_jsonl(files[0])
        assert len(records) == 1
        rec = records[0]
        assert rec["provider"] == "ollama"
        assert rec["error"] is None
        assert rec["prompt"] == "the ollama prompt"
        assert rec["response"]["action"] == "buy"
        assert rec["tokens"] == {"input": 200, "output": 90}

    def test_disabled_via_env_writes_no_file(self, reset_analyst_state, tmp_path):
        """LLM_TRAFFIC_LOG_ENABLED=False must produce no JSONL output at all."""
        from core.analyst import _call_gemini
        with patch("core.analyst.LOG_DIR", tmp_path), \
             patch("core.analyst.LLM_TRAFFIC_LOG_ENABLED", False), \
             patch("core.analyst._gemini_keys", ["KEY"]), \
             patch("core.analyst.urllib.request.urlopen") as mu:
            mu.return_value = _mock_http_response(_gemini_success_body(
                _valid_llm_response_text(),
            ))
            _call_gemini("prompt")
        files = list(tmp_path.glob("llm_traffic_*.jsonl"))
        assert files == [], "no file should be created when logging disabled"

    def test_disk_error_does_not_break_analyst(self, reset_analyst_state, tmp_path):
        """If the JSONL write fails (disk full, perms), the analyst call MUST
        still succeed — diagnostic logging is best-effort, never load-bearing."""
        from core.analyst import _call_gemini
        # Point LOG_DIR at a path that cannot be written to (a regular file
        # masquerading as a directory). Any open() / mkdir() will raise.
        bad_path = tmp_path / "not_a_directory"
        bad_path.write_text("blocking file")
        with patch("core.analyst.LOG_DIR", bad_path), \
             patch("core.analyst.LLM_TRAFFIC_LOG_ENABLED", True), \
             patch("core.analyst._gemini_keys", ["KEY"]), \
             patch("core.analyst.urllib.request.urlopen") as mu:
            mu.return_value = _mock_http_response(_gemini_success_body(
                _valid_llm_response_text(),
            ))
            result = _call_gemini("prompt")
        assert result is not None, "Analyst must still return its result"

    def test_caller_context_ticker_and_kind_recorded(
        self, reset_analyst_state, tmp_path,
    ):
        """The trading-prompt caller (analyze_candidate) must annotate the
        record with the ticker and kind='trading' so we can correlate per-symbol."""
        from core.analyst import _call_gemini_payload
        payload = {
            "contents": [{"parts": [{"text": "hi"}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        }
        with patch("core.analyst.LOG_DIR", tmp_path), \
             patch("core.analyst.LLM_TRAFFIC_LOG_ENABLED", True), \
             patch("core.analyst._gemini_keys", ["KEY"]), \
             patch("core.analyst.urllib.request.urlopen") as mu:
            mu.return_value = _mock_http_response(_gemini_success_body(
                _valid_llm_response_text(),
            ))
            _call_gemini_payload(payload, ctx={"ticker": "AAPL", "kind": "trading"})

        files = list(tmp_path.glob("llm_traffic_*.jsonl"))
        records = self._read_jsonl(files[0])
        assert records[0]["ticker"] == "AAPL"
        assert records[0]["kind"] == "trading"

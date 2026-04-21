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
        assert "5 of 7" in prompt


class TestValidation:
    def test_valid_response(self):
        data = {
            "action": "buy",
            "confidence": 80,
            "entry_price": 150.0,
            "stop_loss": 145.0,
            "take_profit": 160.0,
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
            "entry_price": 150.0,
            "stop_loss": 145.0,
            "take_profit": 160.0,
            "trade_type": "day",
            "reasoning": "test",
        }
        assert _validate_response(data) is False

    def test_invalid_trade_type(self):
        data = {
            "action": "buy",
            "confidence": 80,
            "entry_price": 150.0,
            "stop_loss": 145.0,
            "take_profit": 160.0,
            "trade_type": "overnight",
            "reasoning": "test",
        }
        assert _validate_response(data) is False

    def test_confidence_out_of_range(self):
        data = {
            "action": "buy",
            "confidence": 150,
            "entry_price": 150.0,
            "stop_loss": 145.0,
            "take_profit": 160.0,
            "trade_type": "day",
            "reasoning": "test",
        }
        assert _validate_response(data) is False

    def test_negative_price(self):
        data = {
            "action": "buy",
            "confidence": 80,
            "entry_price": -10.0,
            "stop_loss": 145.0,
            "take_profit": 160.0,
            "trade_type": "day",
            "reasoning": "test",
        }
        assert _validate_response(data) is False


class TestShortSellingGate:
    """The ANALYSIS_PROMPT and validator must gate on ALLOW_SHORT_SELLING.

    When shorts are disabled, the LLM should only be offered 'buy' or 'hold',
    and any SELL that slips through is rejected without a retry.
    """

    def _valid_sell(self):
        """A well-formed sell response (direction-correct for a short)."""
        return {
            "action": "sell",
            "confidence": 80,
            "entry_price": 100.0,
            "stop_loss": 105.0,   # above entry for short
            "take_profit": 90.0,  # below entry for short
            "trade_type": "day",
            "reasoning": "Bearish setup",
        }

    def _valid_buy(self):
        return {
            "action": "buy",
            "confidence": 80,
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "take_profit": 110.0,
            "trade_type": "day",
            "reasoning": "Bullish setup",
        }

    def _valid_hold(self):
        return {
            "action": "hold",
            "confidence": 70,
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "take_profit": 110.0,
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
            "entry_price": 150.0,
            "stop_loss": 145.0,
            "take_profit": 160.0,
            "trade_type": "day",
            "reasoning": "Strong bullish divergence",
        }
        df = _make_df()
        signal = analyze_candidate("AAPL", "SMART", df, {"RSI": 28}, ["Good earnings"])
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
            "entry_price": 150.0,
            "stop_loss": 145.0,
            "take_profit": 160.0,
            "trade_type": "day",
            "reasoning": "Weak signal",
        }
        df = _make_df()
        signal = analyze_candidate("AAPL", "SMART", df, {}, [])
        assert signal is None

    @patch("core.analyst._call_llm")
    def test_filters_hold(self, mock_llm):
        mock_llm.return_value = {
            "action": "hold",
            "confidence": 90,
            "entry_price": 150.0,
            "stop_loss": 145.0,
            "take_profit": 160.0,
            "trade_type": "day",
            "reasoning": "Unclear direction",
        }
        df = _make_df()
        signal = analyze_candidate("AAPL", "SMART", df, {}, [])
        assert signal is None

    @patch("core.analyst._call_llm")
    def test_handles_llm_failure(self, mock_llm):
        mock_llm.return_value = None
        df = _make_df()
        signal = analyze_candidate("AAPL", "SMART", df, {}, [])
        assert signal is None

    @patch("core.analyst._call_llm")
    def test_macro_news_passed_to_prompt(self, mock_llm):
        mock_llm.return_value = {
            "action": "buy",
            "confidence": 85,
            "entry_price": 150.0,
            "stop_loss": 145.0,
            "take_profit": 160.0,
            "trade_type": "day",
            "reasoning": "Strong with favorable macro",
        }
        df = _make_df()
        signal = analyze_candidate(
            "AAPL", "SMART", df, {"RSI": 28}, ["Good earnings"],
            macro_news=["Fed cuts rates"],
        )
        assert signal is not None
        prompt_arg = mock_llm.call_args[0][0]
        assert "Fed cuts rates" in prompt_arg


class TestPriceRelationshipValidation:
    """Verify LLM response validation catches invalid price relationships."""

    def test_buy_stop_above_entry_rejected(self):
        data = {
            "action": "buy", "confidence": 80,
            "entry_price": 150.0, "stop_loss": 160.0, "take_profit": 170.0,
            "trade_type": "day", "reasoning": "test",
        }
        assert _validate_response(data) is False

    def test_buy_tp_below_entry_rejected(self):
        data = {
            "action": "buy", "confidence": 80,
            "entry_price": 150.0, "stop_loss": 145.0, "take_profit": 140.0,
            "trade_type": "day", "reasoning": "test",
        }
        assert _validate_response(data) is False

    def test_sell_stop_below_entry_rejected(self):
        data = {
            "action": "sell", "confidence": 80,
            "entry_price": 150.0, "stop_loss": 140.0, "take_profit": 130.0,
            "trade_type": "day", "reasoning": "test",
        }
        assert _validate_response(data) is False

    def test_sell_tp_above_entry_rejected(self):
        data = {
            "action": "sell", "confidence": 80,
            "entry_price": 150.0, "stop_loss": 160.0, "take_profit": 155.0,
            "trade_type": "day", "reasoning": "test",
        }
        assert _validate_response(data) is False

    def test_valid_buy_passes(self):
        data = {
            "action": "buy", "confidence": 80,
            "entry_price": 150.0, "stop_loss": 145.0, "take_profit": 160.0,
            "trade_type": "day", "reasoning": "test",
        }
        assert _validate_response(data) is True

    @patch("core.analyst.ALLOW_SHORT_SELLING", True)
    def test_valid_sell_passes(self):
        data = {
            "action": "sell", "confidence": 80,
            "entry_price": 150.0, "stop_loss": 155.0, "take_profit": 140.0,
            "trade_type": "day", "reasoning": "test",
        }
        assert _validate_response(data) is True


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

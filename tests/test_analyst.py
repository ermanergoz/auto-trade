"""Tests for core/analyst.py."""

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


class TestValidation:
    def test_valid_response(self):
        data = {
            "action": "buy",
            "confidence": 80,
            "entry_price": 150.0,
            "stop_loss": 145.0,
            "take_profit": 160.0,
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
            "reasoning": "test",
        }
        assert _validate_response(data) is False


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


class TestPriceRelationshipValidation:
    """Verify LLM response validation catches invalid price relationships."""

    def test_buy_stop_above_entry_rejected(self):
        data = {
            "action": "buy", "confidence": 80,
            "entry_price": 150.0, "stop_loss": 160.0, "take_profit": 170.0,
            "reasoning": "test",
        }
        assert _validate_response(data) is False

    def test_buy_tp_below_entry_rejected(self):
        data = {
            "action": "buy", "confidence": 80,
            "entry_price": 150.0, "stop_loss": 145.0, "take_profit": 140.0,
            "reasoning": "test",
        }
        assert _validate_response(data) is False

    def test_sell_stop_below_entry_rejected(self):
        data = {
            "action": "sell", "confidence": 80,
            "entry_price": 150.0, "stop_loss": 140.0, "take_profit": 130.0,
            "reasoning": "test",
        }
        assert _validate_response(data) is False

    def test_sell_tp_above_entry_rejected(self):
        data = {
            "action": "sell", "confidence": 80,
            "entry_price": 150.0, "stop_loss": 160.0, "take_profit": 155.0,
            "reasoning": "test",
        }
        assert _validate_response(data) is False

    def test_valid_buy_passes(self):
        data = {
            "action": "buy", "confidence": 80,
            "entry_price": 150.0, "stop_loss": 145.0, "take_profit": 160.0,
            "reasoning": "test",
        }
        assert _validate_response(data) is True

    def test_valid_sell_passes(self):
        data = {
            "action": "sell", "confidence": 80,
            "entry_price": 150.0, "stop_loss": 155.0, "take_profit": 140.0,
            "reasoning": "test",
        }
        assert _validate_response(data) is True


class TestOllamaTimeout:
    """Verify Ollama timeout is reasonable (not 600s)."""

    def test_timeout_is_reasonable(self):
        """Ollama timeout must be <= 120s to avoid blocking scan cycles."""
        import core.analyst as analyst_module
        import inspect
        source = inspect.getsource(analyst_module._call_ollama)
        # Check that timeout is not 600
        assert "timeout=600" not in source, "Ollama timeout must not be 600s"
        assert "timeout=60" in source, "Ollama timeout should be 60s"

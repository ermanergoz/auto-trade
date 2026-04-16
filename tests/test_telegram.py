"""Tests for notifications/telegram.py — TDD: written before implementation."""

from unittest.mock import patch, MagicMock

import pytest

from core.models import Action, TradeType
from tests.conftest import make_signal as _make_signal, make_position as _make_position


# ---------------------------------------------------------------------------
# TestIsStatusCommand
# ---------------------------------------------------------------------------

class TestIsStatusCommand:
    """Test that only 'status' and '/status' are valid triggers."""

    def test_status_recognized(self):
        from notifications.telegram import _is_status_command
        assert _is_status_command("status") is True

    def test_slash_status_recognized(self):
        from notifications.telegram import _is_status_command
        assert _is_status_command("/status") is True

    def test_status_case_insensitive(self):
        from notifications.telegram import _is_status_command
        assert _is_status_command("Status") is True
        assert _is_status_command("STATUS") is True

    def test_whatsup_not_recognized(self):
        from notifications.telegram import _is_status_command
        assert _is_status_command("whatsup") is False

    def test_whats_up_not_recognized(self):
        from notifications.telegram import _is_status_command
        assert _is_status_command("whats up") is False

    def test_whats_up_apostrophe_not_recognized(self):
        from notifications.telegram import _is_status_command
        assert _is_status_command("what's up") is False

    def test_random_text_not_recognized(self):
        from notifications.telegram import _is_status_command
        assert _is_status_command("hello") is False
        assert _is_status_command("buy AAPL") is False


# ---------------------------------------------------------------------------
# TestBuildStatusResponse
# ---------------------------------------------------------------------------

class TestBuildStatusResponse:
    """Test enhanced status response with portfolio data."""

    def test_basic_status_without_portfolio_data(self):
        from notifications.telegram import _build_status_response, _system_status

        # Ensure no portfolio data is cached
        _system_status["account"] = None
        _system_status["positions"] = None
        _system_status["daily_pnl"] = None
        _system_status["phase"] = "waiting"
        _system_status["mode"] = "paper"

        response = _build_status_response()
        assert "Status" in response
        assert "paper" in response
        assert "Waiting for next scan cycle" in response

    def test_status_includes_account_data(self):
        from notifications.telegram import _build_status_response, _system_status

        _system_status["phase"] = "scan_complete"
        _system_status["mode"] = "paper"
        _system_status["account"] = {
            "NetLiquidation": 100000.0,
            "TotalCashValue": 60000.0,
            "GrossPositionValue": 40000.0,
            "UnrealizedPnL": 1500.0,
        }
        _system_status["daily_pnl"] = 250.0
        _system_status["positions"] = None

        response = _build_status_response()
        assert "$100,000.00" in response
        assert "$60,000.00" in response
        assert "$40,000.00" in response
        assert "+1,500.00" in response  # unrealized
        assert "+250.00" in response  # daily pnl

    def test_status_includes_positions(self):
        from notifications.telegram import _build_status_response, _system_status

        positions = [
            _make_position(ticker="AAPL", quantity=50, entry_price=150.0),
            _make_position(ticker="GOOGL", quantity=20, entry_price=140.0),
        ]
        _system_status["phase"] = "scan_complete"
        _system_status["mode"] = "paper"
        _system_status["account"] = None
        _system_status["daily_pnl"] = None
        _system_status["positions"] = positions

        response = _build_status_response()
        assert "AAPL" in response
        assert "50" in response
        assert "150.00" in response
        assert "GOOGL" in response
        assert "20" in response
        assert "Open Positions (2)" in response

    def test_status_shows_ai_progress(self):
        from notifications.telegram import _build_status_response, _system_status

        _system_status["phase"] = "ai_analysis"
        _system_status["mode"] = "paper"
        _system_status["detail"] = "3/90 candidates for US"
        _system_status["account"] = None
        _system_status["positions"] = None
        _system_status["daily_pnl"] = None

        response = _build_status_response()
        assert "AI analyzing candidates" in response
        assert "3/90 candidates for US" in response

    def test_status_no_positions_section_when_empty(self):
        from notifications.telegram import _build_status_response, _system_status

        _system_status["phase"] = "waiting"
        _system_status["mode"] = "paper"
        _system_status["account"] = None
        _system_status["daily_pnl"] = None
        _system_status["positions"] = []

        response = _build_status_response()
        assert "Open Positions" not in response


# ---------------------------------------------------------------------------
# TestNotifyRiskResults
# ---------------------------------------------------------------------------

class TestNotifyRiskResults:
    """Test consolidated risk-approved signal notification."""

    @patch("notifications.telegram._send_sync")
    def test_sends_consolidated_message(self, mock_send):
        from notifications.telegram import notify_risk_results

        mock_send.return_value = True
        signals = [
            _make_signal(ticker="AAPL", action=Action.BUY, confidence=80, entry_price=150.0),
            _make_signal(ticker="TSLA", action=Action.SELL, confidence=72, entry_price=250.0),
        ]

        result = notify_risk_results(signals)

        assert result is True
        mock_send.assert_called_once()
        text = mock_send.call_args[0][0]
        assert "AAPL" in text
        assert "BUY" in text
        assert "TSLA" in text
        assert "SELL" in text
        assert "80%" in text
        assert "72%" in text
        assert "$150.00" in text
        assert "$250.00" in text

    @patch("notifications.telegram._send_sync")
    def test_empty_list_returns_false(self, mock_send):
        from notifications.telegram import notify_risk_results

        result = notify_risk_results([])

        assert result is False
        mock_send.assert_not_called()

    @patch("notifications.telegram._send_sync")
    def test_single_signal(self, mock_send):
        from notifications.telegram import notify_risk_results

        mock_send.return_value = True
        signals = [_make_signal(ticker="NVDA", action=Action.BUY, confidence=90)]

        result = notify_risk_results(signals)

        assert result is True
        text = mock_send.call_args[0][0]
        assert "NVDA" in text
        assert "1 signal(s)" in text


# ---------------------------------------------------------------------------
# TestConfidenceThreshold
# ---------------------------------------------------------------------------

class TestStatusRefreshesBeforeResponse:
    """When user asks for /status, data must be refreshed from DB first."""

    @patch("notifications.telegram.refresh_positions_cache")
    @patch("notifications.telegram._send_sync")
    @patch("notifications.telegram._get_updates_sync")
    def test_status_command_triggers_refresh(self, mock_updates, mock_send, mock_refresh):
        """Receiving 'status' must call refresh_positions_cache before responding."""
        from notifications.telegram import _poll_loop, _stop_event, _system_status

        _system_status["phase"] = "waiting"
        _system_status["mode"] = "paper"

        # Simulate one update with 'status', then stop the loop
        mock_msg = MagicMock()
        mock_msg.text = "status"
        mock_msg.chat_id = "123"

        mock_update = MagicMock()
        mock_update.update_id = 1
        mock_update.message = mock_msg

        call_count = 0
        def side_effect(offset=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [mock_update]
            _stop_event.set()
            return []

        mock_updates.side_effect = side_effect
        mock_send.return_value = True

        with patch("notifications.telegram.TELEGRAM_CHAT_ID", "123"):
            _stop_event.clear()
            _poll_loop()

        # refresh_positions_cache must have been called before the response
        mock_refresh.assert_called_once()
        mock_send.assert_called_once()


class TestConfidenceThreshold:
    """Test that confidence threshold is 65."""

    def test_threshold_is_65(self):
        from config.settings import AI_CONFIDENCE_THRESHOLD
        assert AI_CONFIDENCE_THRESHOLD == 65

    def test_prompt_references_65_not_70(self):
        from core.analyst import ANALYSIS_PROMPT
        assert "above 65" in ANALYSIS_PROMPT
        assert "above 70" not in ANALYSIS_PROMPT


# ---------------------------------------------------------------------------
# TestUpdatePortfolioData
# ---------------------------------------------------------------------------

class TestUpdatePortfolioData:
    """Test the portfolio data caching function."""

    def test_caches_data_in_system_status(self):
        from notifications.telegram import update_portfolio_data, _system_status

        account = {"NetLiquidation": 50000.0, "TotalCashValue": 30000.0}
        positions = [_make_position()]
        daily_pnl = 123.45

        update_portfolio_data(account, positions, daily_pnl)

        assert _system_status["account"] == account
        assert _system_status["positions"] == positions
        assert _system_status["daily_pnl"] == 123.45


class TestRefreshPositionsCache:
    """Test that refresh_positions_cache reads DB and updates the cache."""

    @patch("core.portfolio.get_daily_pnl", return_value=42.0)
    @patch("core.portfolio.get_open_positions")
    def test_refreshes_from_db(self, mock_get_pos, mock_get_pnl):
        from notifications.telegram import refresh_positions_cache, _system_status

        fake_pos = [_make_position(ticker="INTC")]
        mock_get_pos.return_value = fake_pos

        # Pre-populate with stale data
        _system_status["positions"] = [_make_position(ticker="SYRE")]
        _system_status["account"] = {}

        refresh_positions_cache()

        assert _system_status["positions"] == fake_pos
        assert _system_status["positions"][0].ticker == "INTC"
        mock_get_pos.assert_called_once()
        mock_get_pnl.assert_called_once()


class TestThreadSafety:
    """Verify _system_status is protected by a lock for thread-safe access."""

    def test_status_lock_exists(self):
        """A threading.Lock must protect _system_status access."""
        import notifications.telegram as tg
        assert hasattr(tg, "_status_lock"), (
            "_system_status must be protected by a _status_lock"
        )

    def test_concurrent_update_and_read_no_torn_state(self):
        """Concurrent writes and reads must not produce inconsistent state.

        This verifies that update_portfolio_data and _build_status_response
        don't mix old and new values when called from different threads.
        """
        import threading
        from notifications.telegram import (
            update_portfolio_data, _build_status_response, _system_status,
        )

        _system_status["phase"] = "scan_complete"
        _system_status["mode"] = "paper"
        errors = []

        account_a = {"NetLiquidation": 100_000.0, "TotalCashValue": 60_000.0,
                      "GrossPositionValue": 40_000.0, "UnrealizedPnL": 0.0}
        account_b = {"NetLiquidation": 200_000.0, "TotalCashValue": 120_000.0,
                      "GrossPositionValue": 80_000.0, "UnrealizedPnL": 0.0}

        def writer():
            for _ in range(200):
                update_portfolio_data(account_a, [], 100.0)
                update_portfolio_data(account_b, [], 200.0)

        def reader():
            for _ in range(200):
                resp = _build_status_response()
                # If we see account_a's NLV, daily_pnl must be account_a's too
                if "$100,000.00" in resp and "+200.00" in resp:
                    errors.append("Torn read: account_a NLV with account_b pnl")
                if "$200,000.00" in resp and "+100.00" in resp:
                    errors.append("Torn read: account_b NLV with account_a pnl")

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Torn reads detected: {errors[:5]}"

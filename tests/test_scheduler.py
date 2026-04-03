"""Tests for core/scheduler.py — streaming signal pipeline.

Verifies that AI-approved signals are immediately sent to risk check
via the on_signal callback, rather than waiting for all AI analysis
to complete before starting risk checks.
"""

from unittest.mock import patch, MagicMock, call

import pytest

from core.risk import RiskResult
from tests.conftest import make_signal as _make_signal, make_position as _make_position


# Patches common to all pipeline tests.
_PATCHES = [
    "core.scheduler.notify_risk_results",
    "core.scheduler.setup_fill_handler",
    "core.scheduler.setup_exit_handler",
    "core.scheduler.notify_trade",
    "core.scheduler.place_order",
    "core.scheduler.record_signal",
    "core.scheduler.evaluate",
    "core.scheduler.get_open_positions",
    "core.scheduler.get_daily_pnl",
    "core.scheduler.update_portfolio_data",
    "core.scheduler.screen_stocks",
    "core.scheduler.analyze_batch",
    "core.scheduler.build_universe",
    "core.scheduler.get_tickers_for_market",
    "core.scheduler.update_status",
    "core.scheduler.ensure_connected",
    "core.scheduler.get_news",
    "core.scheduler.get_active_markets",
]


def _make_stock_data(tickers):
    """Build a minimal stock_data dict for _fetch_market_data mock."""
    import pandas as pd

    df = pd.DataFrame({"close": [100.0], "open": [99.0], "high": [101.0], "low": [98.0], "volume": [1000000]})
    return {t: ("SMART", df) for t in tickers}


def _setup_mocks(
    mocks,
    signals,
    risk_approved=None,
    positions_sequence=None,
):
    """Configure mocks for a standard pipeline run.

    The fake analyze_batch fires each signal via on_signal callback,
    which is how the streaming pipeline processes them. The return
    value is an empty list — risk checks must happen in the callback,
    not a post-batch loop.
    """
    if risk_approved is None:
        risk_approved = [True] * len(signals)
    if positions_sequence is None:
        positions_sequence = [[]] * (len(signals) + 1)

    mocks["get_active_markets"].return_value = ["US"]
    mocks["get_daily_pnl"].return_value = 0.0
    mocks["get_open_positions"].side_effect = list(positions_sequence)

    from core.models import StockInfo

    stock_infos = [StockInfo(s.ticker, s.exchange, "", 0.0, 0.0) for s in signals]
    mocks["get_tickers_for_market"].return_value = stock_infos
    mocks["build_universe"].return_value = {}
    mocks["screen_stocks"].return_value = signals
    mocks["get_news"].return_value = []

    # analyze_batch fires signals via on_signal callback and returns
    # an empty list. The streaming pipeline must rely on the callback
    # for risk check + execution, not the return value.
    def fake_analyze_batch(ai_input, on_signal=None, on_progress=None):
        for sig in signals:
            if on_signal:
                on_signal(sig)
        return []

    mocks["analyze_batch"].side_effect = fake_analyze_batch

    risk_results = []
    for approved in risk_approved:
        if approved:
            risk_results.append(RiskResult(approved=True, reasons=[], position_size=10))
        else:
            risk_results.append(RiskResult(approved=False, reasons=["test rejection"]))
    mocks["evaluate"].side_effect = risk_results

    mocks["place_order"].return_value = [MagicMock()]


def _run_cycle(mocks):
    """Run a scan cycle with a fake IB connection."""
    from core.scheduler import run_scan_cycle

    ib = MagicMock()

    with patch("core.scheduler._fetch_market_data") as mock_fetch, \
         patch("core.connection.get_account_summary") as mock_account, \
         patch("core.scheduler.minutes_to_close", return_value=999):
        mock_account.return_value = {"NetLiquidation": 100_000}
        mock_fetch.return_value = _make_stock_data(
            [s.ticker for s in mocks["screen_stocks"].return_value],
        )
        return run_scan_cycle(ib, ["US"])


class TestStreamingPipeline:
    """Verify that AI-approved signals are streamed to risk check via callback."""

    @pytest.fixture(autouse=True)
    def _patch_all(self):
        patchers = {name.split(".")[-1]: patch(name) for name in _PATCHES}
        self.m = {}
        for key, p in patchers.items():
            self.m[key] = p.start()
        yield
        for p in patchers.values():
            p.stop()

    def test_analyze_batch_called_with_on_signal(self):
        """analyze_batch must receive on_signal as a callable."""
        sig = _make_signal(ticker="TSLA")
        _setup_mocks(self.m, [sig])

        _run_cycle(self.m)

        self.m["analyze_batch"].assert_called_once()
        kwargs = self.m["analyze_batch"].call_args[1]
        assert "on_signal" in kwargs
        assert callable(kwargs["on_signal"])

    def test_on_signal_triggers_risk_check(self):
        """evaluate() must be called from the on_signal callback."""
        sig = _make_signal(ticker="TSLA")
        _setup_mocks(self.m, [sig])

        _run_cycle(self.m)

        self.m["evaluate"].assert_called_once()
        assert self.m["evaluate"].call_args[0][0].ticker == "TSLA"

    def test_risk_approved_signal_gets_order(self):
        sig = _make_signal(ticker="GOOG")
        _setup_mocks(self.m, [sig], risk_approved=[True])

        _run_cycle(self.m)

        self.m["place_order"].assert_called_once()
        assert self.m["place_order"].call_args[0][1].ticker == "GOOG"

    def test_risk_rejected_signal_skips_execution(self):
        sig = _make_signal(ticker="BAD")
        _setup_mocks(self.m, [sig], risk_approved=[False])

        _run_cycle(self.m)

        self.m["place_order"].assert_not_called()

    def test_positions_refreshed_after_trade(self):
        sig1 = _make_signal(ticker="AAPL")
        sig2 = _make_signal(ticker="MSFT")
        pos_after_trade = [_make_position(ticker="AAPL")]
        # get_open_positions call sequence:
        #   0: initial fetch (line 134)
        #   1: refresh after sig1 trade -> returns pos_after_trade
        #   2: refresh after sig2 trade
        _setup_mocks(
            self.m,
            [sig1, sig2],
            risk_approved=[True, True],
            positions_sequence=[[], pos_after_trade, pos_after_trade],
        )

        _run_cycle(self.m)

        assert self.m["evaluate"].call_count == 2
        # Second evaluate call should receive positions refreshed after sig1 trade
        second_call_positions = self.m["evaluate"].call_args_list[1][0][1]
        assert len(second_call_positions) == 1
        assert second_call_positions[0].ticker == "AAPL"

    def test_summary_counts_correct(self):
        sig1 = _make_signal(ticker="A")
        sig2 = _make_signal(ticker="B")
        sig3 = _make_signal(ticker="C")
        _setup_mocks(
            self.m,
            [sig1, sig2, sig3],
            risk_approved=[True, False, True],
            positions_sequence=[[], [], [], []],
        )

        summary = _run_cycle(self.m)

        assert summary["ai_approved"] == 3
        assert summary["risk_approved"] == 2
        assert summary["orders_placed"] == 2

    def test_fill_handler_attached_before_order(self):
        """setup_fill_handler must be called BEFORE place_order to avoid race."""
        sig = _make_signal(ticker="RACE")
        _setup_mocks(self.m, [sig], risk_approved=[True])

        call_order = []
        self.m["setup_fill_handler"].side_effect = lambda *a, **kw: call_order.append("handler")
        self.m["place_order"].side_effect = lambda *a, **kw: (call_order.append("order"), [MagicMock()])[1]

        _run_cycle(self.m)

        assert call_order == ["handler", "order"], (
            f"Fill handler must be attached before order placement, got: {call_order}"
        )

    def test_exit_handler_attached_for_approved_signals(self):
        """setup_exit_handler must be called for risk-approved signals."""
        sig = _make_signal(ticker="EXIT")
        _setup_mocks(self.m, [sig], risk_approved=[True])

        _run_cycle(self.m)

        self.m["setup_exit_handler"].assert_called_once()

    def test_evaluate_receives_current_price(self):
        """evaluate() must receive current_price (not default 0) for anti-momentum."""
        sig = _make_signal(ticker="AMOM", entry_price=100.0)
        _setup_mocks(self.m, [sig], risk_approved=[True])

        _run_cycle(self.m)

        self.m["evaluate"].assert_called_once()
        kwargs = self.m["evaluate"].call_args[1]
        assert "current_price" in kwargs
        assert kwargs["current_price"] > 0, "current_price must not be 0"

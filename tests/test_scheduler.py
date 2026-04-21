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
    "core.scheduler.get_macro_news",
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
    sectors=None,
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
    if sectors is None:
        sectors = [""] * len(signals)

    mocks["get_active_markets"].return_value = ["US"]
    mocks["get_daily_pnl"].return_value = 0.0
    mocks["get_open_positions"].side_effect = list(positions_sequence)

    from core.models import StockInfo

    stock_infos = [
        StockInfo(s.ticker, s.exchange, sector, 0.0, 0.0)
        for s, sector in zip(signals, sectors)
    ]
    mocks["get_tickers_for_market"].return_value = stock_infos
    mocks["build_universe"].return_value = {}
    mocks["screen_stocks"].return_value = signals
    mocks["get_news"].return_value = []
    mocks["get_macro_news"].return_value = []

    # analyze_batch fires signals via on_signal callback and returns
    # an empty list. The streaming pipeline must rely on the callback
    # for risk check + execution, not the return value.
    def fake_analyze_batch(ai_input, on_signal=None, on_progress=None, macro_news=None):
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
         patch("core.connection.get_account_summary", return_value={"NetLiquidation": 100_000}), \
         patch("core.scheduler.minutes_to_close", return_value=999), \
         patch("core.scheduler.get_trades", return_value=[]):
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

    def test_skips_place_order_when_inside_close_window(self):
        """A scan that started before the close window must not place orders
        once wall-clock time has crossed into the close window — otherwise
        close_all_day_trades would immediately flatten the new position.
        """
        sig = _make_signal(ticker="LATE")
        _setup_mocks(self.m, [sig], risk_approved=[True])

        # When the scan-cycle starts, minutes_to_close = 20 (> CLOSE_MINUTES_BEFORE=15)
        # so the scan proceeds. When _on_signal runs for this signal, time has
        # advanced into the window (10 minutes to close). The callback must
        # see this and skip place_order.
        from core.scheduler import run_scan_cycle

        ib = MagicMock()
        from core.models import StockInfo

        # Build candidate list
        self.m["get_tickers_for_market"].return_value = [
            StockInfo("LATE", "SMART", "Technology", 0.0, 0.0),
        ]

        # Sequence: [scan_start=20, _on_signal=10, ...]
        close_sequence = iter([20, 10, 10, 10, 10, 10])
        with patch("core.scheduler._fetch_market_data") as mock_fetch, \
             patch("core.connection.get_account_summary", return_value={"NetLiquidation": 100_000}), \
             patch("core.scheduler.minutes_to_close", side_effect=lambda m: next(close_sequence)), \
             patch("core.scheduler.get_trades", return_value=[]):
            mock_fetch.return_value = _make_stock_data(["LATE"])
            run_scan_cycle(ib, ["US"])

        # place_order must not be called because the signal arrived inside the close window
        self.m["place_order"].assert_not_called()

    def test_pending_signal_included_in_next_risk_check(self):
        """A signal just approved + placed must appear in the next risk check's
        position list, even if the DB refresh (get_open_positions) hasn't seen
        the fill yet. Without this, two rapid-fire AI approvals can both pass
        the max-positions/sector-concentration gates against a stale view."""
        sig1 = _make_signal(ticker="AAPL")
        sig2 = _make_signal(ticker="MSFT")
        # Simulate slow fill: neither fill has been recorded in DB before the
        # second risk check runs. get_open_positions returns empty both times.
        _setup_mocks(
            self.m,
            [sig1, sig2],
            risk_approved=[True, True],
            positions_sequence=[[], [], []],
        )

        _run_cycle(self.m)

        assert self.m["evaluate"].call_count == 2
        # Second evaluate must see AAPL as a pending/virtual position,
        # even though get_open_positions returned empty.
        second_call_positions = self.m["evaluate"].call_args_list[1][0][1]
        tickers = {p.ticker for p in second_call_positions}
        assert "AAPL" in tickers, (
            "Second risk check must see the just-placed AAPL order as a "
            "pending position; DB refresh can lag the fill by seconds."
        )

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

    def test_fill_handler_attached_after_order_with_parent(self):
        """setup_fill_handler must be called AFTER place_order with parent_order.

        The bracket uses transmit=False until the last child order, so fills
        cannot arrive before place_order returns. Attaching handlers after
        allows passing the parent order for precise permId matching.
        """
        sig = _make_signal(ticker="RACE")
        _setup_mocks(self.m, [sig], risk_approved=[True])

        call_order = []
        self.m["setup_fill_handler"].side_effect = lambda *a, **kw: call_order.append("handler")
        self.m["place_order"].side_effect = lambda *a, **kw: (call_order.append("order"), [MagicMock()])[1]

        _run_cycle(self.m)

        assert call_order == ["order", "handler"], (
            f"Fill handler must be attached after order placement, got: {call_order}"
        )
        # Verify parent_order is passed for precise matching
        _, kwargs = self.m["setup_fill_handler"].call_args
        assert "parent_order" in kwargs and kwargs["parent_order"] is not None

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

    def test_sector_injected_into_candidates(self):
        """Sector from universe must be injected into screener candidates."""
        sig = _make_signal(ticker="TECH")
        _setup_mocks(self.m, [sig], risk_approved=[True], sectors=["Technology"])

        _run_cycle(self.m)

        # The signal passed to evaluate should have sector in indicator_values
        evaluated_signal = self.m["evaluate"].call_args[0][0]
        assert evaluated_signal.indicator_values.get("sector") == "Technology"


class TestNewsSkip:
    """Candidates with zero headlines should be dropped before the LLM call.

    Rationale: Yahoo/yfinance doesn't index news for the micro-caps the
    screener surfaces (SPAC units, tiny ETFs). When both Tavily and yfinance
    return empty, the LLM has no external signal — burning 200+s per call
    is wasted compute. Skip instead.
    """

    @pytest.fixture(autouse=True)
    def _patch_all(self):
        patchers = {name.split(".")[-1]: patch(name) for name in _PATCHES}
        self.m = {}
        for key, p in patchers.items():
            self.m[key] = p.start()
        yield
        for p in patchers.values():
            p.stop()

    def test_candidate_with_empty_news_is_dropped_from_ai_input(self):
        sig = _make_signal(ticker="NONEWS")
        _setup_mocks(self.m, [sig])
        self.m["get_news"].return_value = []

        _run_cycle(self.m)

        self.m["analyze_batch"].assert_called_once()
        ai_input = self.m["analyze_batch"].call_args[0][0]
        assert ai_input == [], (
            f"Expected empty ai_input when get_news returns [], got {ai_input}"
        )

    def test_candidate_with_headlines_is_kept_in_ai_input(self):
        sig = _make_signal(ticker="HASNEWS")
        _setup_mocks(self.m, [sig])
        self.m["get_news"].return_value = ["headline 1"]

        _run_cycle(self.m)

        self.m["analyze_batch"].assert_called_once()
        ai_input = self.m["analyze_batch"].call_args[0][0]
        assert len(ai_input) == 1
        assert ai_input[0]["ticker"] == "HASNEWS"
        assert ai_input[0]["news"] == ["headline 1"]

    def test_mixed_news_drops_only_empty_candidates(self):
        """With two candidates — one with news, one without — only the newsless is dropped."""
        sig_news = _make_signal(ticker="HAS")
        sig_nonews = _make_signal(ticker="NONE")
        _setup_mocks(self.m, [sig_news, sig_nonews])

        # Return news for HAS, empty for NONE
        def news_by_ticker(ticker, market=None):
            return ["good headline"] if ticker == "HAS" else []
        self.m["get_news"].side_effect = news_by_ticker

        _run_cycle(self.m)

        ai_input = self.m["analyze_batch"].call_args[0][0]
        tickers_in_input = [item["ticker"] for item in ai_input]
        assert tickers_in_input == ["HAS"]


# ---------------------------------------------------------------------------
# Feature 3: Nightly reconciliation
# ---------------------------------------------------------------------------

from datetime import datetime, date, timezone  # noqa: E402

from core.scheduler import run_nightly_reconciliation  # noqa: E402


class _FakeIBPos:
    def __init__(self, symbol, quantity):
        self.contract = type("C", (), {"symbol": symbol})()
        self.position = quantity


class TestNightlyReconciliation:
    """run_nightly_reconciliation(ib) fetches positions + fills, calls
    reconcile_positions (read-only), and sends Telegram alert on mismatch.
    """

    def test_in_sync_does_not_alert(self, tmp_path):
        """When DB and IBKR agree, no alert should be sent."""
        from core.portfolio import init_db, add_position
        from core.models import TradeType, Position

        db_path = tmp_path / "test.db"
        init_db(db_path)
        add_position(Position(
            ticker="AAPL", exchange="SMART", quantity=10,
            entry_price=150.0, entry_time=datetime(2024, 1, 1),
            stop_loss=145.0, take_profit=160.0, trade_type=TradeType.DAY,
        ), db_path)

        ib = MagicMock()
        ib.positions.return_value = [_FakeIBPos("AAPL", 10)]
        ib.fills.return_value = []

        with patch("core.scheduler.notify_reconciliation_mismatch") as mock_notify:
            report = run_nightly_reconciliation(ib, db_path=db_path)

        assert report["in_sync"] is True
        mock_notify.assert_not_called()

    def test_orphaned_db_position_sends_alert(self, tmp_path):
        """DB has AAPL but IBKR doesn't → alert."""
        from core.portfolio import init_db, add_position
        from core.models import TradeType, Position

        db_path = tmp_path / "test.db"
        init_db(db_path)
        add_position(Position(
            ticker="AAPL", exchange="SMART", quantity=10,
            entry_price=150.0, entry_time=datetime(2024, 1, 1),
            stop_loss=145.0, take_profit=160.0, trade_type=TradeType.DAY,
        ), db_path)

        ib = MagicMock()
        ib.positions.return_value = []
        ib.fills.return_value = []

        with patch("core.scheduler.notify_reconciliation_mismatch") as mock_notify:
            report = run_nightly_reconciliation(ib, db_path=db_path)

        assert report["in_sync"] is False
        assert "AAPL" in report["orphaned_db"]
        mock_notify.assert_called_once()
        args = mock_notify.call_args.args[0]
        assert args["orphaned_db"] == ["AAPL"]

    def test_orphaned_ibkr_position_sends_alert(self, tmp_path):
        """IBKR has MSFT but DB doesn't → alert."""
        from core.portfolio import init_db

        db_path = tmp_path / "test.db"
        init_db(db_path)

        ib = MagicMock()
        ib.positions.return_value = [_FakeIBPos("MSFT", 20)]
        ib.fills.return_value = []

        with patch("core.scheduler.notify_reconciliation_mismatch") as mock_notify:
            report = run_nightly_reconciliation(ib, db_path=db_path)

        assert "MSFT" in report["orphaned_ibkr"]
        mock_notify.assert_called_once()

    def test_quantity_mismatch_sends_alert(self, tmp_path):
        """DB says 10 shares, IBKR says 100 → alert."""
        from core.portfolio import init_db, add_position
        from core.models import TradeType, Position

        db_path = tmp_path / "test.db"
        init_db(db_path)
        add_position(Position(
            ticker="AAPL", exchange="SMART", quantity=10,
            entry_price=150.0, entry_time=datetime(2024, 1, 1),
            stop_loss=145.0, take_profit=160.0, trade_type=TradeType.DAY,
        ), db_path)

        ib = MagicMock()
        ib.positions.return_value = [_FakeIBPos("AAPL", 100)]
        ib.fills.return_value = []

        with patch("core.scheduler.notify_reconciliation_mismatch") as mock_notify:
            report = run_nightly_reconciliation(ib, db_path=db_path)

        assert "AAPL" in report["qty_mismatches"]
        mock_notify.assert_called_once()

    def test_does_not_auto_close_positions(self, tmp_path):
        """Nightly reconciliation is read-only — never closes positions.
        Auto-fix is reserved for reconnect-time reconciliation only.
        """
        from core.portfolio import init_db, add_position, get_open_positions
        from core.models import TradeType, Position

        db_path = tmp_path / "test.db"
        init_db(db_path)
        add_position(Position(
            ticker="AAPL", exchange="SMART", quantity=10,
            entry_price=150.0, entry_time=datetime(2024, 1, 1),
            stop_loss=145.0, take_profit=160.0, trade_type=TradeType.DAY,
        ), db_path)

        ib = MagicMock()
        ib.positions.return_value = []  # IBKR disagrees
        ib.fills.return_value = []

        with patch("core.scheduler.notify_reconciliation_mismatch"):
            run_nightly_reconciliation(ib, db_path=db_path)

        # Position must still be in DB (not auto-closed)
        assert len(get_open_positions(db_path)) == 1

    def test_returns_full_report(self, tmp_path):
        """Return value should be the reconcile_positions report dict."""
        from core.portfolio import init_db

        db_path = tmp_path / "test.db"
        init_db(db_path)

        ib = MagicMock()
        ib.positions.return_value = []
        ib.fills.return_value = []

        with patch("core.scheduler.notify_reconciliation_mismatch"):
            report = run_nightly_reconciliation(ib, db_path=db_path)

        # Report should include all standard reconcile_positions keys
        for key in ("db_count", "ibkr_count", "orphaned_db",
                    "orphaned_ibkr", "qty_mismatches", "in_sync"):
            assert key in report

    def test_handles_ibkr_positions_error(self, tmp_path):
        """If ib.positions() raises, fail gracefully and notify error."""
        from core.portfolio import init_db

        db_path = tmp_path / "test.db"
        init_db(db_path)

        ib = MagicMock()
        ib.positions.side_effect = ConnectionError("disconnected")
        ib.fills.return_value = []

        # Should not propagate the exception; should return error report
        with patch("core.scheduler.notify_error") as mock_err, \
             patch("core.scheduler.notify_reconciliation_mismatch"):
            report = run_nightly_reconciliation(ib, db_path=db_path)

        assert report.get("error") is not None
        mock_err.assert_called_once()

    def test_sign_mismatch_marks_critical(self, tmp_path):
        """Direction mismatch (DB long, IBKR short) is a critical alert."""
        from core.portfolio import init_db, add_position
        from core.models import TradeType, Position

        db_path = tmp_path / "test.db"
        init_db(db_path)
        add_position(Position(
            ticker="AAPL", exchange="SMART", quantity=10,
            entry_price=150.0, entry_time=datetime(2024, 1, 1),
            stop_loss=145.0, take_profit=160.0, trade_type=TradeType.DAY,
        ), db_path)

        ib = MagicMock()
        ib.positions.return_value = [_FakeIBPos("AAPL", -10)]  # short!
        ib.fills.return_value = []

        with patch("core.scheduler.notify_reconciliation_mismatch") as mock_notify:
            report = run_nightly_reconciliation(ib, db_path=db_path)

        assert report["qty_mismatches"]["AAPL"]["type"] == "sign_mismatch"
        mock_notify.assert_called_once()

    def test_multiple_mismatches_single_alert(self, tmp_path):
        """All mismatches reported in one alert, not one per ticker."""
        from core.portfolio import init_db, add_position
        from core.models import TradeType, Position

        db_path = tmp_path / "test.db"
        init_db(db_path)
        for t in ("AAPL", "MSFT", "NVDA"):
            add_position(Position(
                ticker=t, exchange="SMART", quantity=10,
                entry_price=150.0, entry_time=datetime(2024, 1, 1),
                stop_loss=145.0, take_profit=160.0, trade_type=TradeType.DAY,
            ), db_path)

        ib = MagicMock()
        ib.positions.return_value = []  # IBKR has none of them
        ib.fills.return_value = []

        with patch("core.scheduler.notify_reconciliation_mismatch") as mock_notify:
            run_nightly_reconciliation(ib, db_path=db_path)

        mock_notify.assert_called_once()
        args = mock_notify.call_args.args[0]
        assert sorted(args["orphaned_db"]) == ["AAPL", "MSFT", "NVDA"]

    def test_empty_both_sides_is_in_sync(self, tmp_path):
        """DB empty and IBKR empty = in sync, no alert."""
        from core.portfolio import init_db

        db_path = tmp_path / "test.db"
        init_db(db_path)

        ib = MagicMock()
        ib.positions.return_value = []
        ib.fills.return_value = []

        with patch("core.scheduler.notify_reconciliation_mismatch") as mock_notify:
            report = run_nightly_reconciliation(ib, db_path=db_path)

        assert report["in_sync"] is True
        mock_notify.assert_not_called()


class TestNotifyReconciliationMismatch:
    """notify_reconciliation_mismatch(report) formats and sends alert."""

    def test_formats_orphaned_db(self):
        from notifications.telegram import notify_reconciliation_mismatch

        report = {
            "db_count": 2, "ibkr_count": 1,
            "orphaned_db": ["AAPL", "MSFT"],
            "orphaned_ibkr": [],
            "qty_mismatches": {},
            "in_sync": False,
        }
        with patch("notifications.telegram._send_sync") as mock_send:
            mock_send.return_value = True
            notify_reconciliation_mismatch(report)

        mock_send.assert_called_once()
        text = mock_send.call_args.args[0]
        assert "AAPL" in text
        assert "MSFT" in text

    def test_formats_orphaned_ibkr(self):
        from notifications.telegram import notify_reconciliation_mismatch

        report = {
            "db_count": 0, "ibkr_count": 1,
            "orphaned_db": [],
            "orphaned_ibkr": ["GOOG"],
            "qty_mismatches": {},
            "in_sync": False,
        }
        with patch("notifications.telegram._send_sync") as mock_send:
            mock_send.return_value = True
            notify_reconciliation_mismatch(report)

        text = mock_send.call_args.args[0]
        assert "GOOG" in text

    def test_formats_qty_mismatch(self):
        from notifications.telegram import notify_reconciliation_mismatch

        report = {
            "db_count": 1, "ibkr_count": 1,
            "orphaned_db": [], "orphaned_ibkr": [],
            "qty_mismatches": {
                "AAPL": {"db": 10, "ibkr": 100, "type": "quantity_mismatch"},
            },
            "in_sync": False,
        }
        with patch("notifications.telegram._send_sync") as mock_send:
            mock_send.return_value = True
            notify_reconciliation_mismatch(report)

        text = mock_send.call_args.args[0]
        assert "AAPL" in text
        assert "10" in text and "100" in text

    def test_sign_mismatch_flagged_critical(self):
        from notifications.telegram import notify_reconciliation_mismatch

        report = {
            "db_count": 1, "ibkr_count": 1,
            "orphaned_db": [], "orphaned_ibkr": [],
            "qty_mismatches": {
                "AAPL": {"db": 10, "ibkr": -10, "type": "sign_mismatch"},
            },
            "in_sync": False,
        }
        with patch("notifications.telegram._send_sync") as mock_send:
            mock_send.return_value = True
            notify_reconciliation_mismatch(report)

        text = mock_send.call_args.args[0]
        # Message should loudly warn about direction mismatch
        assert "direction" in text.lower() or "critical" in text.lower() or "sign" in text.lower()


class TestMarketHoursDST:
    """Verify NYSE market hours follow US DST, not the local display timezone.

    NYSE uses America/New_York which observes DST (EDT summer = UTC-4,
    EST winter = UTC-5). Europe/Istanbul (TRT) is a fixed UTC+3 offset.
    The gap between NYSE open (09:30 ET) in Istanbul wall-clock time is:
      - 16:30 TRT in summer (EDT)
      - 17:30 TRT in winter (EST)

    If market hours are stored as Istanbul clock times, the bot opens and
    closes scanning an hour off during DST shifts. The authoritative hours
    must be in ET so the NYSE open/close tracks the actual market.
    """

    def _run_at(self, iso_utc: str):
        """Return is_market_open('US') as if now() were the given UTC timestamp."""
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from unittest.mock import patch
        import core.scheduler as sched

        fixed = datetime.fromisoformat(iso_utc).replace(tzinfo=ZoneInfo("UTC"))

        class FakeDT(datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed.astimezone(tz) if tz else fixed.replace(tzinfo=None)

        with patch.object(sched, "datetime", FakeDT):
            return sched.is_market_open("US")

    def test_nyse_open_summer_edt(self):
        """2026-07-15 is EDT. 09:30 ET = 13:30 UTC → market open."""
        assert self._run_at("2026-07-15T13:35:00") is True

    def test_nyse_closed_before_open_summer(self):
        """2026-07-15 09:00 ET = 13:00 UTC → market not yet open."""
        assert self._run_at("2026-07-15T13:00:00") is False

    def test_nyse_open_winter_est(self):
        """2026-01-15 is EST. 09:30 ET = 14:30 UTC → market open.

        This is the critical DST case: in winter, NYSE opens 1 hour later
        in UTC than in summer. If hours are hard-coded as Istanbul clock
        times the winter NYSE open is missed entirely for the first hour.
        """
        assert self._run_at("2026-01-15T14:35:00") is True

    def test_nyse_closed_before_open_winter(self):
        """2026-01-15 09:00 ET = 14:00 UTC → market not yet open.

        In summer this same UTC time would be market-open. The test ensures
        the bot does not scan before the ET open during EST.
        """
        assert self._run_at("2026-01-15T14:00:00") is False

    def test_nyse_closed_after_close_winter(self):
        """2026-01-15 16:05 ET = 21:05 UTC → past close."""
        assert self._run_at("2026-01-15T21:05:00") is False

    def test_nyse_open_just_before_close_winter(self):
        """2026-01-15 15:55 ET = 20:55 UTC → still open."""
        assert self._run_at("2026-01-15T20:55:00") is True

    def test_weekend_closed(self):
        """Saturday is closed regardless of time."""
        # 2026-01-17 is a Saturday
        assert self._run_at("2026-01-17T15:00:00") is False


class TestPdtTradeWindow:
    """The PDT rule counts day trades over a rolling 5-business-day window,
    which is ~7 calendar days. The scheduler must query the DB with a
    start_date 7 days back so `check_pdt_restriction` sees the full window.

    Previously the scheduler queried `start_date=today`, which truncated the
    trade history to today only. With that truncation the 7-day filter inside
    check_pdt_restriction is a no-op and the rolling count is effectively 0,
    which defeats the protection entirely and can trigger an IBKR 30-day
    account lockout on sub-$5k accounts.
    """

    @pytest.fixture(autouse=True)
    def _patch_all(self):
        patchers = {name.split(".")[-1]: patch(name) for name in _PATCHES}
        self.m = {}
        for key, p in patchers.items():
            self.m[key] = p.start()
        yield
        for p in patchers.values():
            p.stop()

    def test_get_trades_queries_seven_days_back_for_pdt_window(self):
        """scheduler._on_signal must query get_trades with start_date ~7 days ago.

        Using `datetime.now().date()` as the start date truncates the PDT
        trade history to today only, silently disabling PDT protection.
        """
        from datetime import date, timedelta, datetime as _dt, timezone as _tz

        sig = _make_signal(ticker="PDTX")
        _setup_mocks(self.m, [sig], risk_approved=[True])

        captured = {}

        def fake_get_trades(start_date=None, end_date=None, ticker=None, db_path=None):
            # record only the first call that passes start_date (the PDT lookup).
            if start_date is not None and "start_date" not in captured:
                captured["start_date"] = start_date
            return []

        from core.scheduler import run_scan_cycle
        ib = MagicMock()
        with patch("core.scheduler._fetch_market_data") as mock_fetch, \
             patch("core.connection.get_account_summary", return_value={"NetLiquidation": 100_000}), \
             patch("core.scheduler.minutes_to_close", return_value=999), \
             patch("core.scheduler.get_trades", side_effect=fake_get_trades):
            mock_fetch.return_value = _make_stock_data(["PDTX"])
            run_scan_cycle(ib, ["US"])

        assert "start_date" in captured, "scheduler did not call get_trades with start_date"
        today_utc = _dt.now(_tz.utc).date()
        delta = (today_utc - captured["start_date"]).days
        assert delta >= 5, (
            f"PDT trade window too narrow: scheduler looked back {delta} days, "
            f"need at least 5 business days (~7 calendar days) to honour IBKR's "
            f"5-day rolling PDT rule."
        )

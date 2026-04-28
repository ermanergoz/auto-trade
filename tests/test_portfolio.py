"""Tests for core/portfolio.py."""

import tempfile
from datetime import datetime, date
from pathlib import Path

import pytest

from core.models import (
    Position, Trade, Signal, DailySummary, Action, TradeType,
)
from core.portfolio import (
    init_db, add_position, close_position, get_open_positions,
    get_trades, get_daily_pnl, record_signal, record_daily_summary,
    get_daily_summary,
)


@pytest.fixture
def db_path(tmp_path):
    """Create a temporary database for testing."""
    path = tmp_path / "test_portfolio.db"
    init_db(path)
    return path


def _make_position(**kwargs) -> Position:
    defaults = dict(
        ticker="AAPL",
        exchange="SMART",
        quantity=10,
        entry_price=150.0,
        entry_time=datetime(2024, 1, 15, 10, 30),
        stop_loss=145.5,
        take_profit=159.0,
        trade_type=TradeType.DAY,
        sector="Technology",
    )
    defaults.update(kwargs)
    return Position(**defaults)


class TestPositions:
    def test_add_and_get(self, db_path):
        pos = _make_position()
        row_id = add_position(pos, db_path)
        assert row_id >= 1

        positions = get_open_positions(db_path)
        assert len(positions) == 1
        assert positions[0].ticker == "AAPL"
        assert positions[0].quantity == 10
        assert positions[0].entry_price == 150.0

    def test_multiple_positions(self, db_path):
        add_position(_make_position(ticker="AAPL"), db_path)
        add_position(_make_position(ticker="MSFT", entry_price=300.0), db_path)
        add_position(_make_position(ticker="NVDA", entry_price=800.0), db_path)

        positions = get_open_positions(db_path)
        assert len(positions) == 3
        tickers = {p.ticker for p in positions}
        assert tickers == {"AAPL", "MSFT", "NVDA"}

    def test_concurrent_close_position_inserts_one_trade(self, db_path):
        """Concurrent close_position calls for the same ticker must insert
        exactly one Trade row. The exit handler (ib_insync event thread) and
        close_all_day_trades (main thread) can both fire close_position for
        the same ticker during a day-trade close; without serialization, two
        trade records with identical data get written.
        """
        import threading

        add_position(_make_position(ticker="AAPL"), db_path)

        results = []
        barrier = threading.Barrier(2)

        def _close():
            barrier.wait()  # release both threads simultaneously
            t = close_position("AAPL", exit_price=160.0, db_path=db_path)
            results.append(t)

        t1 = threading.Thread(target=_close)
        t2 = threading.Thread(target=_close)
        t1.start(); t2.start()
        t1.join(); t2.join()

        trades = get_trades(db_path=db_path)
        assert len(trades) == 1, (
            f"Expected exactly 1 trade row after concurrent close, got {len(trades)}"
        )
        # Exactly one thread should have returned a Trade; the other gets None
        non_none = [r for r in results if r is not None]
        assert len(non_none) == 1, (
            f"Expected exactly 1 close_position() to succeed, got {len(non_none)}"
        )

    def test_close_position(self, db_path):
        add_position(_make_position(ticker="AAPL"), db_path)
        exit_dt = datetime(2024, 1, 15, 15, 0)
        trade = close_position(
            "AAPL", exit_price=160.0,
            exit_time=exit_dt,
            db_path=db_path,
        )

        assert trade is not None
        assert trade.ticker == "AAPL"
        assert trade.exit_price == 160.0
        assert trade.pnl == 100.0  # (160 - 150) * 10
        assert trade.exit_time == exit_dt

        # Position should be removed
        positions = get_open_positions(db_path)
        assert len(positions) == 0

        # Trade should be recorded and datetime round-trips correctly
        trades = get_trades(db_path=db_path)
        assert len(trades) == 1
        assert trades[0].pnl == 100.0
        assert isinstance(trades[0].exit_time, datetime)
        assert isinstance(trades[0].entry_time, datetime)
        # DB round-trip should preserve timezone (UTC)
        assert trades[0].exit_time.tzinfo is not None
        assert trades[0].entry_time.tzinfo is not None

    def test_close_nonexistent(self, db_path):
        result = close_position("NOPE", 100.0, db_path=db_path)
        assert result is None

    def test_concurrent_add_position_does_not_duplicate(self, db_path):
        """Two threads calling add_position for the same ticker must not
        both succeed in INSERTing a row.

        The fill handler runs on ib_insync's event thread while the main
        scheduler loop also calls add_position through reconciliation /
        handle_fill paths. Without a DB-level uniqueness guarantee, the
        SELECT-then-INSERT check inside add_position is a TOCTOU race: both
        threads' SELECTs can return empty before either INSERT commits, and
        both INSERTs succeed. The DB must enforce uniqueness so that at
        most one row per ticker exists no matter how calls interleave.
        """
        import threading

        pos = _make_position(ticker="AAPL")
        barrier = threading.Barrier(4)

        def _insert():
            barrier.wait()
            add_position(_make_position(ticker="AAPL"), db_path)

        threads = [threading.Thread(target=_insert) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        positions = get_open_positions(db_path)
        tickers = [p.ticker for p in positions]
        assert tickers.count("AAPL") == 1, (
            f"Expected exactly one AAPL row after 4 concurrent inserts; "
            f"got {tickers.count('AAPL')}: {positions}"
        )


class TestTrades:
    def test_get_trades_empty(self, db_path):
        trades = get_trades(db_path=db_path)
        assert trades == []

    def test_get_daily_pnl(self, db_path):
        add_position(
            _make_position(ticker="AAPL", entry_time=datetime(2024, 1, 15, 10, 0)),
            db_path,
        )
        close_position(
            "AAPL", exit_price=160.0,
            exit_time=datetime(2024, 1, 15, 15, 0),
            db_path=db_path,
        )

        pnl = get_daily_pnl(date(2024, 1, 15), db_path)
        assert pnl == 100.0  # (160 - 150) * 10

        # Different day should be 0
        pnl_other = get_daily_pnl(date(2024, 1, 16), db_path)
        assert pnl_other == 0.0


class TestSignals:
    def test_record_signal(self, db_path):
        sig = Signal(
            ticker="AAPL",
            action=Action.BUY,
            confidence=85.0,
            entry_price=150.0,
            stop_loss=145.5,
            take_profit=159.0,
            reasoning="Strong MACD crossover",
            source="ai",
            exchange="SMART",
            timestamp=datetime(2024, 1, 15, 10, 0),
        )
        row_id = record_signal(sig, db_path)
        assert row_id >= 1


class TestDailySummary:
    def test_record_and_get(self, db_path):
        summary = DailySummary(
            date=date(2024, 1, 15),
            portfolio_value=100_000.0,
            daily_pnl=500.0,
            daily_pnl_pct=0.5,
            num_trades=3,
            winning_trades=2,
            losing_trades=1,
        )
        record_daily_summary(summary, db_path)

        result = get_daily_summary(date(2024, 1, 15), db_path)
        assert result is not None
        assert result.portfolio_value == 100_000.0
        assert result.daily_pnl == 500.0
        assert result.num_trades == 3

    def test_get_nonexistent(self, db_path):
        result = get_daily_summary(date(2024, 6, 1), db_path)
        assert result is None


class TestReconcilePositions:
    """Verify position reconciliation between DB and IBKR."""

    def test_in_sync(self, db_path):
        from core.portfolio import reconcile_positions

        pos = _make_position(ticker="AAPL")
        add_position(pos, db_path)

        ibkr_positions = [{"ticker": "AAPL", "quantity": pos.quantity}]
        report = reconcile_positions(ibkr_positions, db_path=db_path)

        assert report["in_sync"] is True
        assert report["orphaned_db"] == []
        assert report["orphaned_ibkr"] == []
        assert report["qty_mismatches"] == {}

    def test_quantity_mismatch(self, db_path):
        from core.portfolio import reconcile_positions

        pos = _make_position(ticker="AAPL")  # quantity=10
        add_position(pos, db_path)

        ibkr_positions = [{"ticker": "AAPL", "quantity": 100}]
        report = reconcile_positions(ibkr_positions, db_path=db_path)

        assert report["in_sync"] is False
        assert "AAPL" in report["qty_mismatches"]
        assert report["qty_mismatches"]["AAPL"]["db"] == 10
        assert report["qty_mismatches"]["AAPL"]["ibkr"] == 100

    def test_orphaned_in_db(self, db_path):
        from core.portfolio import reconcile_positions

        pos = _make_position(ticker="AAPL")
        add_position(pos, db_path)

        ibkr_positions = []  # IBKR has nothing
        report = reconcile_positions(ibkr_positions, db_path=db_path)

        assert report["in_sync"] is False
        assert "AAPL" in report["orphaned_db"]

    def test_orphaned_in_ibkr(self, db_path):
        from core.portfolio import reconcile_positions

        ibkr_positions = [{"ticker": "MSFT", "quantity": 50}]
        report = reconcile_positions(ibkr_positions, db_path=db_path)

        assert report["in_sync"] is False
        assert "MSFT" in report["orphaned_ibkr"]

    def test_empty_both(self, db_path):
        from core.portfolio import reconcile_positions

        report = reconcile_positions([], db_path=db_path)
        assert report["in_sync"] is True

    def test_auto_fix_closes_orphaned_db_positions(self, db_path):
        from core.portfolio import reconcile_positions

        pos = _make_position(ticker="SYRE")
        add_position(pos, db_path)
        pos2 = _make_position(ticker="AAOI", entry_price=148.0)
        add_position(pos2, db_path)

        # IBKR has neither — they were stopped out while bot was offline
        report = reconcile_positions([], auto_fix=True, db_path=db_path)

        assert report["auto_closed"] == ["AAOI", "SYRE"]
        # DB should now be empty
        assert get_open_positions(db_path) == []
        # Trades should be recorded
        trades = get_trades(db_path=db_path)
        assert len(trades) == 2
        tickers = {t.ticker for t in trades}
        assert tickers == {"SYRE", "AAOI"}

    def test_auto_fix_preserves_matching_positions(self, db_path):
        from core.portfolio import reconcile_positions

        pos = _make_position(ticker="INTC")
        add_position(pos, db_path)
        pos2 = _make_position(ticker="SYRE")
        add_position(pos2, db_path)

        # IBKR still holds INTC but not SYRE
        ibkr = [{"ticker": "INTC", "quantity": pos.quantity}]
        report = reconcile_positions(ibkr, auto_fix=True, db_path=db_path)

        assert report["auto_closed"] == ["SYRE"]
        remaining = get_open_positions(db_path)
        assert len(remaining) == 1
        assert remaining[0].ticker == "INTC"

    def test_auto_fix_false_does_not_close(self, db_path):
        from core.portfolio import reconcile_positions

        pos = _make_position(ticker="AAPL")
        add_position(pos, db_path)

        report = reconcile_positions([], auto_fix=False, db_path=db_path)

        assert report["auto_closed"] == []
        assert len(get_open_positions(db_path)) == 1

    def test_auto_fix_uses_midpoint_exit_price_when_no_fills(self, db_path):
        """Auto-reconcile must use the unbiased midpoint of SL/TP, not stop_loss.

        Previous behavior used stop_loss unconditionally, which records a loss
        for every orphaned position regardless of whether it actually hit SL or
        TP. That systematically biases the trade journal and circuit-breaker
        history toward losses.

        Assuming equal prior probability that a bracketed position exited via
        SL or TP, the midpoint (SL + TP) / 2 is the unbiased expected-value
        estimator of the exit price. Individual trades carry an "estimated"
        flag in the reasoning so downstream consumers know it is not actual.
        """
        from core.portfolio import reconcile_positions

        pos = _make_position(
            ticker="SYRE", entry_price=150.0, stop_loss=145.5, take_profit=159.0,
        )
        add_position(pos, db_path)

        report = reconcile_positions([], auto_fix=True, db_path=db_path)

        assert report["auto_closed"] == ["SYRE"]
        trades = get_trades(db_path=db_path)
        assert len(trades) == 1
        # Midpoint of SL (145.5) and TP (159.0) = 152.25
        assert trades[0].exit_price == pytest.approx(152.25)
        # P&L: (152.25 - 150.0) * 10 = $22.50
        assert trades[0].pnl == pytest.approx(22.5)
        # Reasoning must flag the exit as an estimate so downstream logic
        # (circuit breaker, daily P&L) can recognize unreliable data.
        assert "estimate" in trades[0].reasoning.lower()

    def test_auto_fix_uses_entry_price_when_no_stop_loss_or_tp(self, db_path):
        """When both SL and TP are 0, fall back to entry_price (0 P&L)."""
        from core.portfolio import reconcile_positions

        pos = _make_position(
            ticker="NOSTOP", entry_price=150.0, stop_loss=0.0, take_profit=0.0,
        )
        add_position(pos, db_path)

        report = reconcile_positions([], auto_fix=True, db_path=db_path)
        trades = get_trades(db_path=db_path)
        assert len(trades) == 1
        # No SL/TP → falls back to entry_price (neutral, $0 P&L)
        assert trades[0].exit_price == 150.0

    def test_auto_fix_uses_stop_loss_when_no_tp(self, db_path):
        """When only stop_loss is set (no TP), use stop_loss as fallback."""
        from core.portfolio import reconcile_positions

        pos = _make_position(
            ticker="SLONLY", entry_price=150.0, stop_loss=145.5, take_profit=0.0,
        )
        add_position(pos, db_path)

        report = reconcile_positions([], auto_fix=True, db_path=db_path)
        trades = get_trades(db_path=db_path)
        assert len(trades) == 1
        assert trades[0].exit_price == 145.5

    def test_sign_mismatch_detected(self, db_path):
        """Direction mismatch (DB long but IBKR short) must be flagged."""
        from core.portfolio import reconcile_positions

        pos = _make_position(ticker="AAPL", quantity=10)  # long in DB
        add_position(pos, db_path)

        # IBKR says short!
        ibkr = [{"ticker": "AAPL", "quantity": -10}]
        report = reconcile_positions(ibkr, db_path=db_path)

        assert report["in_sync"] is False
        assert "AAPL" in report["qty_mismatches"]
        assert report["qty_mismatches"]["AAPL"]["type"] == "sign_mismatch"


class TestAutoFixWithFills:
    """Auto-reconcile should prefer actual IBKR fill prices over stop_loss
    estimates when fills are provided — so P&L reflects reality."""

    def test_naive_fill_time_does_not_crash(self, db_path):
        """ib_insync sometimes returns naive datetimes for fill.time. The
        reconcile path must not raise TypeError when comparing naive fill
        times against the aware entry_time. Either the fill must be
        normalized to UTC, or the comparison must tolerate naive times.
        """
        from core.portfolio import reconcile_positions
        from datetime import timezone

        pos = _make_position(
            ticker="NAIV", quantity=5, entry_price=50.0, stop_loss=45.0,
            entry_time=datetime(2026, 4, 16, 14, 0, tzinfo=timezone.utc),
        )
        add_position(pos, db_path)

        # Naive fill time (no tzinfo) — represents what ib_insync may return
        fills = [{
            "ticker": "NAIV", "side": "SLD", "shares": 5.0, "price": 46.0,
            "time": datetime(2026, 4, 17, 13, 30, 0),  # naive
        }]
        # Must not raise TypeError
        report = reconcile_positions(
            [], auto_fix=True, ibkr_fills=fills, db_path=db_path,
        )
        assert report["auto_closed"] == ["NAIV"]
        trades = get_trades(db_path=db_path)
        # Naive time was treated as UTC (positive); fill actually used
        assert trades[0].exit_price == pytest.approx(46.0)

    def test_fill_price_overrides_stop_loss(self, db_path):
        """When a matching IBKR fill exists, use its price, not stop_loss."""
        from core.portfolio import reconcile_positions
        from datetime import timezone

        pos = _make_position(
            ticker="IMOS", quantity=5, entry_price=48.40, stop_loss=46.00,
            entry_time=datetime(2026, 4, 16, 14, 0, tzinfo=timezone.utc),
        )
        add_position(pos, db_path)

        # IBKR reports the actual sell fill at $45.725 (below stop_loss)
        fills = [{
            "ticker": "IMOS", "side": "SLD", "shares": 5.0, "price": 45.725,
            "time": datetime(2026, 4, 17, 13, 42, 43, tzinfo=timezone.utc),
            "realized_pnl": -15.38,
        }]
        report = reconcile_positions(
            [], auto_fix=True, ibkr_fills=fills, db_path=db_path,
        )

        assert report["auto_closed"] == ["IMOS"]
        trades = get_trades(db_path=db_path)
        assert len(trades) == 1
        # Must use actual fill price, not stop_loss
        assert trades[0].exit_price == pytest.approx(45.725)
        # P&L based on actual fill: (45.725 - 48.40) * 5 = -13.375
        assert trades[0].pnl == pytest.approx(-13.375)

    def test_fill_price_overrides_when_take_profit_hit(self, db_path):
        """A take-profit fill above stop_loss must also use actual price.

        Without fill data, auto-reconcile would record this as a stop-out
        (loss), when in reality it was a take-profit (gain).
        """
        from core.portfolio import reconcile_positions
        from datetime import timezone

        pos = _make_position(
            ticker="INTC", quantity=3, entry_price=64.33, stop_loss=60.00,
            take_profit=69.00,
            entry_time=datetime(2026, 4, 16, 9, 0, tzinfo=timezone.utc),
        )
        add_position(pos, db_path)

        fills = [{
            "ticker": "INTC", "side": "SLD", "shares": 3.0, "price": 69.00,
            "time": datetime(2026, 4, 17, 13, 30, 3, tzinfo=timezone.utc),
            "realized_pnl": 13.00,
        }]
        report = reconcile_positions(
            [], auto_fix=True, ibkr_fills=fills, db_path=db_path,
        )

        trades = get_trades(db_path=db_path)
        assert trades[0].exit_price == pytest.approx(69.00)
        # Gain: (69.00 - 64.33) * 3 = 14.01 (gross, ignoring commission)
        assert trades[0].pnl == pytest.approx(14.01)

    def test_falls_back_to_midpoint_when_no_matching_fill(self, db_path):
        """When no fill is provided for a ticker, fall back to SL/TP midpoint.

        Using the midpoint (rather than SL alone) avoids systematically biasing
        the trade journal toward recorded losses when the actual outcome is
        unknown.
        """
        from core.portfolio import reconcile_positions
        from datetime import timezone

        pos = _make_position(
            ticker="ORPHAN", entry_price=100.0,
            stop_loss=95.0, take_profit=110.0,
            entry_time=datetime(2026, 4, 16, 9, 0, tzinfo=timezone.utc),
        )
        add_position(pos, db_path)

        fills = [{
            "ticker": "OTHER", "side": "SLD", "shares": 1.0, "price": 50.0,
            "time": datetime(2026, 4, 17, 13, 0, tzinfo=timezone.utc),
            "realized_pnl": 0.0,
        }]
        report = reconcile_positions(
            [], auto_fix=True, ibkr_fills=fills, db_path=db_path,
        )

        trades = get_trades(db_path=db_path)
        # Midpoint of SL 95 and TP 110 = 102.5
        assert trades[0].exit_price == pytest.approx(102.5)

    def test_ignores_fills_before_position_entry(self, db_path):
        """A fill from a prior, unrelated trade on the same ticker must
        not be used to close the current position."""
        from core.portfolio import reconcile_positions
        from datetime import timezone

        pos = _make_position(
            ticker="AAPL", entry_price=150.0,
            stop_loss=145.0, take_profit=159.0,
            entry_time=datetime(2026, 4, 17, 10, 0, tzinfo=timezone.utc),
        )
        add_position(pos, db_path)

        # Old fill from last week — belongs to a prior trade, ignore it
        fills = [{
            "ticker": "AAPL", "side": "SLD", "shares": 10.0, "price": 200.0,
            "time": datetime(2026, 4, 10, 10, 0, tzinfo=timezone.utc),
            "realized_pnl": 500.0,
        }]
        report = reconcile_positions(
            [], auto_fix=True, ibkr_fills=fills, db_path=db_path,
        )

        trades = get_trades(db_path=db_path)
        # Old fill ignored, falls back to SL/TP midpoint = (145+159)/2 = 152
        assert trades[0].exit_price == pytest.approx(152.0)

    def test_uses_most_recent_fill_when_multiple(self, db_path):
        """If multiple sell fills exist after entry (partial fills), use
        the most recent one's price."""
        from core.portfolio import reconcile_positions
        from datetime import timezone

        pos = _make_position(
            ticker="MULT", quantity=10, entry_price=100.0, stop_loss=95.0,
            entry_time=datetime(2026, 4, 17, 9, 0, tzinfo=timezone.utc),
        )
        add_position(pos, db_path)

        fills = [
            {"ticker": "MULT", "side": "SLD", "shares": 5.0, "price": 98.0,
             "time": datetime(2026, 4, 17, 10, 0, tzinfo=timezone.utc),
             "realized_pnl": -10.0},
            {"ticker": "MULT", "side": "SLD", "shares": 5.0, "price": 97.0,
             "time": datetime(2026, 4, 17, 11, 0, tzinfo=timezone.utc),
             "realized_pnl": -15.0},
        ]
        report = reconcile_positions(
            [], auto_fix=True, ibkr_fills=fills, db_path=db_path,
        )

        trades = get_trades(db_path=db_path)
        assert trades[0].exit_price == pytest.approx(97.0)

    def test_ignores_buy_fills(self, db_path):
        """BOT (buy) fills must not be used to close a long position."""
        from core.portfolio import reconcile_positions
        from datetime import timezone

        pos = _make_position(
            ticker="AAPL", entry_price=100.0,
            stop_loss=95.0, take_profit=110.0,
            entry_time=datetime(2026, 4, 17, 9, 0, tzinfo=timezone.utc),
        )
        add_position(pos, db_path)

        fills = [{
            "ticker": "AAPL", "side": "BOT", "shares": 10.0, "price": 110.0,
            "time": datetime(2026, 4, 17, 10, 0, tzinfo=timezone.utc),
            "realized_pnl": 0.0,
        }]
        report = reconcile_positions(
            [], auto_fix=True, ibkr_fills=fills, db_path=db_path,
        )

        trades = get_trades(db_path=db_path)
        # BOT fill ignored, falls back to SL/TP midpoint = (95+110)/2 = 102.5
        assert trades[0].exit_price == pytest.approx(102.5)


# ---------------------------------------------------------------------------
# TestPendingOrdersConfidenceColumn
# ---------------------------------------------------------------------------

class TestPendingOrdersConfidence:
    """AI confidence must be persisted per pending order to power eviction.

    Motivating bug (2026-04-22 paper run): when a stronger candidate appears
    mid-scan we need to evict the weakest pending BUY. The ranking metric is
    the AI confidence at placement time, so it must survive across restarts.
    """

    def test_save_pending_order_accepts_confidence(self, db_path):
        from core.portfolio import save_pending_order, get_pending_order_confidence
        save_pending_order(12345, "AAPL", confidence=72.5, db_path=db_path)
        assert get_pending_order_confidence(12345, db_path=db_path) == pytest.approx(72.5)

    def test_save_pending_order_confidence_is_optional(self, db_path):
        """Legacy callers that don't pass confidence must still work; column is NULL."""
        from core.portfolio import save_pending_order, get_pending_order_confidence
        save_pending_order(67890, "MSFT", db_path=db_path)
        assert get_pending_order_confidence(67890, db_path=db_path) is None

    def test_get_pending_order_confidence_missing_row_returns_none(self, db_path):
        from core.portfolio import get_pending_order_confidence
        assert get_pending_order_confidence(999, db_path=db_path) is None


class TestPendingOrdersMigration:
    """Existing DBs from prior releases must upgrade to include the confidence column."""

    def test_init_db_adds_confidence_column_to_legacy_pending_orders(self, tmp_path):
        """A DB that pre-dates the confidence column must be upgraded in place."""
        import sqlite3
        path = tmp_path / "legacy.db"
        # Build the *pre-migration* schema by hand
        with sqlite3.connect(str(path)) as conn:
            conn.execute(
                """CREATE TABLE pending_orders (
                    perm_id INTEGER PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    placed_at TEXT NOT NULL
                )"""
            )
            conn.execute(
                "INSERT INTO pending_orders (perm_id, ticker, placed_at) VALUES (?, ?, ?)",
                (111, "LEGACY", "2026-04-20T10:00:00+00:00"),
            )
            conn.commit()

        # Run migration (init_db must be idempotent and migrate the column)
        from core.portfolio import init_db, get_pending_order_time, get_pending_order_confidence
        init_db(path)

        # Existing row's placed_at must still be readable
        from datetime import datetime as _dt
        placed_at = get_pending_order_time(111, db_path=path)
        assert placed_at is not None
        assert placed_at.year == 2026 and placed_at.month == 4 and placed_at.day == 20

        # New confidence column exists and returns NULL for legacy rows
        assert get_pending_order_confidence(111, db_path=path) is None

        # Column is writable via the updated save_pending_order
        from core.portfolio import save_pending_order
        save_pending_order(222, "NEW", confidence=81.0, db_path=path)
        assert get_pending_order_confidence(222, db_path=path) == pytest.approx(81.0)

    def test_init_db_is_idempotent_on_already_migrated_schema(self, tmp_path):
        """Running init_db twice must not raise (no duplicate-column error)."""
        from core.portfolio import init_db
        path = tmp_path / "fresh.db"
        init_db(path)
        init_db(path)  # would raise 'duplicate column name: confidence' without the PRAGMA guard

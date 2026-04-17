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

    def test_auto_fix_uses_stop_loss_as_exit_price(self, db_path):
        """Auto-reconcile should use stop_loss as exit price, not entry_price.

        When a position disappears from IBKR (likely filled via stop-loss while
        bot was offline), recording exit at entry_price produces $0 P&L which
        hides the real loss. The stop_loss is the best available estimate of
        the actual fill price.
        """
        from core.portfolio import reconcile_positions

        pos = _make_position(
            ticker="SYRE", entry_price=150.0, stop_loss=145.5,
        )
        add_position(pos, db_path)

        # IBKR has nothing — position was stopped out while offline
        report = reconcile_positions([], auto_fix=True, db_path=db_path)

        assert report["auto_closed"] == ["SYRE"]
        trades = get_trades(db_path=db_path)
        assert len(trades) == 1
        # Exit price should be stop_loss (145.5), not entry_price (150.0)
        assert trades[0].exit_price == 145.5
        # P&L should reflect the loss: (145.5 - 150.0) * 10 = -$45
        assert trades[0].pnl == pytest.approx(-45.0)

    def test_auto_fix_uses_entry_price_when_no_stop_loss(self, db_path):
        """When stop_loss is 0, fall back to entry_price as exit."""
        from core.portfolio import reconcile_positions

        pos = _make_position(
            ticker="NOSTOP", entry_price=150.0, stop_loss=0.0,
        )
        add_position(pos, db_path)

        report = reconcile_positions([], auto_fix=True, db_path=db_path)
        trades = get_trades(db_path=db_path)
        assert len(trades) == 1
        # No stop_loss → falls back to entry_price
        assert trades[0].exit_price == 150.0

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

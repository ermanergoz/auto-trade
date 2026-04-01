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
        trade = close_position(
            "AAPL", exit_price=160.0,
            exit_time=datetime(2024, 1, 15, 15, 0),
            db_path=db_path,
        )

        assert trade is not None
        assert trade.ticker == "AAPL"
        assert trade.exit_price == 160.0
        assert trade.pnl == 100.0  # (160 - 150) * 10

        # Position should be removed
        positions = get_open_positions(db_path)
        assert len(positions) == 0

        # Trade should be recorded
        trades = get_trades(db_path=db_path)
        assert len(trades) == 1
        assert trades[0].pnl == 100.0

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

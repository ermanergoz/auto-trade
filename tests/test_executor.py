"""Tests for core/executor.py — focused on logic bugs around fill handling."""

from datetime import datetime, timezone

import pytest

from core.executor import handle_fill
from core.models import Action, Signal, TradeType
from core.portfolio import init_db, get_open_positions


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test_exec.db"
    init_db(path)
    return path


def _signal(action: Action) -> Signal:
    return Signal(
        ticker="AAPL",
        action=action,
        confidence=80.0,
        entry_price=150.0,
        stop_loss=145.0 if action == Action.BUY else 155.0,
        take_profit=160.0 if action == Action.BUY else 140.0,
        reasoning="test",
        source="ai",
        exchange="SMART",
        trade_type=TradeType.DAY,
        timestamp=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
    )


class TestHandleFillDirection:
    """Economic correctness: short entries must store negative quantity.

    IBKR reports filled quantity as positive regardless of BUY/SELL.
    For a SELL parent (short opening), the database must store a negative
    quantity so:
      - P&L calculation `(exit - entry) * qty` works correctly
      - reconcile_positions matches IBKR's signed position
      - close_position_market picks the right closing side
    """

    def test_long_entry_stores_positive_quantity(self, db_path):
        sig = _signal(Action.BUY)
        handle_fill(sig, quantity=10, fill_price=150.0, db_path=db_path)
        positions = get_open_positions(db_path)
        assert len(positions) == 1
        assert positions[0].quantity == 10

    def test_short_entry_stores_negative_quantity(self, db_path):
        sig = _signal(Action.SELL)
        handle_fill(sig, quantity=10, fill_price=150.0, db_path=db_path)
        positions = get_open_positions(db_path)
        assert len(positions) == 1, "Short entry should create a position"
        assert positions[0].quantity == -10, (
            "Short entry must store negative quantity so P&L math works; "
            f"got {positions[0].quantity}"
        )

    def test_invalid_quantity_ignored(self, db_path):
        sig = _signal(Action.BUY)
        result = handle_fill(sig, quantity=0, fill_price=150.0, db_path=db_path)
        assert result is None
        assert get_open_positions(db_path) == []

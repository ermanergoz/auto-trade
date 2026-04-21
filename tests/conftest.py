"""Shared test fixtures for the auto-trader test suite."""

from datetime import datetime

import pytest

from core.models import Signal, Position, Action, TradeType


def make_signal(**kwargs) -> Signal:
    """Create a Signal with sensible defaults for testing."""
    defaults = dict(
        ticker="AAPL",
        action=Action.BUY,
        confidence=85,
        entry_price=150.0,
        stop_loss=145.0,
        take_profit=165.0,
        reasoning="Test signal",
        source="ai",
        exchange="SMART",
    )
    defaults.update(kwargs)
    return Signal(**defaults)


def make_position(**kwargs) -> Position:
    """Create a Position with sensible defaults for testing."""
    defaults = dict(
        ticker="MSFT",
        exchange="SMART",
        quantity=10,
        entry_price=300.0,
        entry_time=datetime(2024, 1, 15),
        stop_loss=291.0,
        take_profit=318.0,
        trade_type=TradeType.DAY,
    )
    defaults.update(kwargs)
    return Position(**defaults)

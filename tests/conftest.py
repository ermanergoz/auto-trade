"""Shared test fixtures for the auto-trader test suite."""

from datetime import datetime

import pytest

from core.models import Signal, Position, Action, TradeType


@pytest.fixture(autouse=True)
def drain_gemini_rate_limiter():
    """Autouse across all tests: start each test with an empty Gemini rate-limit window.

    core.analyst._gemini_rate_limiter is a module-level singleton that appends
    a timestamp on every acquire(). Without draining between tests, any suite
    that makes >10 Gemini calls (the default GEMINI_RPM_LIMIT) in <60s will
    block on the 11th call for ~60s. Draining is cheap and idempotent.
    """
    from core import analyst as _a
    with _a._gemini_rate_limiter._lock:
        _a._gemini_rate_limiter._calls.clear()
    yield


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

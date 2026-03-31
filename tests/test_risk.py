"""Tests for core/risk.py."""

from datetime import datetime

import pytest

from core.models import Signal, Position, Action, TradeType
from core.risk import (
    check_position_size,
    check_daily_loss_limit,
    check_max_positions,
    check_stop_loss,
    check_no_duplicate,
    calculate_position_size,
    evaluate,
)


def _make_signal(**kwargs) -> Signal:
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


def _make_position(**kwargs) -> Position:
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


class TestPositionSize:
    def test_passes_normal(self):
        sig = _make_signal(entry_price=150.0)
        ok, _ = check_position_size(sig, 100_000)
        assert ok is True

    def test_fails_zero_portfolio(self):
        sig = _make_signal()
        ok, reason = check_position_size(sig, 0)
        assert ok is False
        assert "zero" in reason.lower()


class TestDailyLossLimit:
    def test_passes_within_limit(self):
        ok, _ = check_daily_loss_limit(-500, 100_000)
        assert ok is True

    def test_fails_beyond_limit(self):
        ok, reason = check_daily_loss_limit(-3000, 100_000)
        assert ok is False
        assert "halted" in reason.lower()

    def test_passes_positive_pnl(self):
        ok, _ = check_daily_loss_limit(500, 100_000)
        assert ok is True


class TestMaxPositions:
    def test_passes_under_limit(self):
        positions = [_make_position() for _ in range(5)]
        ok, _ = check_max_positions(positions)
        assert ok is True

    def test_fails_at_limit(self):
        positions = [_make_position() for _ in range(10)]
        ok, reason = check_max_positions(positions)
        assert ok is False
        assert "10/10" in reason


class TestStopLoss:
    def test_valid_buy_stop(self):
        sig = _make_signal(action=Action.BUY, entry_price=150, stop_loss=145)
        ok, _ = check_stop_loss(sig)
        assert ok is True

    def test_invalid_buy_stop_above_entry(self):
        sig = _make_signal(action=Action.BUY, entry_price=150, stop_loss=155)
        ok, reason = check_stop_loss(sig)
        assert ok is False
        assert "below" in reason.lower()

    def test_valid_sell_stop(self):
        sig = _make_signal(action=Action.SELL, entry_price=150, stop_loss=155)
        ok, _ = check_stop_loss(sig)
        assert ok is True

    def test_no_stop_loss(self):
        sig = _make_signal(stop_loss=0)
        ok, reason = check_stop_loss(sig)
        assert ok is False


class TestNoDuplicate:
    def test_no_existing(self):
        sig = _make_signal(ticker="AAPL")
        ok, _ = check_no_duplicate(sig, [_make_position(ticker="MSFT")])
        assert ok is True

    def test_duplicate_blocked(self):
        sig = _make_signal(ticker="AAPL")
        ok, reason = check_no_duplicate(sig, [_make_position(ticker="AAPL")])
        assert ok is False
        assert "Already holding" in reason


class TestPositionSizing:
    def test_basic_sizing(self):
        sig = _make_signal(entry_price=150.0, stop_loss=145.0)
        qty = calculate_position_size(sig, 100_000)
        assert qty > 0
        # Max position 5% = $5000 / $150 = 33 shares
        assert qty <= 33

    def test_zero_portfolio(self):
        sig = _make_signal()
        qty = calculate_position_size(sig, 0)
        assert qty == 0


class TestFullEvaluation:
    def test_approved(self):
        sig = _make_signal()
        result = evaluate(sig, [], 100_000, 0)
        assert result.approved is True
        assert result.position_size > 0
        assert result.reasons == []

    def test_rejected_daily_loss(self):
        sig = _make_signal()
        result = evaluate(sig, [], 100_000, -3000)
        assert result.approved is False
        assert any("halted" in r.lower() for r in result.reasons)

    def test_rejected_duplicate(self):
        sig = _make_signal(ticker="AAPL")
        positions = [_make_position(ticker="AAPL")]
        result = evaluate(sig, positions, 100_000, 0)
        assert result.approved is False

    def test_rejected_max_positions(self):
        sig = _make_signal()
        positions = [_make_position(ticker=f"STK{i}") for i in range(10)]
        result = evaluate(sig, positions, 100_000, 0)
        assert result.approved is False

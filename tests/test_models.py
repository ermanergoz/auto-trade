"""Tests for core/models.py."""

from datetime import datetime

from core.models import (
    Signal, Position, Trade, DailySummary, StockInfo,
    Action, TradeType, Market,
)


def test_signal_creation():
    sig = Signal(
        ticker="AAPL",
        action=Action.BUY,
        confidence=85.0,
        entry_price=150.0,
        stop_loss=145.5,
        take_profit=160.0,
        reasoning="MACD crossover with volume spike",
        source="screener",
        exchange="SMART",
    )
    assert sig.ticker == "AAPL"
    assert sig.action == Action.BUY
    assert sig.confidence == 85.0
    assert sig.source == "screener"
    assert isinstance(sig.timestamp, datetime)
    assert sig.indicator_values == {}


def test_position_pnl():
    pos = Position(
        ticker="MSFT",
        exchange="SMART",
        quantity=10,
        entry_price=300.0,
        entry_time=datetime(2024, 1, 15, 10, 30),
        stop_loss=291.0,
        take_profit=318.0,
        trade_type=TradeType.DAY,
        current_price=310.0,
    )
    assert pos.unrealized_pnl == 100.0  # (310 - 300) * 10
    assert abs(pos.unrealized_pnl_pct - 3.333) < 0.01


def test_position_pnl_none_when_no_current_price():
    pos = Position(
        ticker="MSFT",
        exchange="SMART",
        quantity=10,
        entry_price=300.0,
        entry_time=datetime(2024, 1, 15),
        stop_loss=291.0,
        take_profit=318.0,
        trade_type=TradeType.SWING,
    )
    assert pos.unrealized_pnl is None
    assert pos.unrealized_pnl_pct is None


def test_trade_pnl():
    trade = Trade(
        ticker="THYAO",
        exchange="BIST",
        quantity=100,
        entry_price=200.0,
        exit_price=220.0,
        entry_time=datetime(2024, 1, 15, 10, 0),
        exit_time=datetime(2024, 1, 15, 14, 0),
        trade_type=TradeType.DAY,
        sector="Industrials",
    )
    assert trade.pnl == 2000.0  # (220 - 200) * 100
    assert trade.pnl_pct == 10.0
    assert trade.duration == 4.0  # 4 hours


def test_trade_losing():
    trade = Trade(
        ticker="AAPL",
        exchange="SMART",
        quantity=5,
        entry_price=150.0,
        exit_price=140.0,
        entry_time=datetime(2024, 1, 15, 10, 0),
        exit_time=datetime(2024, 1, 16, 10, 0),
        trade_type=TradeType.SWING,
    )
    assert trade.pnl == -50.0
    assert abs(trade.pnl_pct - (-6.667)) < 0.01
    assert trade.duration == 24.0


def test_enums():
    assert Action.BUY.value == "buy"
    assert Action.SELL.value == "sell"
    assert Action.HOLD.value == "hold"
    assert TradeType.DAY.value == "day"
    assert TradeType.SWING.value == "swing"
    assert Market.US.value == "US"
    assert Market.BIST.value == "BIST"


def test_stock_info():
    info = StockInfo(
        ticker="THYAO",
        exchange="BIST",
        sector="Industrials",
        market_cap=5_000_000_000,
        avg_volume=10_000_000,
        currency="TRY",
        name="Turkish Airlines",
    )
    assert info.ticker == "THYAO"
    assert info.currency == "TRY"

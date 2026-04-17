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
        ticker="NVDA",
        exchange="SMART",
        quantity=100,
        entry_price=200.0,
        exit_price=220.0,
        entry_time=datetime(2024, 1, 15, 10, 0),
        exit_time=datetime(2024, 1, 15, 14, 0),
        trade_type=TradeType.DAY,
        sector="Technology",
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


def test_trade_duration_mixed_tz():
    """A Trade with one tz-aware and one tz-naive datetime must not crash.

    Backtest deserializers and DB rows can yield either flavor; the property
    should normalize before subtracting.
    """
    from datetime import timezone
    trade = Trade(
        ticker="AAPL",
        exchange="SMART",
        quantity=1,
        entry_price=100.0,
        exit_price=110.0,
        entry_time=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
        exit_time=datetime(2024, 1, 15, 14, 0),  # naive
        trade_type=TradeType.DAY,
    )
    # Must not raise TypeError on naive/aware subtraction
    assert trade.duration == 4.0


def test_enums():
    assert Action.BUY.value == "buy"
    assert Action.SELL.value == "sell"
    assert Action.HOLD.value == "hold"
    assert TradeType.DAY.value == "day"
    assert TradeType.SWING.value == "swing"
    assert Market.US.value == "US"


def test_stock_info():
    info = StockInfo(
        ticker="AAPL",
        exchange="SMART",
        sector="Technology",
        market_cap=3_000_000_000_000,
        avg_volume=50_000_000,
        currency="USD",
        name="Apple Inc",
    )
    assert info.ticker == "AAPL"
    assert info.currency == "USD"


# ---------------------------------------------------------------------------
# CSV Logger Tests
# ---------------------------------------------------------------------------

class TestCSVLoggerAction:
    """log_trade_to_csv must record the correct entry action, not always 'BUY'."""

    def test_short_trade_records_sell_action(self):
        """A short trade (negative quantity) should record action as SELL."""
        import csv
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from core.logger import log_trade_to_csv

        trade = Trade(
            ticker="TSLA",
            exchange="SMART",
            quantity=-10,  # short position
            entry_price=200.0,
            exit_price=180.0,
            entry_time=datetime(2024, 1, 15, 10, 0),
            exit_time=datetime(2024, 1, 15, 14, 0),
            trade_type=TradeType.DAY,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "trades_2024-01-15.csv"
            with patch("core.logger._trades_csv_path", return_value=csv_path):
                log_trade_to_csv(trade)

            with open(csv_path) as f:
                reader = csv.DictReader(f)
                row = next(reader)
                assert row["action"] == "SELL", (
                    f"Short trade should record action as SELL, got '{row['action']}'"
                )

    def test_long_trade_records_buy_action(self):
        """A long trade (positive quantity) should record action as BUY."""
        import csv
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from core.logger import log_trade_to_csv

        trade = Trade(
            ticker="AAPL",
            exchange="SMART",
            quantity=10,  # long position
            entry_price=150.0,
            exit_price=160.0,
            entry_time=datetime(2024, 1, 15, 10, 0),
            exit_time=datetime(2024, 1, 15, 14, 0),
            trade_type=TradeType.DAY,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "trades_2024-01-15.csv"
            with patch("core.logger._trades_csv_path", return_value=csv_path):
                log_trade_to_csv(trade)

            with open(csv_path) as f:
                reader = csv.DictReader(f)
                row = next(reader)
                assert row["action"] == "BUY", (
                    f"Long trade should record action as BUY, got '{row['action']}'"
                )

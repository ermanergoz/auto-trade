"""Tests for backtest/engine.py and backtest/report.py."""

from datetime import datetime, date

import pytest

from core.models import Trade, TradeType
from backtest.engine import SimulatedPortfolio, BacktestConfig
from backtest.report import calculate_metrics, _calculate_max_drawdown, _std
from tests.conftest import make_signal as _make_signal


class TestSimulatedPortfolio:
    def test_initial_state(self):
        p = SimulatedPortfolio(initial_capital=50_000)
        assert p.cash == 50_000
        assert p.portfolio_value == 50_000
        assert p.positions == []
        assert p.trades == []

    def test_open_position(self):
        p = SimulatedPortfolio(initial_capital=100_000)
        sig = _make_signal(entry_price=100.0)
        p.open_position(sig, 10, 100.0, datetime(2024, 1, 15))
        assert len(p.positions) == 1
        assert p.positions[0].ticker == "AAPL"
        assert p.cash < 100_000  # deducted cost + slippage + commission

    def test_close_position(self):
        p = SimulatedPortfolio(initial_capital=100_000)
        sig = _make_signal(entry_price=100.0)
        p.open_position(sig, 10, 100.0, datetime(2024, 1, 15))
        trade = p.close_position("AAPL", 110.0, datetime(2024, 1, 16))
        assert trade is not None
        assert trade.pnl > 0  # profitable
        assert len(p.positions) == 0
        assert len(p.trades) == 1

    def test_close_nonexistent(self):
        p = SimulatedPortfolio()
        result = p.close_position("NOPE", 100, datetime.now())
        assert result is None

    def test_insufficient_cash(self):
        p = SimulatedPortfolio(initial_capital=100)
        sig = _make_signal(entry_price=1000.0)
        p.open_position(sig, 10, 1000.0, datetime(2024, 1, 15))
        assert len(p.positions) == 0  # should not open

    def test_equity_recording(self):
        p = SimulatedPortfolio(initial_capital=100_000)
        p.record_equity(date(2024, 1, 15))
        p.record_equity(date(2024, 1, 16))
        assert len(p.equity_curve) == 2
        assert p.equity_curve[0][1] == 100_000


class TestMetrics:
    def _make_trades(self):
        return [
            Trade("AAPL", "SMART", 10, 150, 160, datetime(2024, 1, 1),
                  datetime(2024, 1, 2), TradeType.DAY),
            Trade("MSFT", "SMART", 5, 300, 290, datetime(2024, 1, 3),
                  datetime(2024, 1, 4), TradeType.DAY),
            Trade("GOOGL", "SMART", 8, 140, 155, datetime(2024, 1, 5),
                  datetime(2024, 1, 6), TradeType.SWING),
        ]

    def test_basic_metrics(self):
        trades = self._make_trades()
        equity = [(date(2024, 1, i), 100_000 + i * 50) for i in range(1, 7)]
        m = calculate_metrics(trades, equity, 100_000)

        assert m["num_trades"] == 3
        assert m["winning_trades"] == 2
        assert m["losing_trades"] == 1
        assert m["win_rate_pct"] > 60
        assert m["total_pnl"] == (100 + (-50) + 120)  # 170
        assert m["final_value"] > 0

    def test_empty_trades(self):
        m = calculate_metrics([], [], 100_000)
        assert m["num_trades"] == 0
        assert m["total_return_pct"] == 0
        assert m["final_value"] == 100_000

    def test_max_drawdown(self):
        curve = [
            (date(2024, 1, 1), 100_000),
            (date(2024, 1, 2), 105_000),
            (date(2024, 1, 3), 95_000),  # 9.5% dd from peak
            (date(2024, 1, 4), 98_000),
        ]
        dd = _calculate_max_drawdown(curve)
        assert abs(dd - 9.52) < 0.1  # ~9.52% drawdown

    def test_std(self):
        assert _std([]) == 0.0
        assert _std([5.0]) == 0.0
        assert _std([1, 2, 3, 4, 5]) > 0


class TestGapDownStopLoss:
    """Verify stop-loss uses min(low, stop_loss) for gap-down modeling."""

    def test_gap_down_fills_at_low(self):
        import pandas as pd
        from backtest.engine import SimulatedPortfolio, _check_exits
        from core.models import Position, TradeType

        portfolio = SimulatedPortfolio(initial_capital=100_000)
        # Position with stop-loss at $95
        portfolio.positions.append(Position(
            ticker="GAP", exchange="SMART", quantity=100,
            entry_price=100.0, entry_time=datetime(2024, 1, 1),
            stop_loss=95.0, take_profit=110.0, trade_type=TradeType.DAY,
        ))

        # Day bar gaps down to $90 (below stop of $95)
        day_data = {
            "GAP": pd.Series({"open": 91.0, "high": 92.0, "low": 90.0, "close": 91.0}),
        }

        _check_exits(portfolio, day_data, datetime(2024, 1, 2))

        assert len(portfolio.positions) == 0, "Position should be closed"
        assert len(portfolio.trades) == 1
        # Fill should be at $90 (the low), not $95 (the stop price)
        assert portfolio.trades[0].exit_price < 95.0, (
            f"Gap-down should fill below stop price, got {portfolio.trades[0].exit_price}"
        )


class TestDailyPnlBaseline:
    """Verify daily PnL calculation is clean."""

    def test_daily_pnl_zero_when_no_equity_history(self):
        p = SimulatedPortfolio(initial_capital=100_000)
        assert p.daily_pnl == 0.0

    def test_daily_pnl_after_one_day(self):
        p = SimulatedPortfolio(initial_capital=100_000)
        p.record_equity(date(2024, 1, 1))
        # Cash unchanged, so PnL should be 0
        assert p.daily_pnl == 0.0


class TestLookAheadBias:
    """Verify screener only sees data strictly before current date."""

    def test_stock_data_excludes_current_date(self):
        """The backtest must feed the screener data < current_date, not <=."""
        import pandas as pd
        import inspect
        from backtest.engine import run_backtest

        # Check the source code for the correct comparison operator
        source = inspect.getsource(run_backtest)
        # Should use < not <= for date filtering in stock_data construction
        assert "df.index.date < current_date" in source or "df.index < current_date" in source, (
            "Backtester must use strict < for date filtering to avoid look-ahead bias"
        )

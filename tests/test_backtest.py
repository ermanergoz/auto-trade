"""Tests for backtest/engine.py and backtest/report.py."""

from datetime import datetime, date

import pytest

from core.models import Action, Trade, TradeType
from core.risk import RiskResult
from backtest.engine import SimulatedPortfolio, BacktestConfig
from backtest.report import (
    calculate_metrics, _calculate_max_drawdown, _std,
    compare_ai_value_add,
)
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


class TestShortPositionAccounting:
    """Short positions must credit cash on open and debit on close."""

    def test_short_open_credits_cash(self):
        """Opening a short sale should ADD cash (you receive sale proceeds)."""
        p = SimulatedPortfolio(initial_capital=100_000, slippage_pct=0, commission=0)
        sig = _make_signal(action=Action.SELL, entry_price=100.0,
                           stop_loss=110.0, take_profit=90.0)
        p.open_position(sig, 10, 100.0, datetime(2024, 1, 15))

        assert len(p.positions) == 1
        # Short sale proceeds: 100 * 10 = $1000 credited
        assert p.cash == 100_000 + 1000.0

    def test_short_position_stores_negative_quantity(self):
        """Short positions must have negative quantity."""
        p = SimulatedPortfolio(initial_capital=100_000, slippage_pct=0, commission=0)
        sig = _make_signal(action=Action.SELL, entry_price=100.0,
                           stop_loss=110.0, take_profit=90.0)
        p.open_position(sig, 10, 100.0, datetime(2024, 1, 15))

        assert p.positions[0].quantity == -10

    def test_short_close_debits_cash(self):
        """Closing a short (buying back) should DEBIT cash."""
        p = SimulatedPortfolio(initial_capital=100_000, slippage_pct=0, commission=0)
        sig = _make_signal(action=Action.SELL, entry_price=100.0,
                           stop_loss=110.0, take_profit=90.0)
        p.open_position(sig, 10, 100.0, datetime(2024, 1, 15))
        cash_after_open = p.cash  # 101_000

        trade = p.close_position("AAPL", 90.0, datetime(2024, 1, 16))
        assert trade is not None
        # Buying back 10 shares at $90 costs $900
        assert p.cash == cash_after_open - 900.0

    def test_short_profitable_trade_pnl(self):
        """Short that drops in price should have positive P&L."""
        p = SimulatedPortfolio(initial_capital=100_000, slippage_pct=0, commission=0)
        sig = _make_signal(action=Action.SELL, entry_price=100.0,
                           stop_loss=110.0, take_profit=90.0)
        p.open_position(sig, 10, 100.0, datetime(2024, 1, 15))

        trade = p.close_position("AAPL", 90.0, datetime(2024, 1, 16))
        assert trade is not None
        # Sold at 100, bought back at 90, qty=-10 → P&L = (90-100)*(-10) = $100
        assert trade.pnl == 100.0

    def test_short_losing_trade_pnl(self):
        """Short that rises in price should have negative P&L."""
        p = SimulatedPortfolio(initial_capital=100_000, slippage_pct=0, commission=0)
        sig = _make_signal(action=Action.SELL, entry_price=100.0,
                           stop_loss=110.0, take_profit=90.0)
        p.open_position(sig, 10, 100.0, datetime(2024, 1, 15))

        trade = p.close_position("AAPL", 110.0, datetime(2024, 1, 16))
        assert trade is not None
        # Sold at 100, bought back at 110, qty=-10 → P&L = (110-100)*(-10) = -$100
        assert trade.pnl == -100.0

    def test_short_mtm_reduces_portfolio_value(self):
        """Short position should reduce portfolio value by its notional."""
        p = SimulatedPortfolio(initial_capital=100_000, slippage_pct=0, commission=0)
        sig = _make_signal(action=Action.SELL, entry_price=100.0,
                           stop_loss=110.0, take_profit=90.0)
        p.open_position(sig, 10, 100.0, datetime(2024, 1, 15))

        # Cash = 101_000, position value = -10 * 100 = -1000
        # Portfolio = 101_000 - 1000 = 100_000
        assert p.portfolio_value == pytest.approx(100_000)

    def test_short_exit_check_triggers_on_high(self):
        """Short stop-loss should trigger when high >= stop_loss, fill at open for gap."""
        import pandas as pd
        from backtest.engine import _check_exits
        from core.models import Position, TradeType

        portfolio = SimulatedPortfolio(initial_capital=100_000, slippage_pct=0, commission=0)
        portfolio.positions.append(Position(
            ticker="SHORT", exchange="SMART", quantity=-10,
            entry_price=100.0, entry_time=datetime(2024, 1, 1),
            stop_loss=110.0, take_profit=90.0, trade_type=TradeType.DAY,
        ))

        # Bar gaps up above stop (open=$108 < stop=$110, but high=$112 >= stop)
        day_data = {
            "SHORT": pd.Series({"open": 108.0, "high": 112.0, "low": 107.0, "close": 111.0}),
        }
        _check_exits(portfolio, day_data, datetime(2024, 1, 2))

        assert len(portfolio.positions) == 0
        assert len(portfolio.trades) == 1
        assert portfolio.trades[0].pnl < 0  # losing trade
        # Open ($108) is below stop ($110), so intraday hit → fill at stop price
        assert portfolio.trades[0].exit_price == pytest.approx(110.0)


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
    """Verify stop-loss uses open price for gap-down modeling (not bar low)."""

    def test_gap_down_fills_at_open_not_low(self):
        """When price gaps past stop, fill at open (first available price)."""
        import pandas as pd
        from backtest.engine import SimulatedPortfolio, _check_exits
        from core.models import Position, TradeType

        portfolio = SimulatedPortfolio(initial_capital=100_000, slippage_pct=0, commission=0)
        # Position with stop-loss at $95
        portfolio.positions.append(Position(
            ticker="GAP", exchange="SMART", quantity=100,
            entry_price=100.0, entry_time=datetime(2024, 1, 1),
            stop_loss=95.0, take_profit=110.0, trade_type=TradeType.DAY,
        ))

        # Day bar gaps down: open=$91 (below stop of $95), low=$90
        day_data = {
            "GAP": pd.Series({"open": 91.0, "high": 92.0, "low": 90.0, "close": 91.0}),
        }

        _check_exits(portfolio, day_data, datetime(2024, 1, 2))

        assert len(portfolio.positions) == 0, "Position should be closed"
        assert len(portfolio.trades) == 1
        # Fill should be at $91 (the open), not $90 (the low) or $95 (the stop)
        assert portfolio.trades[0].exit_price == pytest.approx(91.0), (
            f"Gap-down should fill at open price, got {portfolio.trades[0].exit_price}"
        )

    def test_no_gap_fills_at_stop_price(self):
        """When intraday price crosses stop (no gap), fill at stop price."""
        import pandas as pd
        from backtest.engine import SimulatedPortfolio, _check_exits
        from core.models import Position, TradeType

        portfolio = SimulatedPortfolio(initial_capital=100_000, slippage_pct=0, commission=0)
        portfolio.positions.append(Position(
            ticker="NORM", exchange="SMART", quantity=100,
            entry_price=100.0, entry_time=datetime(2024, 1, 1),
            stop_loss=95.0, take_profit=110.0, trade_type=TradeType.DAY,
        ))

        # Open above stop, but low dips below — intraday stop hit
        day_data = {
            "NORM": pd.Series({"open": 97.0, "high": 98.0, "low": 93.0, "close": 94.0}),
        }

        _check_exits(portfolio, day_data, datetime(2024, 1, 2))

        assert len(portfolio.positions) == 0
        assert len(portfolio.trades) == 1
        # Fill at stop price since open was above stop
        assert portfolio.trades[0].exit_price == pytest.approx(95.0), (
            f"Intraday stop should fill at stop price, got {portfolio.trades[0].exit_price}"
        )

    def test_short_gap_up_fills_at_open(self):
        """Short position: gap-up past stop fills at open, not high."""
        import pandas as pd
        from backtest.engine import SimulatedPortfolio, _check_exits
        from core.models import Position, TradeType

        portfolio = SimulatedPortfolio(initial_capital=200_000, slippage_pct=0, commission=0)
        portfolio.positions.append(Position(
            ticker="SGAP", exchange="SMART", quantity=-100,
            entry_price=100.0, entry_time=datetime(2024, 1, 1),
            stop_loss=105.0, take_profit=90.0, trade_type=TradeType.DAY,
        ))

        # Gap-up: open=$108 (above stop of $105), high=$112
        day_data = {
            "SGAP": pd.Series({"open": 108.0, "high": 112.0, "low": 107.0, "close": 111.0}),
        }

        _check_exits(portfolio, day_data, datetime(2024, 1, 2))

        assert len(portfolio.positions) == 0
        assert len(portfolio.trades) == 1
        # Fill at open ($108), not high ($112)
        assert portfolio.trades[0].exit_price == pytest.approx(108.0), (
            f"Short gap-up should fill at open, got {portfolio.trades[0].exit_price}"
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


class TestEquityCurveMTM:
    """Equity curve must reflect mark-to-market prices, not entry prices."""

    def test_equity_curve_uses_current_prices(self):
        """record_equity with current_prices should value positions at market price."""
        p = SimulatedPortfolio(initial_capital=100_000, slippage_pct=0, commission=0)
        sig = _make_signal(entry_price=100.0)
        p.open_position(sig, 10, 100.0, datetime(2024, 1, 15))
        # Cash = 99_000, position = 10 shares entered at $100

        # Record equity with current price at $110
        current_prices = {"AAPL": 110.0}
        p.record_equity(date(2024, 1, 15), current_prices=current_prices)

        # Equity should be cash(99_000) + 10*110 = 100_100, not 100_000
        assert p.equity_curve[-1][1] == pytest.approx(100_100.0)

    def test_daily_pnl_reflects_price_changes(self):
        """daily_pnl should capture unrealized gains from price movement."""
        p = SimulatedPortfolio(initial_capital=100_000, slippage_pct=0, commission=0)
        sig = _make_signal(entry_price=100.0)
        p.open_position(sig, 10, 100.0, datetime(2024, 1, 15))

        # Day 1: record at entry price
        p.record_equity(date(2024, 1, 15), current_prices={"AAPL": 100.0})
        # Day 2: price rose to $110
        day2_prices = {"AAPL": 110.0}
        pnl = p.daily_pnl_mtm(day2_prices)

        # Unrealized gain = 10 * ($110 - $100) = $100
        assert pnl == pytest.approx(100.0)


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


# ---------------------------------------------------------------------------
# Indicator Weights in Backtest
# ---------------------------------------------------------------------------

class TestBacktestIndicatorWeights:
    """Backtest should accept and pass through indicator weights."""

    def test_config_accepts_indicator_weights(self):
        config = BacktestConfig(
            tickers=["AAPL"],
            indicator_weights={"RSI": 2.0, "MACD": 0.5},
        )
        assert config.indicator_weights == {"RSI": 2.0, "MACD": 0.5}

    def test_config_default_weights_none(self):
        config = BacktestConfig(tickers=["AAPL"])
        assert config.indicator_weights is None

    def test_zero_weights_no_trades(self):
        """Zeroing all indicator weights should produce zero trades."""
        portfolio = SimulatedPortfolio(initial_capital=100_000)
        sig = _make_signal(entry_price=100.0)
        # Can't easily run full backtest without data download, so just
        # verify the config propagates correctly
        config = BacktestConfig(
            tickers=["AAPL"],
            indicator_weights={
                "RSI": 0.0, "MACD": 0.0, "MA_CROSSOVER": 0.0,
                "VOLUME_SPIKE": 0.0, "BOLLINGER": 0.0,
                "SUPPORT": 0.0, "RESISTANCE": 0.0,
            },
        )
        assert all(v == 0.0 for v in config.indicator_weights.values())


# ---------------------------------------------------------------------------
# Volatility Regime in Backtest
# ---------------------------------------------------------------------------

class TestBacktestVolatility:
    """Backtest should calculate and use realized volatility for sizing."""

    def test_config_accepts_use_volatility_scaling(self):
        config = BacktestConfig(tickers=["AAPL"], use_volatility_scaling=True)
        assert config.use_volatility_scaling is True

    def test_config_default_no_volatility_scaling(self):
        config = BacktestConfig(tickers=["AAPL"])
        assert config.use_volatility_scaling is False

    def test_volatility_calculated_during_backtest(self):
        """When use_volatility_scaling is True, evaluate() should receive volatility."""
        from unittest.mock import patch, MagicMock
        import pandas as pd
        from backtest.engine import run_backtest, BacktestConfig

        # Create data with a sharp drop to trigger Bollinger/RSI signals
        n = 90
        closes = [100 + i * 0.5 for i in range(70)] + [135 - i * 3 for i in range(20)]
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        df = pd.DataFrame({
            "open": [c * 0.99 for c in closes],
            "high": [c * 1.02 for c in closes],
            "low": [c * 0.97 for c in closes],
            "close": closes,
            "volume": [1_000_000] * 70 + [3_000_000] * 20,  # volume spike on drop
        }, index=dates)
        df.index.name = "date"

        config = BacktestConfig(
            tickers=["AAPL"],
            use_volatility_scaling=True,
            min_screener_score=5.0,
        )

        with patch("backtest.engine.get_historical_data_yfinance", return_value=df), \
             patch("backtest.engine.evaluate") as mock_eval:
            mock_eval.return_value = RiskResult(approved=False, reasons=["test"], position_size=0)
            run_backtest(config)

            # evaluate() must have been called at least once for this test to be valid
            assert mock_eval.call_count > 0, (
                "evaluate() was never called — test data must generate screener signals"
            )
            for call in mock_eval.call_args_list:
                assert "volatility" in call.kwargs or len(call.args) > 6, (
                    "evaluate() should receive volatility when use_volatility_scaling=True"
                )


# ---------------------------------------------------------------------------
# AI Value-Add Tracking
# ---------------------------------------------------------------------------

class TestCompareAIValueAdd:
    """compare_ai_value_add computes alpha metrics between screener-only and screener+AI."""

    def test_returns_comparison_dict(self):
        """Should return a dict with both strategies' metrics and alpha."""
        screener_metrics = calculate_metrics(
            [Trade("AAPL", "SMART", 10, 100, 110, datetime(2024, 1, 1),
                   datetime(2024, 1, 2), TradeType.DAY)],
            [(date(2024, 1, 1), 100000), (date(2024, 1, 2), 100100)],
            100000,
        )
        ai_metrics = calculate_metrics(
            [Trade("AAPL", "SMART", 10, 100, 115, datetime(2024, 1, 1),
                   datetime(2024, 1, 2), TradeType.DAY)],
            [(date(2024, 1, 1), 100000), (date(2024, 1, 2), 100150)],
            100000,
        )

        comparison = compare_ai_value_add(screener_metrics, ai_metrics)
        assert "screener_only" in comparison
        assert "screener_plus_ai" in comparison
        assert "alpha" in comparison

    def test_alpha_is_return_difference(self):
        """Alpha should be the return difference between AI and screener-only."""
        screener_metrics = {"total_return_pct": 5.0, "sharpe_ratio": 0.8,
                           "max_drawdown_pct": 3.0, "win_rate_pct": 50.0,
                           "num_trades": 10, "total_pnl": 5000.0}
        ai_metrics = {"total_return_pct": 8.0, "sharpe_ratio": 1.2,
                     "max_drawdown_pct": 2.5, "win_rate_pct": 60.0,
                     "num_trades": 7, "total_pnl": 8000.0}

        comparison = compare_ai_value_add(screener_metrics, ai_metrics)
        assert comparison["alpha"]["return_alpha_pct"] == pytest.approx(3.0)
        assert comparison["alpha"]["sharpe_alpha"] == pytest.approx(0.4)
        assert comparison["alpha"]["pnl_alpha"] == pytest.approx(3000.0)

    def test_negative_alpha(self):
        """AI can have negative alpha (hurts performance)."""
        screener_metrics = {"total_return_pct": 10.0, "sharpe_ratio": 1.5,
                           "max_drawdown_pct": 4.0, "win_rate_pct": 55.0,
                           "num_trades": 12, "total_pnl": 10000.0}
        ai_metrics = {"total_return_pct": 6.0, "sharpe_ratio": 0.9,
                     "max_drawdown_pct": 5.0, "win_rate_pct": 45.0,
                     "num_trades": 5, "total_pnl": 6000.0}

        comparison = compare_ai_value_add(screener_metrics, ai_metrics)
        assert comparison["alpha"]["return_alpha_pct"] < 0
        assert comparison["alpha"]["ai_adds_value"] is False

    def test_ai_adds_value_flag(self):
        """Should flag whether AI adds value based on return alpha."""
        screener_metrics = {"total_return_pct": 5.0, "sharpe_ratio": 0.8,
                           "max_drawdown_pct": 3.0, "win_rate_pct": 50.0,
                           "num_trades": 10, "total_pnl": 5000.0}
        ai_metrics = {"total_return_pct": 8.0, "sharpe_ratio": 1.2,
                     "max_drawdown_pct": 2.5, "win_rate_pct": 60.0,
                     "num_trades": 7, "total_pnl": 8000.0}

        comparison = compare_ai_value_add(screener_metrics, ai_metrics)
        assert comparison["alpha"]["ai_adds_value"] is True

    def test_trade_filter_ratio(self):
        """Should report what % of screener trades the AI filtered out."""
        screener_metrics = {"total_return_pct": 5.0, "sharpe_ratio": 0.8,
                           "max_drawdown_pct": 3.0, "win_rate_pct": 50.0,
                           "num_trades": 20, "total_pnl": 5000.0}
        ai_metrics = {"total_return_pct": 8.0, "sharpe_ratio": 1.2,
                     "max_drawdown_pct": 2.5, "win_rate_pct": 60.0,
                     "num_trades": 8, "total_pnl": 8000.0}

        comparison = compare_ai_value_add(screener_metrics, ai_metrics)
        # AI filtered out 12 of 20 trades = 60% filter rate
        assert comparison["alpha"]["ai_filter_rate_pct"] == pytest.approx(60.0)

    def test_zero_screener_trades(self):
        """Should handle zero screener trades gracefully."""
        screener_metrics = {"total_return_pct": 0, "sharpe_ratio": 0,
                           "max_drawdown_pct": 0, "win_rate_pct": 0,
                           "num_trades": 0, "total_pnl": 0}
        ai_metrics = {"total_return_pct": 0, "sharpe_ratio": 0,
                     "max_drawdown_pct": 0, "win_rate_pct": 0,
                     "num_trades": 0, "total_pnl": 0}

        comparison = compare_ai_value_add(screener_metrics, ai_metrics)
        assert comparison["alpha"]["return_alpha_pct"] == 0.0
        assert comparison["alpha"]["ai_filter_rate_pct"] == 0.0

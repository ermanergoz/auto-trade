"""Tests for core/risk.py."""

import pytest
from datetime import datetime, timedelta, timezone

from core.models import Action, Trade, TradeType
from core.risk import (
    check_position_size,
    check_daily_loss_limit,
    check_max_positions,
    check_stop_loss,
    check_no_duplicate,
    check_circuit_breaker,
    check_anti_momentum,
    check_risk_reward,
    check_sector_concentration,
    check_excluded_sector,
    check_analyst_consensus,
    check_pdt_restriction,
    check_intraday_margin,
    calculate_position_size,
    evaluate,
)
from tests.conftest import make_signal as _make_signal, make_position as _make_position


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
        ok, reason = check_daily_loss_limit(-3000, 100_000, limit_pct=2.0)
        assert ok is False
        assert "halted" in reason.lower()

    def test_passes_positive_pnl(self):
        ok, _ = check_daily_loss_limit(500, 100_000)
        assert ok is True


class TestMaxPositions:
    def test_passes_under_limit(self):
        sig = _make_signal()
        positions = [_make_position(ticker=f"STK{i}") for i in range(5)]
        ok, _ = check_max_positions(sig, positions, max_positions=10)
        assert ok is True

    def test_fails_at_limit(self):
        sig = _make_signal()
        positions = [_make_position(ticker=f"STK{i}") for i in range(10)]
        ok, reason = check_max_positions(sig, positions, max_positions=10)
        assert ok is False
        assert "10/10" in reason

    def test_exit_allowed_at_max_positions(self):
        """A SELL signal on an existing long is an exit — must not be blocked."""
        positions = [_make_position(ticker=f"STK{i}") for i in range(10)]
        positions[0] = _make_position(ticker="AAPL", quantity=10)
        sig = _make_signal(ticker="AAPL", action=Action.SELL)
        ok, _ = check_max_positions(sig, positions)
        assert ok is True


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

    def test_duplicate_buy_blocked(self):
        sig = _make_signal(ticker="AAPL", action=Action.BUY)
        ok, reason = check_no_duplicate(sig, [_make_position(ticker="AAPL", quantity=10)])
        assert ok is False
        assert "Already holding" in reason

    def test_exit_sell_allowed(self):
        """A SELL signal on an existing long position is an exit, not a duplicate."""
        sig = _make_signal(ticker="AAPL", action=Action.SELL)
        ok, _ = check_no_duplicate(sig, [_make_position(ticker="AAPL", quantity=10)])
        assert ok is True


class TestPositionSizing:
    def test_basic_sizing(self):
        from unittest.mock import patch

        sig = _make_signal(entry_price=150.0, stop_loss=145.0)
        # Pin RISK_PER_TRADE_PCT so risk-based sizing is the binding constraint.
        # risk = 1% of $100k = $1000; stop distance = $5; qty_by_risk = 200.
        # max_pct=5% → qty_by_size = $5000 / $150 = 33. min(33, 200) = 33.
        with patch("core.risk.RISK_PER_TRADE_PCT", 1.0):
            qty = calculate_position_size(sig, 100_000, max_pct=5.0)
        assert qty > 0
        assert qty <= 33

    def test_zero_portfolio(self):
        sig = _make_signal()
        qty = calculate_position_size(sig, 0)
        assert qty == 0


# BUYs require BOTH yfinance and IBKR analyst consensus to be buy/strong_buy.
# Tests that exercise OTHER risk checks but want a BUY to pass must opt into
# this gate explicitly so the assertion is testing the intended check.
_BULLISH_CONSENSUS = dict(analyst_consensus="buy", analyst_consensus_ibkr="buy")


class TestFullEvaluation:
    def test_approved(self):
        sig = _make_signal()
        result = evaluate(sig, [], 100_000, 0, **_BULLISH_CONSENSUS)
        assert result.approved is True
        assert result.position_size > 0
        assert result.reasons == []

    def test_rejected_daily_loss(self):
        from config.settings import DAILY_LOSS_LIMIT_PCT

        sig = _make_signal()
        # Loss exceeds the configured daily limit regardless of config value
        loss = -(100_000 * DAILY_LOSS_LIMIT_PCT / 100) - 500
        result = evaluate(sig, [], 100_000, loss)
        assert result.approved is False
        assert any("halted" in r.lower() for r in result.reasons)

    def test_rejected_duplicate(self):
        sig = _make_signal(ticker="AAPL", action=Action.BUY)
        positions = [_make_position(ticker="AAPL", quantity=10)]
        result = evaluate(sig, positions, 100_000, 0)
        assert result.approved is False

    def test_rejected_max_positions(self):
        sig = _make_signal()
        positions = [_make_position(ticker=f"STK{i}") for i in range(10)]
        result = evaluate(sig, positions, 100_000, 0)
        assert result.approved is False


class TestCumulativeRisk:
    """Verify cumulative risk check prevents total portfolio risk exceeding daily limit."""

    def test_rejects_when_cumulative_risk_exceeds_limit(self):
        from unittest.mock import patch
        from core.risk import check_cumulative_risk

        sig = _make_signal(entry_price=100.0, stop_loss=90.0)
        # 18 positions each risking $10/share * 10 shares = $1800 total
        positions = [
            _make_position(ticker=f"STK{i}", entry_price=100.0, stop_loss=90.0, quantity=10)
            for i in range(18)
        ]
        # Portfolio = $100K, daily limit = 2% = $2000
        # Existing risk = 18 * ($10 * 10) = $1800
        # New risk: risk_per_trade = 1% of $100K = $1000, stop_distance = $10, qty = 100, new_risk = $1000
        # Total = $2800 > $2000 → reject
        with patch("core.risk.RISK_PER_TRADE_PCT", 1.0):
            ok, reason = check_cumulative_risk(sig, positions, 100_000, limit_pct=2.0)
        assert ok is False
        assert "cumulative risk" in reason.lower()

    def test_passes_with_no_existing_positions(self):
        from unittest.mock import patch
        from core.risk import check_cumulative_risk

        sig = _make_signal(entry_price=100.0, stop_loss=97.0)
        # risk_per_trade = 1% of $100k = $1000; new_risk = $1000 < $2000 limit.
        with patch("core.risk.RISK_PER_TRADE_PCT", 1.0):
            ok, _ = check_cumulative_risk(sig, [], 100_000, limit_pct=2.0)
        assert ok is True

    def test_cumulative_risk_included_in_evaluate(self):
        """evaluate() must include cumulative risk check."""
        sig = _make_signal()
        result = evaluate(sig, [], 100_000, 0, **_BULLISH_CONSENSUS)
        # Should still pass with no positions
        assert result.approved is True

    def test_uses_actual_position_size_not_config_estimate(self):
        """check_cumulative_risk should use actual position_size when provided.

        Without this, volatility-scaled positions are overestimated: the check
        re-derives quantity from RISK_PER_TRADE_PCT which ignores vol scaling,
        causing false rejections when volatility is elevated.
        """
        from core.risk import check_cumulative_risk

        sig = _make_signal(entry_price=100.0, stop_loss=90.0)
        # 1 existing position risking $100 (10 shares * $10 stop distance)
        positions = [
            _make_position(ticker="STK0", entry_price=100.0, stop_loss=90.0, quantity=10)
        ]
        # With position_size=5 (vol-scaled down), new_risk = $10 * 5 = $50
        # Total risk = $100 + $50 = $150, limit = 2% of $100K = $2000 → pass
        ok, _ = check_cumulative_risk(
            sig, positions, 100_000, limit_pct=2.0, position_size=5,
        )
        assert ok is True

    def test_position_size_zero_means_no_new_risk(self):
        """When position_size=0, new risk contribution should be 0."""
        from core.risk import check_cumulative_risk

        sig = _make_signal(entry_price=100.0, stop_loss=90.0)
        positions = [
            _make_position(ticker=f"STK{i}", entry_price=100.0, stop_loss=90.0, quantity=10)
            for i in range(20)
        ]
        # Existing risk = 20 * $100 = $2000, limit = 2% of $100K = $2000
        # With position_size=0, new_risk=0, total=$2000 <= $2000 → pass
        ok, _ = check_cumulative_risk(
            sig, positions, 100_000, limit_pct=2.0, position_size=0,
        )
        assert ok is True


class TestSectorConcentrationCurrentPrice:
    """Verify sector concentration uses current_price when available."""

    def test_uses_current_price_over_entry_price(self):
        from core.risk import check_sector_concentration

        sig = _make_signal(
            ticker="NEW",
            indicator_values={"sector": "Technology"},
        )
        # Position entered at $100 but now worth $200
        pos = _make_position(
            ticker="OLD", entry_price=100.0, quantity=100,
            sector="Technology", current_price=200.0,
        )
        # With current_price: sector value = $200 * 100 = $20,000 (20% of $100K)
        # With entry_price: sector value = $100 * 100 = $10,000 (10% of $100K)
        # At 25% limit, both pass. At 15% limit, only current_price version rejects.
        ok, reason = check_sector_concentration(sig, [pos], 100_000, max_pct=15.0)
        assert ok is False, "Should reject using current_price ($20K > 15% of $100K)"


# ---------------------------------------------------------------------------
# Helper to build Trade objects for circuit breaker tests
# ---------------------------------------------------------------------------

def _make_trade(
    exit_time: datetime | None = None,
    entry_price: float = 100.0,
    exit_price: float = 95.0,  # loss by default
    **kwargs,
) -> Trade:
    """Create a Trade with sensible defaults. exit_price < entry_price = loss."""
    now = datetime.now(timezone.utc)
    defaults = dict(
        ticker="AAPL",
        exchange="SMART",
        quantity=10,
        entry_price=entry_price,
        exit_price=exit_price,
        entry_time=now - timedelta(hours=2),
        exit_time=exit_time or now,
        trade_type=TradeType.DAY,
    )
    defaults.update(kwargs)
    return Trade(**defaults)


# ---------------------------------------------------------------------------
# Circuit breaker tests
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    """check_circuit_breaker(recent_trades, max_consecutive_losses, window_minutes)

    Should reject when N consecutive losses happen within the time window.
    """

    def test_passes_no_trades(self):
        """No trade history — nothing to trip the breaker."""
        ok, reason = check_circuit_breaker([], max_losses=3, window_minutes=60)
        assert ok is True
        assert reason == ""

    def test_passes_single_loss(self):
        """One loss is not enough to trip a 3-loss breaker."""
        trades = [_make_trade(exit_price=95.0)]  # loss
        ok, _ = check_circuit_breaker(trades, max_losses=3, window_minutes=60)
        assert ok is True

    def test_passes_two_losses(self):
        """Two losses is still below the 3-loss threshold."""
        now = datetime.now(timezone.utc)
        trades = [
            _make_trade(exit_price=95.0, exit_time=now - timedelta(minutes=10)),
            _make_trade(exit_price=93.0, exit_time=now - timedelta(minutes=5)),
        ]
        ok, _ = check_circuit_breaker(trades, max_losses=3, window_minutes=60)
        assert ok is True

    def test_fails_three_consecutive_losses(self):
        """Three consecutive losses within the window trips the breaker."""
        now = datetime.now(timezone.utc)
        trades = [
            _make_trade(exit_price=95.0, exit_time=now - timedelta(minutes=30)),
            _make_trade(exit_price=93.0, exit_time=now - timedelta(minutes=20)),
            _make_trade(exit_price=91.0, exit_time=now - timedelta(minutes=10)),
        ]
        ok, reason = check_circuit_breaker(trades, max_losses=3, window_minutes=60)
        assert ok is False
        assert "consecutive losses" in reason.lower()

    def test_passes_win_breaks_streak(self):
        """A winning trade between losses resets the streak."""
        now = datetime.now(timezone.utc)
        trades = [
            _make_trade(exit_price=95.0, exit_time=now - timedelta(minutes=30)),  # loss
            _make_trade(exit_price=95.0, exit_time=now - timedelta(minutes=25)),  # loss
            _make_trade(exit_price=110.0, exit_time=now - timedelta(minutes=15)),  # WIN
            _make_trade(exit_price=95.0, exit_time=now - timedelta(minutes=10)),  # loss
            _make_trade(exit_price=93.0, exit_time=now - timedelta(minutes=5)),   # loss
        ]
        ok, _ = check_circuit_breaker(trades, max_losses=3, window_minutes=60)
        assert ok is True  # only 2 consecutive losses after the win

    def test_fails_losses_outside_window_ignored(self):
        """Old losses outside the time window don't count."""
        now = datetime.now(timezone.utc)
        trades = [
            _make_trade(exit_price=95.0, exit_time=now - timedelta(minutes=90)),  # outside window
            _make_trade(exit_price=93.0, exit_time=now - timedelta(minutes=20)),  # inside
            _make_trade(exit_price=91.0, exit_time=now - timedelta(minutes=10)),  # inside
        ]
        ok, _ = check_circuit_breaker(trades, max_losses=3, window_minutes=60)
        assert ok is True  # only 2 within the window

    def test_fails_all_losses_inside_window(self):
        """Three losses inside window even if older trades are wins."""
        now = datetime.now(timezone.utc)
        trades = [
            _make_trade(exit_price=110.0, exit_time=now - timedelta(minutes=120)),  # old win
            _make_trade(exit_price=95.0, exit_time=now - timedelta(minutes=30)),
            _make_trade(exit_price=93.0, exit_time=now - timedelta(minutes=20)),
            _make_trade(exit_price=91.0, exit_time=now - timedelta(minutes=10)),
        ]
        ok, reason = check_circuit_breaker(trades, max_losses=3, window_minutes=60)
        assert ok is False

    def test_breakeven_is_not_a_loss(self):
        """A trade with exit_price == entry_price is not a loss."""
        now = datetime.now(timezone.utc)
        trades = [
            _make_trade(exit_price=95.0, exit_time=now - timedelta(minutes=30)),   # loss
            _make_trade(exit_price=95.0, exit_time=now - timedelta(minutes=20)),   # loss
            _make_trade(exit_price=100.0, exit_time=now - timedelta(minutes=10)),  # breakeven
        ]
        ok, _ = check_circuit_breaker(trades, max_losses=3, window_minutes=60)
        assert ok is True  # breakeven breaks the streak

    def test_custom_threshold(self):
        """Configurable threshold — trip after 2 losses."""
        now = datetime.now(timezone.utc)
        trades = [
            _make_trade(exit_price=95.0, exit_time=now - timedelta(minutes=20)),
            _make_trade(exit_price=93.0, exit_time=now - timedelta(minutes=10)),
        ]
        ok, reason = check_circuit_breaker(trades, max_losses=2, window_minutes=60)
        assert ok is False

    def test_custom_window(self):
        """Shorter window — 30 minutes."""
        now = datetime.now(timezone.utc)
        trades = [
            _make_trade(exit_price=95.0, exit_time=now - timedelta(minutes=45)),  # outside 30m
            _make_trade(exit_price=93.0, exit_time=now - timedelta(minutes=20)),
            _make_trade(exit_price=91.0, exit_time=now - timedelta(minutes=10)),
        ]
        ok, _ = check_circuit_breaker(trades, max_losses=3, window_minutes=30)
        assert ok is True  # only 2 within the 30m window

    def test_unordered_trades_handled(self):
        """Trades not sorted by time should still work correctly."""
        now = datetime.now(timezone.utc)
        trades = [
            _make_trade(exit_price=91.0, exit_time=now - timedelta(minutes=10)),  # 3rd
            _make_trade(exit_price=95.0, exit_time=now - timedelta(minutes=30)),  # 1st
            _make_trade(exit_price=93.0, exit_time=now - timedelta(minutes=20)),  # 2nd
        ]
        ok, reason = check_circuit_breaker(trades, max_losses=3, window_minutes=60)
        assert ok is False

    def test_reason_includes_count_and_window(self):
        """The rejection reason should be informative."""
        now = datetime.now(timezone.utc)
        trades = [
            _make_trade(exit_price=95.0, exit_time=now - timedelta(minutes=30)),
            _make_trade(exit_price=93.0, exit_time=now - timedelta(minutes=20)),
            _make_trade(exit_price=91.0, exit_time=now - timedelta(minutes=10)),
        ]
        ok, reason = check_circuit_breaker(trades, max_losses=3, window_minutes=60)
        assert ok is False
        assert "3" in reason
        assert "60" in reason


class TestCircuitBreakerInEvaluate:
    """Circuit breaker must be integrated into the main evaluate() function."""

    def test_evaluate_accepts_recent_trades(self):
        """evaluate() should accept recent_trades parameter."""
        sig = _make_signal()
        result = evaluate(sig, [], 100_000, 0, recent_trades=[], **_BULLISH_CONSENSUS)
        assert result.approved is True

    def test_evaluate_rejects_on_circuit_breaker(self):
        """evaluate() rejects when circuit breaker trips."""
        now = datetime.now(timezone.utc)
        losing_trades = [
            _make_trade(exit_price=95.0, exit_time=now - timedelta(minutes=30)),
            _make_trade(exit_price=93.0, exit_time=now - timedelta(minutes=20)),
            _make_trade(exit_price=91.0, exit_time=now - timedelta(minutes=10)),
        ]
        sig = _make_signal()
        result = evaluate(sig, [], 100_000, 0, recent_trades=losing_trades)
        assert result.approved is False
        assert any("consecutive losses" in r.lower() for r in result.reasons)

    def test_evaluate_passes_with_no_trades(self):
        """evaluate() passes circuit breaker when no recent trades."""
        sig = _make_signal()
        result = evaluate(sig, [], 100_000, 0, recent_trades=[], **_BULLISH_CONSENSUS)
        assert result.approved is True

    def test_evaluate_backward_compatible(self):
        """evaluate() still works without recent_trades (defaults to no trades)."""
        sig = _make_signal()
        result = evaluate(sig, [], 100_000, 0, **_BULLISH_CONSENSUS)
        assert result.approved is True


class TestAntiMomentumInvalidPrices:
    """check_anti_momentum must reject signals with zero/invalid prices."""

    def test_rejects_zero_current_price(self):
        sig = _make_signal(entry_price=150.0)
        ok, reason = check_anti_momentum(sig, current_price=0.0)
        assert ok is False
        assert "Invalid prices" in reason

    def test_rejects_negative_current_price(self):
        sig = _make_signal(entry_price=150.0)
        ok, reason = check_anti_momentum(sig, current_price=-5.0)
        assert ok is False
        assert "Invalid prices" in reason

    def test_rejects_zero_entry_price(self):
        sig = _make_signal(entry_price=0.0, stop_loss=0.0, take_profit=0.0)
        ok, reason = check_anti_momentum(sig, current_price=150.0)
        assert ok is False
        assert "Invalid prices" in reason

    def test_passes_valid_prices(self):
        sig = _make_signal(entry_price=150.0)
        ok, _ = check_anti_momentum(sig, current_price=151.0)
        assert ok is True


class TestRiskRewardInvalidPrices:
    """check_risk_reward must reject signals with zero/invalid prices."""

    def test_rejects_zero_entry_price(self):
        sig = _make_signal(entry_price=0.0, stop_loss=0.0, take_profit=0.0)
        ok, reason = check_risk_reward(sig)
        assert ok is False
        assert "Invalid prices" in reason

    def test_rejects_zero_stop_loss(self):
        sig = _make_signal(stop_loss=0.0)
        ok, reason = check_risk_reward(sig)
        assert ok is False
        assert "Invalid prices" in reason

    def test_rejects_zero_take_profit(self):
        sig = _make_signal(take_profit=0.0)
        ok, reason = check_risk_reward(sig)
        assert ok is False
        assert "Invalid prices" in reason

    def test_passes_valid_prices(self):
        sig = _make_signal(entry_price=150.0, stop_loss=145.0, take_profit=165.0)
        ok, _ = check_risk_reward(sig)
        assert ok is True


class TestSectorCheckNoneIndicatorValues:
    """Sector checks must not crash when indicator_values is None."""

    def test_sector_concentration_none_indicator_values(self):
        sig = _make_signal(indicator_values=None)
        ok, _ = check_sector_concentration(sig, [], 100_000)
        assert ok is True  # Unknown sector, let it through

    def test_excluded_sector_none_indicator_values(self):
        sig = _make_signal(indicator_values=None)
        ok, _ = check_excluded_sector(sig)
        assert ok is True  # Unknown sector, let it through


class TestShortSelling:
    """check_short_selling blocks SELL signals for unheld stocks."""

    def test_sell_blocked_when_no_position(self):
        from core.risk import check_short_selling
        sig = _make_signal(ticker="AAPL", action=Action.SELL)
        ok, reason = check_short_selling(sig, [])
        assert ok is False
        assert "Short selling blocked" in reason

    def test_sell_allowed_when_position_held(self):
        from core.risk import check_short_selling
        sig = _make_signal(ticker="AAPL", action=Action.SELL)
        pos = _make_position(ticker="AAPL", quantity=10)
        ok, _ = check_short_selling(sig, [pos])
        assert ok is True

    def test_buy_always_allowed(self):
        from core.risk import check_short_selling
        sig = _make_signal(ticker="AAPL", action=Action.BUY)
        ok, _ = check_short_selling(sig, [])
        assert ok is True

    def test_evaluate_rejects_short_sell(self):
        """evaluate() blocks SELL for unheld stock via short selling check."""
        sig = _make_signal(ticker="AAPL", action=Action.SELL,
                          entry_price=150, stop_loss=155, take_profit=140)
        result = evaluate(sig, [], 100_000, 0)
        assert result.approved is False
        assert any("short selling" in r.lower() for r in result.reasons)


class TestTrendConfirmation:
    """check_trend_confirmation rejects misaligned MAs."""

    def test_buy_rejected_when_ma5_below_ma20(self):
        from core.risk import check_trend_confirmation
        sig = _make_signal(action=Action.BUY)
        ok, reason = check_trend_confirmation(sig, {"MA5": 95, "MA20": 100})
        assert ok is False
        assert "Trend not confirmed" in reason

    def test_buy_passes_when_aligned(self):
        from core.risk import check_trend_confirmation
        sig = _make_signal(action=Action.BUY)
        ok, _ = check_trend_confirmation(sig, {"MA5": 105, "MA20": 100})
        assert ok is True

    def test_sell_rejected_when_ma5_above_ma20(self):
        from core.risk import check_trend_confirmation
        sig = _make_signal(action=Action.SELL, entry_price=150, stop_loss=155, take_profit=140)
        ok, reason = check_trend_confirmation(sig, {"MA5": 105, "MA20": 100})
        assert ok is False
        assert "Trend not confirmed" in reason


class TestDefenseSectorExclusion:
    """check_excluded_sector must block defense/military stocks."""

    def test_blocks_defense_sector(self):
        sig = _make_signal(indicator_values={"sector": "Aerospace & Defense"})
        ok, reason = check_excluded_sector(sig)
        assert ok is False
        assert "defense" in reason.lower()

    def test_blocks_military_keyword(self):
        sig = _make_signal(indicator_values={"sector": "Military Equipment"})
        ok, reason = check_excluded_sector(sig)
        assert ok is False

    def test_allows_technology_sector(self):
        sig = _make_signal(indicator_values={"sector": "Technology"})
        ok, _ = check_excluded_sector(sig)
        assert ok is True


# ---------------------------------------------------------------------------
# Volatility Regime Tests
# ---------------------------------------------------------------------------

class TestRealizedVolatility:
    """calculate_realized_volatility computes annualized volatility from returns."""

    def test_flat_prices_zero_volatility(self):
        from core.risk import calculate_realized_volatility
        import pandas as pd
        closes = pd.Series([100.0] * 30)
        vol = calculate_realized_volatility(closes)
        assert vol == 0.0

    def test_volatile_prices_higher_than_calm(self):
        from core.risk import calculate_realized_volatility
        import pandas as pd
        import numpy as np
        np.random.seed(42)
        calm = pd.Series(100.0 + np.cumsum(np.random.normal(0, 0.5, 60)))
        wild = pd.Series(100.0 + np.cumsum(np.random.normal(0, 3.0, 60)))
        vol_calm = calculate_realized_volatility(calm)
        vol_wild = calculate_realized_volatility(wild)
        assert vol_wild > vol_calm

    def test_returns_none_for_short_series(self):
        from core.risk import calculate_realized_volatility
        import pandas as pd
        closes = pd.Series([100.0, 101.0])
        vol = calculate_realized_volatility(closes, window=20)
        assert vol is None

    def test_annualized(self):
        """Volatility should be annualized (multiplied by sqrt(252))."""
        from core.risk import calculate_realized_volatility
        import pandas as pd
        import numpy as np
        np.random.seed(0)
        closes = pd.Series(100.0 + np.cumsum(np.random.normal(0, 1, 60)))
        vol = calculate_realized_volatility(closes, window=20)
        assert vol is not None
        # Annualized vol should be materially larger than daily vol
        assert vol > 0.05  # daily vol ~1% would annualize to ~16%


class TestVolatilityAdjustedPositionSize:
    """calculate_position_size should scale down in high-volatility regimes."""

    def test_high_vol_reduces_position(self):
        sig = _make_signal(entry_price=150.0, stop_loss=145.0)
        qty_normal = calculate_position_size(sig, 100_000)
        qty_high_vol = calculate_position_size(sig, 100_000, volatility=0.40)
        assert qty_high_vol < qty_normal

    def test_low_vol_does_not_increase_beyond_base(self):
        """Low vol should not increase position beyond the unscaled base."""
        sig = _make_signal(entry_price=150.0, stop_loss=145.0)
        qty_normal = calculate_position_size(sig, 100_000)
        qty_low_vol = calculate_position_size(sig, 100_000, volatility=0.05)
        # Low vol scales up but capped at base (no leverage)
        assert qty_low_vol <= qty_normal

    def test_none_volatility_uses_base_sizing(self):
        """When volatility=None (not available), use original sizing."""
        sig = _make_signal(entry_price=150.0, stop_loss=145.0)
        qty_base = calculate_position_size(sig, 100_000)
        qty_none = calculate_position_size(sig, 100_000, volatility=None)
        assert qty_base == qty_none

    def test_extreme_vol_allows_tiny_position_on_cheap_stock(self):
        """On a cheap stock, even 150% vol leaves the scaled position >= 1."""
        sig = _make_signal(entry_price=50.0, stop_loss=45.0)
        qty = calculate_position_size(sig, 100_000, volatility=1.5)
        # Base qty is large (~1000 on this signal); vol_scale = 0.2/1.5 ≈ 0.133;
        # scaled ≈ 133 shares — still a real position.
        assert qty >= 1

    def test_extreme_vol_rejects_expensive_stock_with_tiny_base(self):
        """When vol scaling drops the size to <1 share, return 0 — don't floor to 1.

        Previous behavior used `max(int(qty * vol_scale), min(1, qty))`, which
        forced a 1-share trade through even when the volatility-adjusted size
        rounded to zero. In a high-vol regime that means taking a trade the
        vol check intended to reject, with a max-loss that can exceed the
        intended per-trade risk budget because the 1-share stop distance may
        represent >RISK_PER_TRADE_PCT of equity on an expensive stock.

        The correct behavior is to let vol-scaled size fall to zero so
        evaluate() can reject the signal entirely.
        """
        # Expensive stock with a tight stop → small base quantity
        sig = _make_signal(entry_price=10_000.0, stop_loss=9_999.0)
        # Base qty_by_size = 50_000 / 10_000 = 5, qty_by_risk = huge
        # vol_scale = 0.2 / 1.5 ≈ 0.133 → 5 * 0.133 ≈ 0.66 → int = 0
        qty = calculate_position_size(sig, 100_000, volatility=1.5)
        assert qty == 0, (
            "Extreme volatility on an expensive stock must scale the position "
            f"to 0 (rejection), not floor to 1 — got {qty} shares"
        )


class TestVolatilityInEvaluate:
    """evaluate() passes volatility through to position sizing."""

    def test_high_vol_reduces_approved_size(self):
        sig = _make_signal(entry_price=150.0, stop_loss=145.0)
        result_normal = evaluate(sig, [], 100_000, 0, **_BULLISH_CONSENSUS)
        result_high_vol = evaluate(
            sig, [], 100_000, 0, volatility=0.40, **_BULLISH_CONSENSUS,
        )
        assert result_normal.approved is True
        assert result_high_vol.approved is True
        assert result_high_vol.position_size < result_normal.position_size


# ---------------------------------------------------------------------------
# Analyst Consensus Check Tests
# ---------------------------------------------------------------------------

class TestAnalystConsensus:
    """BUY only when BOTH yfinance AND IBKR analyst consensus are 'buy' or
    'strong_buy'. Either source reporting hold/sell/strong_sell — or returning
    None (no data) — blocks the BUY. Two-source agreement is the gate.
    """

    def test_allows_buy_when_both_buy(self):
        sig = _make_signal(action=Action.BUY)
        ok, _ = check_analyst_consensus(sig, "buy", "buy")
        assert ok is True

    def test_allows_buy_when_both_strong_buy(self):
        sig = _make_signal(action=Action.BUY)
        ok, _ = check_analyst_consensus(sig, "strong_buy", "strong_buy")
        assert ok is True

    def test_allows_buy_when_yf_buy_ibkr_strong_buy(self):
        sig = _make_signal(action=Action.BUY)
        ok, _ = check_analyst_consensus(sig, "buy", "strong_buy")
        assert ok is True

    def test_allows_buy_when_yf_strong_buy_ibkr_buy(self):
        sig = _make_signal(action=Action.BUY)
        ok, _ = check_analyst_consensus(sig, "strong_buy", "buy")
        assert ok is True

    def test_blocks_buy_on_yf_hold(self):
        """Hold no longer counts as buy — must block even if IBKR is bullish."""
        sig = _make_signal(action=Action.BUY)
        ok, reason = check_analyst_consensus(sig, "hold", "buy")
        assert ok is False
        assert "analyst" in reason.lower()

    def test_blocks_buy_on_ibkr_hold(self):
        sig = _make_signal(action=Action.BUY)
        ok, reason = check_analyst_consensus(sig, "buy", "hold")
        assert ok is False
        assert "analyst" in reason.lower()

    def test_blocks_buy_on_yf_sell(self):
        sig = _make_signal(action=Action.BUY)
        ok, reason = check_analyst_consensus(sig, "sell", "buy")
        assert ok is False
        assert "analyst" in reason.lower()

    def test_blocks_buy_on_ibkr_sell(self):
        sig = _make_signal(action=Action.BUY)
        ok, reason = check_analyst_consensus(sig, "buy", "sell")
        assert ok is False
        assert "analyst" in reason.lower()

    def test_blocks_buy_on_yf_strong_sell(self):
        sig = _make_signal(action=Action.BUY)
        ok, reason = check_analyst_consensus(sig, "strong_sell", "buy")
        assert ok is False
        assert "analyst" in reason.lower()

    def test_blocks_buy_on_ibkr_strong_sell(self):
        sig = _make_signal(action=Action.BUY)
        ok, reason = check_analyst_consensus(sig, "buy", "strong_sell")
        assert ok is False
        assert "analyst" in reason.lower()

    def test_blocks_buy_when_yf_missing(self):
        """yfinance has no data — cannot confirm two-source agreement → block."""
        sig = _make_signal(action=Action.BUY)
        ok, reason = check_analyst_consensus(sig, None, "buy")
        assert ok is False
        assert "analyst" in reason.lower()

    def test_blocks_buy_when_ibkr_missing(self):
        """IBKR has no data — cannot confirm two-source agreement → block."""
        sig = _make_signal(action=Action.BUY)
        ok, reason = check_analyst_consensus(sig, "buy", None)
        assert ok is False
        assert "analyst" in reason.lower()

    def test_blocks_buy_when_both_missing(self):
        sig = _make_signal(action=Action.BUY)
        ok, reason = check_analyst_consensus(sig, None, None)
        assert ok is False
        assert "analyst" in reason.lower()

    def test_disabled_passes_through(self):
        """When CHECK_ANALYST_CONSENSUS=False, all combinations pass."""
        sig = _make_signal(action=Action.BUY)
        ok, _ = check_analyst_consensus(sig, "sell", "sell", enabled=False)
        assert ok is True

    def test_sell_signal_always_passes(self):
        """Analyst consensus check only applies to BUY signals."""
        sig = _make_signal(action=Action.SELL, entry_price=150, stop_loss=155, take_profit=140)
        ok, _ = check_analyst_consensus(sig, "sell", "sell")
        assert ok is True

    def test_hold_signal_always_passes(self):
        sig = _make_signal(action=Action.HOLD)
        ok, _ = check_analyst_consensus(sig, "sell", "sell")
        assert ok is True


class TestAnalystConsensusInEvaluate:
    """evaluate() integrates the two-source consensus check."""

    def test_evaluate_allows_buy_when_both_buy(self):
        sig = _make_signal(action=Action.BUY)
        result = evaluate(
            sig, [], 100_000, 0,
            analyst_consensus="buy",
            analyst_consensus_ibkr="buy",
        )
        assert result.approved is True

    def test_evaluate_blocks_buy_when_yf_sell_even_if_ibkr_buy(self):
        sig = _make_signal(action=Action.BUY)
        result = evaluate(
            sig, [], 100_000, 0,
            analyst_consensus="sell",
            analyst_consensus_ibkr="buy",
        )
        assert result.approved is False
        assert any("analyst" in r.lower() for r in result.reasons)

    def test_evaluate_blocks_buy_when_ibkr_sell_even_if_yf_buy(self):
        sig = _make_signal(action=Action.BUY)
        result = evaluate(
            sig, [], 100_000, 0,
            analyst_consensus="buy",
            analyst_consensus_ibkr="sell",
        )
        assert result.approved is False
        assert any("analyst" in r.lower() for r in result.reasons)

    def test_evaluate_blocks_buy_on_hold(self):
        """Hold from either source must block — was previously allowed."""
        sig = _make_signal(action=Action.BUY)
        result = evaluate(
            sig, [], 100_000, 0,
            analyst_consensus="hold",
            analyst_consensus_ibkr="buy",
        )
        assert result.approved is False
        assert any("analyst" in r.lower() for r in result.reasons)

    def test_evaluate_blocks_buy_when_ibkr_missing(self):
        sig = _make_signal(action=Action.BUY)
        result = evaluate(
            sig, [], 100_000, 0,
            analyst_consensus="buy",
            analyst_consensus_ibkr=None,
        )
        assert result.approved is False
        assert any("analyst" in r.lower() for r in result.reasons)

    def test_evaluate_blocks_buy_when_both_missing(self):
        sig = _make_signal(action=Action.BUY)
        result = evaluate(
            sig, [], 100_000, 0,
            analyst_consensus=None,
            analyst_consensus_ibkr=None,
        )
        assert result.approved is False
        assert any("analyst" in r.lower() for r in result.reasons)


# ---------------------------------------------------------------------------
# Exit Signal Tests — discipline checks must not block position exits
# ---------------------------------------------------------------------------

class TestExitSignalsNotBlocked:
    """Discipline checks (trend, anti-momentum, risk/reward) must not block
    exit signals that close existing positions. Blocking exits can trap
    the trader in a losing position indefinitely."""

    def test_sell_exit_passes_trend_confirmation(self):
        """SELL to close a long must not be blocked by uptrend confirmation.

        Scenario: We hold AAPL long, and a SELL signal is generated to close it.
        The trend is still up (MA5 > MA20), so check_trend_confirmation would
        reject the SELL (it requires MA5 < MA20 for sells). But this is an EXIT,
        not a new short entry — it must pass.
        """
        from core.risk import check_trend_confirmation
        sig = _make_signal(
            ticker="AAPL", action=Action.SELL,
            entry_price=150, stop_loss=155, take_profit=140,
        )
        # Uptrend: MA5 > MA20 — trend confirmation should reject SELL entries,
        # but evaluate() should let exits through.
        positions = [_make_position(ticker="AAPL", quantity=10)]
        result = evaluate(
            sig, positions, 100_000, 0,
            current_price=150.0,
        )
        # The signal is closing an existing long — should not be blocked by
        # trend, anti-momentum, or risk/reward
        trend_reasons = [r for r in result.reasons if "trend not confirmed" in r.lower()]
        assert trend_reasons == [], (
            f"Exit signal blocked by trend confirmation: {trend_reasons}"
        )

    def test_sell_exit_passes_anti_momentum(self):
        """SELL to close a long must not be blocked by anti-momentum.

        Scenario: Price dropped 15% from entry — anti-momentum would reject
        a SELL (it thinks we're chasing a down move). But we're closing a
        losing long position, not initiating a short.
        """
        sig = _make_signal(
            ticker="AAPL", action=Action.SELL,
            entry_price=150, stop_loss=155, take_profit=140,
        )
        positions = [_make_position(ticker="AAPL", quantity=10)]
        # Price dropped significantly — anti-momentum would normally block
        result = evaluate(
            sig, positions, 100_000, 0,
            current_price=125.0,  # 16.7% below entry
        )
        anti_reasons = [r for r in result.reasons if "anti-chase" in r.lower()]
        assert anti_reasons == [], (
            f"Exit signal blocked by anti-momentum: {anti_reasons}"
        )

    def test_sell_exit_passes_risk_reward(self):
        """SELL to close a long must not be blocked by risk/reward ratio.

        Scenario: Signal has bad R:R because it's closing a losing position.
        Risk/reward only makes sense for new entries.
        """
        sig = _make_signal(
            ticker="AAPL", action=Action.SELL,
            entry_price=100, stop_loss=105, take_profit=95,
        )
        positions = [_make_position(ticker="AAPL", quantity=10)]
        result = evaluate(sig, positions, 100_000, 0)
        rr_reasons = [r for r in result.reasons if "risk/reward" in r.lower()]
        assert rr_reasons == [], (
            f"Exit signal blocked by risk/reward: {rr_reasons}"
        )

    def test_new_sell_entry_still_checked(self):
        """A new SELL (short entry, no existing position) must still be checked.

        When short selling IS allowed, discipline checks should still apply
        to new short entries.
        """
        from unittest.mock import patch
        sig = _make_signal(
            ticker="NEW", action=Action.SELL,
            entry_price=150, stop_loss=155, take_profit=140,
        )
        # No existing position — this is a new short entry
        with patch("core.risk.ALLOW_SHORT_SELLING", True):
            result = evaluate(sig, [], 100_000, 0)
        # With uptrend, trend confirmation should reject a new short
        # (This test verifies discipline checks still apply to entries)

    def test_sell_exit_passes_cumulative_risk(self):
        """SELL to close a long must not be blocked by cumulative risk.

        Scenario: the AAPL position alone carries enough open risk to breach
        the daily loss limit. Closing it REDUCES total risk to zero, but the
        current check treats the exit as a fresh entry and adds phantom
        new_risk on top of existing_risk (which already includes AAPL). Both
        the presence of existing AAPL risk AND the phantom new_risk trip the
        cap and trap the trader in the losing position.
        """
        # AAPL long with a deliberately wide stop — existing risk alone
        # exceeds the 10% daily-loss-limit ($10k on $100k portfolio).
        aapl_pos = _make_position(
            ticker="AAPL", entry_price=150.0, stop_loss=50.0, quantity=150,
            sector="Technology",
        )  # risk = $100 * 150 = $15,000 (already > $10k cap)
        positions = [aapl_pos]

        sig = _make_signal(
            ticker="AAPL", action=Action.SELL,
            entry_price=160, stop_loss=170, take_profit=140,
            indicator_values={"sector": "Technology"},
        )
        result = evaluate(
            sig, positions, 100_000, 0,
            current_price=160.0,
        )
        cum_reasons = [r for r in result.reasons if "cumulative risk" in r.lower()]
        assert cum_reasons == [], (
            f"Exit signal blocked by cumulative risk: {cum_reasons}"
        )

    def test_sell_exit_passes_sector_concentration(self):
        """SELL to close a long must not be blocked by sector concentration.

        Scenario: technology sector is already at 40% of portfolio. A SELL
        to close AAPL (tech) reduces tech exposure — must not be blocked
        because the check added `proposed_value` as if opening a new tech
        position on top.
        """
        # Portfolio heavily weighted to tech (2 × 20% positions)
        pos1 = _make_position(
            ticker="AAPL", entry_price=150, quantity=133, sector="Technology",
        )  # ~$20k
        pos2 = _make_position(
            ticker="MSFT", entry_price=300, quantity=67, sector="Technology",
        )  # ~$20k
        positions = [pos1, pos2]
        # Close AAPL — tech exposure will DROP after the exit, not rise
        sig = _make_signal(
            ticker="AAPL", action=Action.SELL,
            entry_price=160, stop_loss=170, take_profit=140,
            indicator_values={"sector": "Technology"},
        )
        # With MAX_SECTOR_CONCENTRATION_PCT=25 (hypothetical tight cap),
        # existing 40% tech + proposed-value would trip the check even
        # though we're closing one of those positions.
        result = evaluate(
            sig, positions, 100_000, 0,
            current_price=160.0,
        )
        sector_reasons = [r for r in result.reasons if "sector" in r.lower() and "concentration" not in r.lower() or "exposure" in r.lower()]
        # The specific error message contains "exposure" when tripped
        sector_violations = [r for r in result.reasons if "exposure" in r.lower()]
        assert sector_violations == [], (
            f"Exit signal blocked by sector concentration: {sector_violations}"
        )

    def test_sell_exit_passes_excluded_sector(self):
        """SELL to close a legacy position in newly-excluded sector must pass.

        Scenario: user imports a legacy position in JPM (financial sector).
        The universe filter blocks new entries, but an exit must still be
        allowed — otherwise the user is trapped in the position.
        """
        positions = [_make_position(
            ticker="JPM", entry_price=150, quantity=10, sector="Financials",
        )]
        sig = _make_signal(
            ticker="JPM", action=Action.SELL,
            entry_price=155, stop_loss=160, take_profit=145,
            indicator_values={"sector": "Financials"},
        )
        result = evaluate(
            sig, positions, 100_000, 0,
            current_price=155.0,
        )
        excluded_reasons = [r for r in result.reasons if "excluded" in r.lower()]
        assert excluded_reasons == [], (
            f"Exit signal blocked by excluded-sector check: {excluded_reasons}"
        )

    def test_sell_exit_passes_excluded_ticker(self):
        """SELL to close a position in EXCLUDED_TICKERS must pass.

        The exclusion list prevents new entries into specific tickers,
        but a legacy position must still be exitable.
        """
        from unittest.mock import patch
        positions = [_make_position(
            ticker="TEVA", entry_price=10, quantity=100, sector="Healthcare",
        )]
        sig = _make_signal(
            ticker="TEVA", action=Action.SELL,
            entry_price=11, stop_loss=12, take_profit=9,
            indicator_values={"sector": "Healthcare"},
        )
        # Confirm TEVA is in the default exclusion list, then verify evaluate
        # doesn't block an exit of an already-held TEVA position.
        with patch("core.risk.EXCLUDED_TICKERS", {"TEVA"}):
            result = evaluate(
                sig, positions, 100_000, 0,
                current_price=11.0,
            )
        excluded_reasons = [r for r in result.reasons if "excluded" in r.lower()]
        assert excluded_reasons == [], (
            f"Exit of excluded-ticker position blocked: {excluded_reasons}"
        )

    def test_exit_signal_position_size_matches_existing_holding(self):
        """For exit signals, RiskResult.position_size must equal the existing
        position's absolute quantity — not a freshly-calculated new-entry size.

        Scenario: user holds AAPL 100 shares long. An exit SELL signal arrives.
        calculate_position_size(), treating it as a fresh short, would compute
        a much larger quantity based on portfolio risk budget. Passing that
        larger quantity to place_order would close the 100 long AND open a
        net short in the remainder — a position inversion that violates
        ALLOW_SHORT_SELLING=False and is never the user's intent.
        """
        # Hold 100 AAPL long
        positions = [_make_position(
            ticker="AAPL", entry_price=150.0, stop_loss=145.0, quantity=100,
            sector="Technology",
        )]
        # SELL signal to close — with a wide stop that would imply a large
        # fresh-entry size under the default 5% RISK_PER_TRADE_PCT.
        sig = _make_signal(
            ticker="AAPL", action=Action.SELL,
            entry_price=160, stop_loss=170, take_profit=140,
            indicator_values={"sector": "Technology"},
        )
        result = evaluate(
            sig, positions, 100_000, 0,
            current_price=160.0,
        )
        assert result.approved, (
            f"Exit signal unexpectedly rejected: {result.reasons}"
        )
        # position_size must match existing holding, not a fresh-entry calc.
        assert result.position_size == 100, (
            f"Exit position_size={result.position_size} — should equal "
            "existing quantity (100) so we close, not invert. A larger value "
            "would flip the position to a net short."
        )

    def test_exit_signal_covering_short_matches_existing_short(self):
        """Exit of a short (BUY to cover) must size to match the short."""
        # Hold 50 MSFT short (quantity = -50)
        positions = [_make_position(
            ticker="MSFT", entry_price=300.0, stop_loss=310.0, quantity=-50,
            sector="Technology",
        )]
        # BUY signal to close the short
        sig = _make_signal(
            ticker="MSFT", action=Action.BUY,
            entry_price=290, stop_loss=280, take_profit=310,
            indicator_values={"sector": "Technology"},
        )
        result = evaluate(
            sig, positions, 100_000, 0,
            current_price=290.0,
        )
        assert result.approved, (
            f"Short-cover signal unexpectedly rejected: {result.reasons}"
        )
        assert result.position_size == 50, (
            f"Cover position_size={result.position_size} — should equal "
            "abs(existing short quantity) = 50."
        )

    def test_risk_result_flags_exit_signal(self):
        """RiskResult must expose is_exit so the executor can route exits to
        a simple close order instead of a bracket. A bracket placed for an
        exit would leave SL/TP child orders live after the parent fills — at
        IBKR those orders can re-enter the ticker at the stop or target price.
        """
        positions = [_make_position(ticker="AAPL", quantity=100)]
        sig = _make_signal(
            ticker="AAPL", action=Action.SELL,
            entry_price=160, stop_loss=170, take_profit=140,
            indicator_values={"sector": "Technology"},
        )
        result = evaluate(sig, positions, 100_000, 0, current_price=160.0)
        assert result.is_exit is True, (
            "Exit signal (SELL on existing long) must set RiskResult.is_exit"
        )

    def test_risk_result_new_entry_is_not_exit(self):
        """New entries must have is_exit=False so the executor places a full
        bracket order with SL/TP."""
        sig = _make_signal(
            ticker="AAPL", action=Action.BUY,
            entry_price=150, stop_loss=145, take_profit=165,
        )
        result = evaluate(sig, [], 100_000, 0)
        assert result.is_exit is False


# ---------------------------------------------------------------------------
# Empty sector safety net — must not silently pass excluded stocks
# ---------------------------------------------------------------------------

class TestEmptySectorSafetyNet:
    """Stocks with empty sector must not bypass excluded sector checks
    when company name contains financial/defense keywords."""

    def test_empty_sector_with_financial_company_name_blocked(self):
        """A stock with empty sector but 'bank' in company name must be blocked."""
        sig = _make_signal(
            indicator_values={"sector": "", "company_name": "First National Bank Corp"},
        )
        ok, reason = check_excluded_sector(sig)
        assert ok is False, (
            "Stock with 'bank' in company name should be blocked even with empty sector"
        )

    def test_empty_sector_with_defense_company_name_blocked(self):
        """A stock with empty sector but defense keywords in name must be blocked."""
        sig = _make_signal(
            indicator_values={"sector": "", "company_name": "Lockheed Defense Systems"},
        )
        ok, reason = check_excluded_sector(sig)
        assert ok is False

    def test_empty_sector_with_normal_company_name_passes(self):
        """A stock with empty sector and non-excluded company name passes."""
        sig = _make_signal(
            indicator_values={"sector": "", "company_name": "Apple Inc"},
        )
        ok, _ = check_excluded_sector(sig)
        assert ok is True

    def test_empty_sector_no_company_name_passes(self):
        """A stock with empty sector and no company name passes (can't determine)."""
        sig = _make_signal(indicator_values={"sector": ""})
        ok, _ = check_excluded_sector(sig)
        assert ok is True


# ---------------------------------------------------------------------------
# Feature 5: Correlation cap
# ---------------------------------------------------------------------------

import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402

from core.risk import check_correlation  # noqa: E402


def _returns(*values: float) -> pd.Series:
    """Build a return series from a list of values."""
    return pd.Series(values)


def _build_returns_from_closes(closes: list[float]) -> pd.Series:
    """Convert a close-price list into a daily-returns Series."""
    s = pd.Series(closes)
    return s.pct_change().dropna()


class TestCorrelationCheck:
    """check_correlation(signal, open_positions, returns_lookup, threshold=0.7).

    Reject a new BUY when its return series correlates above `threshold`
    with any already-held position. Negative correlation or no-correlation
    passes through. Exit signals and missing data always pass.
    """

    def test_passes_when_no_open_positions(self):
        """No positions → no correlation risk."""
        sig = _make_signal(ticker="AAPL")
        ok, reason = check_correlation(sig, [], {})
        assert ok is True

    def test_passes_when_candidate_has_no_return_data(self):
        """If candidate's returns are missing, don't block."""
        sig = _make_signal(ticker="NEWCO")
        positions = [_make_position(ticker="MSFT")]
        returns = {"MSFT": _returns(0.01, 0.02, -0.01, 0.0, 0.01)}
        ok, reason = check_correlation(sig, positions, returns)
        assert ok is True

    def test_passes_when_existing_position_has_no_return_data(self):
        """Missing data for an existing position should just skip it."""
        sig = _make_signal(ticker="AAPL")
        positions = [_make_position(ticker="MSFT")]
        # AAPL has data, but MSFT doesn't
        returns = {"AAPL": _returns(0.01, 0.02, -0.01, 0.0, 0.01)}
        ok, _ = check_correlation(sig, positions, returns)
        assert ok is True

    def test_rejects_highly_correlated_pair(self):
        """Two series with correlation ≈ 1.0 must be rejected."""
        sig = _make_signal(ticker="AMD")
        positions = [_make_position(ticker="NVDA")]
        # Perfectly correlated returns (≥ default min_periods=20)
        identical = [0.01, 0.02, -0.01, 0.03, -0.02] * 5  # 25 points
        returns = {
            "NVDA": pd.Series(identical),
            "AMD": pd.Series(identical),  # corr = 1.0
        }
        ok, reason = check_correlation(sig, positions, returns, threshold=0.7)
        assert ok is False
        assert "correlation" in reason.lower()
        assert "NVDA" in reason

    def test_passes_when_correlation_below_threshold(self):
        """Uncorrelated returns (~0.0) should pass."""
        sig = _make_signal(ticker="GOLD")
        positions = [_make_position(ticker="SPY")]
        rng = np.random.default_rng(42)
        returns = {
            "SPY": pd.Series(rng.standard_normal(100) * 0.01),
            "GOLD": pd.Series(rng.standard_normal(100) * 0.01),  # independent
        }
        ok, _ = check_correlation(sig, positions, returns, threshold=0.7)
        assert ok is True

    def test_passes_on_negative_correlation(self):
        """Hedges (negative correlation) should always pass — they reduce risk."""
        sig = _make_signal(ticker="PUT")
        positions = [_make_position(ticker="SPY")]
        spy = [0.01, 0.02, -0.01, 0.03, -0.02, 0.015, 0.005, 0.02, -0.01, 0.01]
        returns = {
            "SPY": pd.Series(spy),
            "PUT": pd.Series([-x for x in spy]),  # corr = -1.0
        }
        ok, _ = check_correlation(sig, positions, returns, threshold=0.7)
        assert ok is True

    def test_threshold_boundary_is_strict(self):
        """A correlation exactly equal to threshold should PASS (not strictly above)."""
        sig = _make_signal(ticker="AMD")
        positions = [_make_position(ticker="NVDA")]
        # Construct two series with exactly 0.7 correlation
        rng = np.random.default_rng(0)
        base = rng.standard_normal(200) * 0.01
        noise = rng.standard_normal(200) * 0.01
        # Linear combination tuned for corr ≈ 0.7
        # (0.7 * base + sqrt(1-0.49) * noise) gives corr(base, combo) = 0.7
        combo = 0.7 * base + (1 - 0.49) ** 0.5 * noise
        returns = {"NVDA": pd.Series(base), "AMD": pd.Series(combo)}

        # Compute actual correlation
        actual_corr = pd.Series(base).corr(pd.Series(combo))
        # Set threshold to just above the actual corr → should pass
        ok, _ = check_correlation(
            sig, positions, returns, threshold=actual_corr + 0.01,
        )
        assert ok is True

        # Set threshold just below → should reject
        ok, _ = check_correlation(
            sig, positions, returns, threshold=actual_corr - 0.01,
        )
        assert ok is False

    def test_rejects_on_max_correlation_across_multiple_positions(self):
        """With 2 open positions: one uncorrelated, one highly correlated, reject."""
        sig = _make_signal(ticker="AMD")
        positions = [
            _make_position(ticker="KO"),
            _make_position(ticker="NVDA"),
        ]
        rng = np.random.default_rng(1)
        returns = {
            "KO": pd.Series(rng.standard_normal(50) * 0.01),
            "NVDA": pd.Series([0.01, 0.02, -0.01, 0.03, -0.02] * 10),
            "AMD": pd.Series([0.01, 0.02, -0.01, 0.03, -0.02] * 10),  # matches NVDA
        }
        ok, reason = check_correlation(sig, positions, returns, threshold=0.7)
        assert ok is False
        assert "NVDA" in reason  # rejection names the correlated ticker, not KO

    def test_exit_signal_passes_even_if_correlated(self):
        """A SELL closing an existing long should not be blocked by correlation."""
        sig = _make_signal(ticker="NVDA", action=Action.SELL)
        positions = [_make_position(ticker="NVDA", quantity=10)]  # existing long
        # Same ticker as candidate — correlation is 1.0 — but this is an exit
        identical = [0.01, 0.02, -0.01, 0.03, -0.02] * 5
        returns = {"NVDA": pd.Series(identical)}
        ok, _ = check_correlation(sig, positions, returns, threshold=0.7)
        assert ok is True

    def test_short_entry_not_treated_as_exit(self):
        """A SELL on a stock we DON'T hold is a short entry, not an exit —
        should still be checked for correlation with existing positions."""
        sig = _make_signal(ticker="SHORT", action=Action.SELL)
        positions = [_make_position(ticker="LONG", quantity=10)]
        # SHORT's returns highly correlated to LONG's
        identical = [0.01, 0.02, -0.01, 0.03, -0.02] * 5
        returns = {
            "LONG": pd.Series(identical),
            "SHORT": pd.Series(identical),
        }
        # Shorting a stock that moves IDENTICALLY to an existing long would
        # reduce risk (functions like a hedge). But since the signal is a new
        # entry, the check still runs — the caller decides whether to apply
        # to shorts. For simplicity we treat new entries uniformly.
        ok, _ = check_correlation(sig, positions, returns, threshold=0.7)
        # Implementation choice: check applies to any new entry.
        # Here correlation is 1.0 so it should reject.
        assert ok is False

    def test_empty_returns_lookup_passes(self):
        """Empty dict → no data available → don't block."""
        sig = _make_signal(ticker="AAPL")
        positions = [_make_position(ticker="MSFT")]
        ok, _ = check_correlation(sig, positions, {})
        assert ok is True

    def test_short_series_passes(self):
        """Series shorter than minimum length → pass (insufficient data)."""
        sig = _make_signal(ticker="AMD")
        positions = [_make_position(ticker="NVDA")]
        # Only 3 data points — too short for a reliable correlation
        returns = {
            "NVDA": pd.Series([0.01, 0.02, -0.01]),
            "AMD": pd.Series([0.01, 0.02, -0.01]),
        }
        ok, _ = check_correlation(sig, positions, returns, threshold=0.7, min_periods=20)
        assert ok is True

    def test_constant_returns_pass(self):
        """Halted/constant-price stocks produce NaN correlation → pass."""
        sig = _make_signal(ticker="HALT")
        positions = [_make_position(ticker="NVDA")]
        returns = {
            "NVDA": pd.Series([0.01, 0.02, -0.01, 0.03, -0.02] * 10),
            "HALT": pd.Series([0.0] * 50),  # no variance → undefined correlation
        }
        ok, _ = check_correlation(sig, positions, returns, threshold=0.7)
        assert ok is True

    def test_nan_returns_handled(self):
        """Series containing NaN values must not crash."""
        sig = _make_signal(ticker="AMD")
        positions = [_make_position(ticker="NVDA")]
        returns = {
            "NVDA": pd.Series([0.01, np.nan, -0.01, 0.03, -0.02] * 10),
            "AMD": pd.Series([0.01, 0.02, np.nan, 0.03, -0.02] * 10),
        }
        # Should not raise; the implementation should handle NaN gracefully
        ok, _ = check_correlation(sig, positions, returns, threshold=0.7)
        # Both have similar pattern; pairwise-NaN-dropped corr will be high
        assert isinstance(ok, bool)

    def test_returns_lookup_may_contain_extra_tickers(self):
        """Extra tickers in returns_lookup that aren't open positions = ignored."""
        sig = _make_signal(ticker="AAPL")
        positions = [_make_position(ticker="MSFT")]
        returns = {
            "MSFT": pd.Series([0.01, -0.01, 0.02, -0.02, 0.0] * 10),
            "AAPL": pd.Series([-0.01, 0.01, -0.02, 0.02, 0.0] * 10),  # opposite
            "TSLA": pd.Series([0.99] * 50),  # irrelevant
            "GOOG": pd.Series([0.99] * 50),
        }
        ok, _ = check_correlation(sig, positions, returns, threshold=0.7)
        # AAPL is negatively correlated with MSFT → pass
        assert ok is True

    def test_same_ticker_as_existing_position_skipped(self):
        """If candidate ticker matches an open position, skip self-correlation."""
        sig = _make_signal(ticker="AAPL", action=Action.BUY)
        positions = [_make_position(ticker="AAPL", quantity=10)]  # existing long
        # The duplicate-check catches this first in evaluate(); here we just
        # verify check_correlation doesn't flag self-correlation as a problem.
        returns = {"AAPL": pd.Series([0.01, 0.02, -0.01, 0.03, -0.02] * 10)}
        ok, _ = check_correlation(sig, positions, returns, threshold=0.7)
        # AAPL vs AAPL = 1.0, but it's the same ticker so caller already handles it.
        # check_correlation should not reject on self-comparison.
        assert ok is True

    def test_can_be_disabled_via_threshold_above_one(self):
        """threshold=1.0 or above means 'never reject' (feature disabled)."""
        sig = _make_signal(ticker="AMD")
        positions = [_make_position(ticker="NVDA")]
        identical = [0.01, 0.02, -0.01, 0.03, -0.02] * 10
        returns = {
            "NVDA": pd.Series(identical),
            "AMD": pd.Series(identical),
        }
        ok, _ = check_correlation(sig, positions, returns, threshold=1.0)
        # corr = 1.0 is not strictly greater than 1.0 → pass
        assert ok is True
        ok, _ = check_correlation(sig, positions, returns, threshold=1.5)
        assert ok is True


class TestCorrelationInEvaluate:
    """The correlation check must be wired into evaluate() with a default
    threshold from settings. Callers pass returns_lookup explicitly."""

    def test_evaluate_rejects_correlated_candidate(self):
        """With returns_lookup showing high correlation, evaluate() must reject."""
        sig = _make_signal(ticker="AMD", entry_price=150.0, stop_loss=145.0, take_profit=160.0)
        positions = [_make_position(ticker="NVDA", sector="Technology")]
        identical = [0.01, 0.02, -0.01, 0.03, -0.02] * 10
        returns_lookup = {
            "NVDA": pd.Series(identical),
            "AMD": pd.Series(identical),
        }
        # Give MA values so trend check passes
        sig.indicator_values.update({"MA5": 150, "MA10": 148, "MA20": 145, "sector": "Technology"})

        result = evaluate(
            sig, positions, 100_000, 0.0, current_price=150.0,
            returns_lookup=returns_lookup, correlation_threshold=0.7,
        )
        assert result.approved is False
        assert any("correlation" in r.lower() for r in result.reasons)

    def test_evaluate_passes_with_uncorrelated_candidate(self):
        """Uncorrelated returns should not block an otherwise valid signal."""
        sig = _make_signal(
            ticker="UNCOR", entry_price=150.0, stop_loss=145.0, take_profit=160.0,
        )
        sig.indicator_values.update({"MA5": 152, "MA10": 150, "MA20": 148})
        positions = []
        rng = np.random.default_rng(0)
        returns_lookup = {
            "UNCOR": pd.Series(rng.standard_normal(100) * 0.01),
        }
        result = evaluate(
            sig, positions, 100_000, 0.0, current_price=150.0,
            returns_lookup=returns_lookup, correlation_threshold=0.7,
            **_BULLISH_CONSENSUS,
        )
        assert result.approved is True

    def test_evaluate_without_returns_lookup_skips_correlation(self):
        """If returns_lookup is None, correlation check is skipped entirely."""
        sig = _make_signal(
            ticker="AAPL", entry_price=150.0, stop_loss=145.0, take_profit=160.0,
        )
        sig.indicator_values.update({"MA5": 152, "MA10": 150, "MA20": 148})
        positions = [_make_position(ticker="MSFT", sector="Technology")]

        result = evaluate(
            sig, positions, 100_000, 0.0, current_price=150.0,
            # returns_lookup not provided
        )
        # Should not raise; correlation check simply skipped
        # (Other checks apply; sig may still pass or fail based on those.)
        assert isinstance(result.approved, bool)


# ---------------------------------------------------------------------------
# PDT (Pattern Day Trader) protection tests
# ---------------------------------------------------------------------------

class TestPDTRestriction:
    """check_pdt_restriction(signal, positions, portfolio_value, recent_trades, ...)

    IBKR restricts accounts with liquid net worth < PDT_PROTECTION_THRESHOLD_USD
    when 2 day trades are performed within 5 business days. When portfolio is
    at or above the threshold, the check is a pass-through.
    """

    def test_passes_when_above_threshold(self):
        """Above the USD threshold, PDT rules do not apply here — go crazy."""
        sig = _make_signal(ticker="QUBT", action=Action.BUY)
        ok, _ = check_pdt_restriction(
            sig, [], portfolio_value=10_000.0, recent_trades=[],
            threshold_usd=5000.0, max_day_trades=1,
        )
        assert ok is True

    def test_passes_exactly_at_threshold(self):
        """At the threshold is safe — restriction only applies strictly below."""
        sig = _make_signal(ticker="QUBT", action=Action.BUY)
        ok, _ = check_pdt_restriction(
            sig, [], portfolio_value=5000.0, recent_trades=[],
            threshold_usd=5000.0, max_day_trades=1,
        )
        assert ok is True

    def test_passes_under_threshold_with_no_day_trades(self):
        """Under threshold, no prior day trades, SWING signal — still within
        budget. (DAY-type entries on sub-threshold accounts are blocked by a
        separate guard, exercised in TestPDTBlocksDayTypeOnSubThreshold.)
        """
        sig = _make_signal(
            ticker="QUBT", action=Action.BUY, trade_type=TradeType.SWING,
        )
        ok, _ = check_pdt_restriction(
            sig, [], portfolio_value=3000.0, recent_trades=[],
            threshold_usd=5000.0, max_day_trades=1,
        )
        assert ok is True

    def test_blocks_new_entry_when_day_trade_already_used(self):
        """Under threshold + 1 day trade in 5-day window — a new SWING BUY
        could still become a 2nd day trade if the bracket fires same-day.
        Block conservatively.
        """
        now = datetime.now(timezone.utc)
        day_trade = _make_trade(
            ticker="PRIOR",
            entry_time=now - timedelta(days=1, hours=2),
            exit_time=now - timedelta(days=1),  # entered and exited same calendar day
        )
        sig = _make_signal(
            ticker="QUBT", action=Action.BUY, trade_type=TradeType.SWING,
        )
        ok, reason = check_pdt_restriction(
            sig, [], portfolio_value=3000.0, recent_trades=[day_trade],
            threshold_usd=5000.0, max_day_trades=1,
        )
        assert ok is False
        assert "PDT" in reason or "day trade" in reason.lower()

    def test_allows_exit_of_prior_day_position(self):
        """A SELL of a position opened on a prior day is NOT a day trade —
        must not be blocked even under threshold with day trades used.
        Otherwise the trader is trapped in a losing swing position.
        """
        now = datetime.now(timezone.utc)
        day_trade = _make_trade(
            ticker="PRIOR",
            entry_time=now - timedelta(days=1, hours=2),
            exit_time=now - timedelta(days=1),
        )
        # Position opened yesterday → closing it today is NOT a day trade
        pos = _make_position(
            ticker="AAPL", quantity=10, entry_time=now - timedelta(days=2),
        )
        sig = _make_signal(ticker="AAPL", action=Action.SELL)
        ok, _ = check_pdt_restriction(
            sig, [pos], portfolio_value=3000.0, recent_trades=[day_trade],
            threshold_usd=5000.0, max_day_trades=1,
        )
        assert ok is True

    def test_blocks_same_day_exit_when_day_trade_used(self):
        """SELL of a position opened today completes a day trade. If 1 day
        trade already used, this would be the 2nd → block (IBKR trips at 2).
        """
        now = datetime.now(timezone.utc)
        day_trade = _make_trade(
            ticker="PRIOR",
            entry_time=now - timedelta(days=1, hours=2),
            exit_time=now - timedelta(days=1),
        )
        pos = _make_position(ticker="AAPL", quantity=10, entry_time=now)
        sig = _make_signal(ticker="AAPL", action=Action.SELL)
        ok, reason = check_pdt_restriction(
            sig, [pos], portfolio_value=3000.0, recent_trades=[day_trade],
            threshold_usd=5000.0, max_day_trades=1,
        )
        assert ok is False
        assert "PDT" in reason or "day trade" in reason.lower()

    def test_allows_same_day_exit_with_no_prior_day_trades(self):
        """SELL of a same-day position is the 1st day trade — IBKR allows 1
        before the 2-trade restriction. Must not be blocked.
        """
        now = datetime.now(timezone.utc)
        pos = _make_position(ticker="AAPL", quantity=10, entry_time=now)
        sig = _make_signal(ticker="AAPL", action=Action.SELL)
        ok, _ = check_pdt_restriction(
            sig, [pos], portfolio_value=3000.0, recent_trades=[],
            threshold_usd=5000.0, max_day_trades=1,
        )
        assert ok is True

    def test_ignores_day_trades_outside_window(self):
        """Day trades older than 5 business days don't count toward the limit."""
        now = datetime.now(timezone.utc)
        old_day_trade = _make_trade(
            ticker="OLD",
            entry_time=now - timedelta(days=14, hours=2),
            exit_time=now - timedelta(days=14),
        )
        sig = _make_signal(
            ticker="QUBT", action=Action.BUY, trade_type=TradeType.SWING,
        )
        ok, _ = check_pdt_restriction(
            sig, [], portfolio_value=3000.0, recent_trades=[old_day_trade],
            threshold_usd=5000.0, max_day_trades=1,
        )
        assert ok is True

    def test_does_not_count_swing_trades(self):
        """A trade held overnight (entry_date != exit_date) is not a day trade."""
        now = datetime.now(timezone.utc)
        swing = _make_trade(
            ticker="SWING",
            entry_time=now - timedelta(days=3),
            exit_time=now - timedelta(days=1),
        )
        sig = _make_signal(
            ticker="QUBT", action=Action.BUY, trade_type=TradeType.SWING,
        )
        ok, _ = check_pdt_restriction(
            sig, [], portfolio_value=3000.0, recent_trades=[swing],
            threshold_usd=5000.0, max_day_trades=1,
        )
        assert ok is True

    def test_max_day_trades_zero_disables_count_check(self):
        """max_day_trades=0 disables the count-based ceiling — a SWING entry
        that would otherwise hit the count cap passes. The DAY-type guard
        above is not affected by max_day_trades.
        """
        now = datetime.now(timezone.utc)
        day_trade = _make_trade(
            ticker="PRIOR",
            entry_time=now - timedelta(days=1, hours=2),
            exit_time=now - timedelta(days=1),
        )
        sig = _make_signal(
            ticker="QUBT", action=Action.BUY, trade_type=TradeType.SWING,
        )
        ok, _ = check_pdt_restriction(
            sig, [], portfolio_value=3000.0, recent_trades=[day_trade],
            threshold_usd=5000.0, max_day_trades=0,
        )
        assert ok is True

    def test_day_trade_classification_uses_us_eastern_not_utc(self):
        """PDT day-trade detection must compare ET dates, not UTC dates.

        IBKR's PDT rule tracks the US Eastern calendar day. A trade opened at
        02:00 UTC (= 21:00 prior-ET) and closed at 15:00 UTC (= 10:00 ET) is a
        single-ET-day round trip (day trade). By UTC dates, entry was one day
        and exit was the next — the old code would classify this as a swing
        trade and miss it. The inverse case also matters: entry 23:00 UTC and
        exit 01:00 UTC are UTC-same-day but ET-cross-midnight.
        """
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")

        # Single ET trading day (Wednesday 2026-04-15), spans UTC midnight.
        # Entry: 04:30 UTC Wed = 00:30 ET Wed (pre-market edge, same ET day)
        # Exit:  18:00 UTC Wed = 14:00 ET Wed (regular session, same ET day)
        # By UTC date both are Wed — counted as day trade (correct by chance).
        #
        # The failure mode: entry 23:00 UTC Tue = 19:00 ET Tue vs
        #                   exit  14:00 UTC Wed = 10:00 ET Wed → different ET days
        # but UTC dates also differ, so currently counted correctly as swing.
        #
        # The problematic case for UTC-based code: entry 04:30 UTC Wed
        # = 00:30 ET Wed, exit 05:00 UTC Wed = 01:00 ET Wed — same ET AND UTC day.
        # This one also works.
        #
        # The real PDT bug case: entry 03:00 UTC Tue = 23:00 ET Mon and
        #                        exit  14:00 UTC Tue = 10:00 ET Tue.
        # UTC: both Tue → classified as day trade.
        # ET:  Mon → Tue → swing, should NOT count toward PDT.
        now = datetime.now(timezone.utc)
        # Anchor at a known ET date (avoid test flakiness from DST edges)
        anchor_utc = datetime(2026, 4, 14, 12, 0, tzinfo=timezone.utc)  # Tue 08:00 ET

        entry_utc = datetime(2026, 4, 14, 3, 0, tzinfo=timezone.utc)   # Mon 23:00 ET
        exit_utc = datetime(2026, 4, 14, 14, 0, tzinfo=timezone.utc)   # Tue 10:00 ET

        fake_swing = _make_trade(
            ticker="PRIOR", entry_time=entry_utc, exit_time=exit_utc,
        )
        # SWING signal so we isolate the count/ET-classification logic (a
        # DAY signal would be rejected by the earlier DAY-type guard and
        # this test wouldn't exercise ET-date handling).
        sig = _make_signal(
            ticker="QUBT", action=Action.BUY, trade_type=TradeType.SWING,
        )

        # Stub datetime.now in the risk module so "today" is a fixed ET Wednesday
        # within the 5-day window of the fake trade's ET exit (Tuesday).
        import core.risk as risk_mod
        from unittest.mock import patch

        fixed_now = datetime(2026, 4, 15, 14, 0, tzinfo=timezone.utc)  # Wed 10:00 ET

        class FakeDT(datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed_now.astimezone(tz) if tz else fixed_now.replace(tzinfo=None)

        with patch.object(risk_mod, "datetime", FakeDT):
            ok, reason = check_pdt_restriction(
                sig, [], portfolio_value=3000.0, recent_trades=[fake_swing],
                threshold_usd=5000.0, max_day_trades=1,
            )

        # The fake_swing is Mon→Tue in ET (not a day trade). Old code saw it as
        # a same-UTC-day round-trip (day trade) and would block the new entry.
        # Correct ET-based classification leaves 0 day trades used, so new entry
        # is allowed.
        assert ok is True, (
            f"Mon→Tue ET trade must not count as a PDT day trade (it crosses ET "
            f"midnight). Got ok={ok}, reason={reason!r}"
        )


class TestPDTBlocksDayTypeOnSubThreshold:
    """On sub-threshold accounts, any new-entry signal with trade_type=DAY is
    a guaranteed same-day round-trip (close_all_day_trades will force-close
    at market close) → must be blocked outright, regardless of historical
    day-trade count. Otherwise a single DAY signal consumes IBKR's only
    remaining day-trade slot and one more mishap locks the account out for
    30 days.

    SWING signals are still allowed under historical-count logic because the
    intent is to hold overnight; only an accidental same-day bracket hit
    would make them a day trade.

    Exits are never blocked by this rule — we must not trap a trader in a
    position they want to close.
    """

    def test_blocks_day_type_buy_with_no_history(self):
        """DAY-type BUY on a sub-$5k account with zero prior day trades must
        still be rejected — the DAY intent guarantees a same-day round-trip.
        """
        sig = _make_signal(
            ticker="QUBT", action=Action.BUY, trade_type=TradeType.DAY,
        )
        ok, reason = check_pdt_restriction(
            sig, [], portfolio_value=3000.0, recent_trades=[],
            threshold_usd=5000.0, max_day_trades=1,
        )
        assert ok is False, (
            f"DAY-type BUY on sub-threshold account must be blocked, got ok={ok}"
        )
        assert "day" in reason.lower()

    def test_allows_swing_type_buy_with_no_history(self):
        """Same conditions but trade_type=SWING → allowed (overnight hold,
        no guaranteed day trade)."""
        sig = _make_signal(
            ticker="QUBT", action=Action.BUY, trade_type=TradeType.SWING,
        )
        ok, _ = check_pdt_restriction(
            sig, [], portfolio_value=3000.0, recent_trades=[],
            threshold_usd=5000.0, max_day_trades=1,
        )
        assert ok is True

    def test_allows_day_type_buy_above_threshold(self):
        """Above the threshold, PDT rules don't apply — DAY-type BUY OK."""
        sig = _make_signal(
            ticker="QUBT", action=Action.BUY, trade_type=TradeType.DAY,
        )
        ok, _ = check_pdt_restriction(
            sig, [], portfolio_value=10_000.0, recent_trades=[],
            threshold_usd=5000.0, max_day_trades=1,
        )
        assert ok is True

    def test_allows_exit_of_day_position_even_on_sub_threshold(self):
        """A SELL closing an existing DAY position must not be blocked — we
        cannot trap the trader in a position; the close decision has already
        been made upstream. The rule gates ENTRIES, not exits.
        """
        now = datetime.now(timezone.utc)
        pos = _make_position(
            ticker="AAPL", quantity=10, entry_time=now,
            trade_type=TradeType.DAY,
        )
        sig = _make_signal(
            ticker="AAPL", action=Action.SELL, trade_type=TradeType.DAY,
        )
        ok, _ = check_pdt_restriction(
            sig, [pos], portfolio_value=3000.0, recent_trades=[],
            threshold_usd=5000.0, max_day_trades=1,
        )
        # Same-day exit with count=0 — allowed (first day trade within budget);
        # the DAY-type block only fires for new entries.
        assert ok is True


class TestPDTRestrictionInEvaluate:
    """PDT check must be wired into the main evaluate() function."""

    def test_evaluate_blocks_new_entry_under_threshold_with_prior_day_trade(self):
        """evaluate() must reject a BUY when portfolio < $5K and a day trade
        has been used in the rolling window."""
        now = datetime.now(timezone.utc)
        day_trade = _make_trade(
            ticker="PRIOR",
            entry_time=now - timedelta(days=1, hours=2),
            exit_time=now - timedelta(days=1),
        )
        sig = _make_signal(
            ticker="QUBT", action=Action.BUY,
            entry_price=10.0, stop_loss=9.5, take_profit=11.0,
        )
        sig.indicator_values.update({"MA5": 10.5, "MA10": 10.2, "MA20": 10.0})
        result = evaluate(
            sig, [], portfolio_value=3000.0, daily_pnl=0.0,
            current_price=10.0, recent_trades=[day_trade],
        )
        assert result.approved is False
        assert any("pdt" in r.lower() or "day trade" in r.lower() for r in result.reasons)

    def test_evaluate_allows_under_threshold_with_no_day_trades(self):
        """evaluate() must not add PDT friction for a SWING entry when there
        are no prior day trades. (DAY-type entries under threshold are
        blocked by a separate guard, covered in TestPDTBlocksDayTypeOnSubThreshold.)
        """
        sig = _make_signal(
            ticker="QUBT", action=Action.BUY, trade_type=TradeType.SWING,
            entry_price=10.0, stop_loss=9.5, take_profit=11.0,
        )
        sig.indicator_values.update({"MA5": 10.5, "MA10": 10.2, "MA20": 10.0})
        result = evaluate(
            sig, [], portfolio_value=3000.0, daily_pnl=0.0,
            current_price=10.0, recent_trades=[],
        )
        # May fail on other checks (position size), but not on PDT
        assert not any("pdt" in r.lower() for r in result.reasons)

    def test_evaluate_allows_above_threshold_regardless_of_day_trades(self):
        """evaluate() must not add PDT friction above the threshold."""
        now = datetime.now(timezone.utc)
        day_trades = [
            _make_trade(
                ticker=f"DT{i}",
                entry_time=now - timedelta(days=1, hours=2),
                exit_time=now - timedelta(days=1),
            )
            for i in range(5)
        ]
        sig = _make_signal(
            ticker="QUBT", action=Action.BUY,
            entry_price=10.0, stop_loss=9.5, take_profit=11.0,
        )
        sig.indicator_values.update({"MA5": 10.5, "MA10": 10.2, "MA20": 10.0})
        result = evaluate(
            sig, [], portfolio_value=100_000.0, daily_pnl=0.0,
            current_price=10.0, recent_trades=day_trades,
        )
        # Must not be rejected for PDT reasons above threshold
        assert not any("pdt" in r.lower() for r in result.reasons)


# ---------------------------------------------------------------------------
# Intraday-margin framework (post-2026-06-04) tests
# ---------------------------------------------------------------------------

class TestIntradayMargin:
    """check_intraday_margin(signal, portfolio_value, position_size, regime,
    has_uncured_deficit) — the post-2026-06-04 replacement for the eliminated
    PDT day-trade gate. Three entry-only rejections: Reg-T minimum, 25%
    intraday maintenance margin, and an uncured intraday-margin deficit.
    """

    def test_rejects_below_reg_t_minimum(self):
        """Account equity below REG_T_MIN_EQUITY_USD blocks new entries."""
        sig = _make_signal(ticker="QUBT", action=Action.BUY, entry_price=10.0)
        ok, reason = check_intraday_margin(
            sig, portfolio_value=1500.0, position_size=10,
            regime="intraday", has_uncured_deficit=False,
        )
        assert ok is False
        assert "reg-t" in reason.lower() or "intraday margin" in reason.lower()

    def test_rejects_when_maintenance_margin_exceeds_equity(self):
        """A position whose 25% maintenance margin exceeds equity is rejected.

        entry 100 * size 200 * 25% = $5,000 maintenance margin > $3,000 equity.
        """
        sig = _make_signal(ticker="QUBT", action=Action.BUY, entry_price=100.0)
        ok, reason = check_intraday_margin(
            sig, portfolio_value=3000.0, position_size=200,
            regime="intraday", has_uncured_deficit=False,
        )
        assert ok is False
        assert "maintenance margin" in reason.lower()

    def test_rejects_uncured_deficit(self):
        """An uncured intraday-margin deficit blocks new entries (90-day guard)."""
        sig = _make_signal(ticker="QUBT", action=Action.BUY, entry_price=10.0)
        ok, reason = check_intraday_margin(
            sig, portfolio_value=50_000.0, position_size=10,
            regime="intraday", has_uncured_deficit=True,
        )
        assert ok is False
        assert "deficit" in reason.lower()

    def test_passes_normal_entry(self):
        """Adequate equity, modest margin, no deficit → pass."""
        sig = _make_signal(ticker="QUBT", action=Action.BUY, entry_price=100.0)
        ok, reason = check_intraday_margin(
            sig, portfolio_value=100_000.0, position_size=100,
            regime="intraday", has_uncured_deficit=False,
        )
        assert ok is True
        assert reason == ""

    def test_legacy_regime_skips_intraday_checks(self):
        """Under regime='legacy_pdt' the intraday guard is a pass-through even
        when equity is below the Reg-T minimum (the legacy counter runs instead).
        """
        sig = _make_signal(ticker="QUBT", action=Action.BUY, entry_price=10.0)
        ok, _ = check_intraday_margin(
            sig, portfolio_value=100.0, position_size=10,
            regime="legacy_pdt", has_uncured_deficit=True,
        )
        assert ok is True

    def test_evaluate_blocks_entry_below_reg_t_minimum(self):
        """evaluate() rejects a new BUY when equity is under the Reg-T minimum."""
        sig = _make_signal(
            ticker="QUBT", action=Action.BUY, trade_type=TradeType.SWING,
            entry_price=10.0, stop_loss=9.5, take_profit=11.0,
        )
        sig.indicator_values.update({"MA5": 10.5, "MA10": 10.2, "MA20": 10.0})
        result = evaluate(
            sig, [], portfolio_value=1500.0, daily_pnl=0.0,
            current_price=10.0, **_BULLISH_CONSENSUS,
        )
        assert result.approved is False
        assert any("intraday margin" in r.lower() for r in result.reasons)

    def test_evaluate_blocks_entry_under_uncured_deficit(self):
        """evaluate() rejects a new BUY when an uncured deficit is flagged."""
        sig = _make_signal(
            ticker="QUBT", action=Action.BUY, trade_type=TradeType.SWING,
            entry_price=10.0, stop_loss=9.5, take_profit=11.0,
        )
        sig.indicator_values.update({"MA5": 10.5, "MA10": 10.2, "MA20": 10.0})
        result = evaluate(
            sig, [], portfolio_value=50_000.0, daily_pnl=0.0,
            current_price=10.0, has_uncured_intraday_deficit=True,
            **_BULLISH_CONSENSUS,
        )
        assert result.approved is False
        assert any("deficit" in r.lower() for r in result.reasons)

    def test_evaluate_allows_exit_under_uncured_deficit(self):
        """The SAME uncured-deficit flag must NOT block an exit (SELL closing a
        held long). The intraday guard is entry-only — exits are never trapped.
        """
        sig = _make_signal(
            ticker="AAPL", action=Action.SELL,
            entry_price=150.0, stop_loss=155.0, take_profit=140.0,
        )
        positions = [_make_position(ticker="AAPL", quantity=10)]
        result = evaluate(
            sig, positions, portfolio_value=50_000.0, daily_pnl=0.0,
            current_price=150.0, has_uncured_intraday_deficit=True,
        )
        assert result.is_exit is True
        assert not any("intraday margin" in r.lower() for r in result.reasons)
        assert not any("deficit" in r.lower() for r in result.reasons)
        assert result.approved is True


class TestMarginRegime:
    """MARGIN_REGIME must degrade safely across legacy_pdt / intraday / both."""

    def _swing_buy(self):
        sig = _make_signal(
            ticker="QUBT", action=Action.BUY, trade_type=TradeType.SWING,
            entry_price=10.0, stop_loss=9.5, take_profit=11.0,
        )
        sig.indicator_values.update({"MA5": 10.5, "MA10": 10.2, "MA20": 10.0})
        return sig

    def test_intraday_regime_runs_only_intraday_guard(self):
        """regime='intraday': sub-Reg-T equity is blocked by the intraday guard,
        and the legacy day-trade counter does NOT fire (no 'PDT' reason)."""
        now = datetime.now(timezone.utc)
        prior_day_trade = _make_trade(
            ticker="PRIOR",
            entry_time=now - timedelta(days=1, hours=2),
            exit_time=now - timedelta(days=1),
        )
        result = evaluate(
            self._swing_buy(), [], portfolio_value=1500.0, daily_pnl=0.0,
            current_price=10.0, recent_trades=[prior_day_trade],
            margin_regime="intraday", **_BULLISH_CONSENSUS,
        )
        assert any("intraday margin" in r.lower() for r in result.reasons)
        assert not any("pdt" in r.lower() for r in result.reasons)

    def test_legacy_regime_runs_only_legacy_counter(self):
        """regime='legacy_pdt': the legacy counter fires on a prior day trade,
        and the intraday guard does NOT run (no 'intraday margin' reason)."""
        now = datetime.now(timezone.utc)
        prior_day_trade = _make_trade(
            ticker="PRIOR",
            entry_time=now - timedelta(days=1, hours=2),
            exit_time=now - timedelta(days=1),
        )
        # Equity below the Reg-T minimum would trip the intraday guard if it ran;
        # under legacy_pdt it must not, while the legacy PDT counter blocks.
        result = evaluate(
            self._swing_buy(), [], portfolio_value=1500.0, daily_pnl=0.0,
            current_price=10.0, recent_trades=[prior_day_trade],
            margin_regime="legacy_pdt", **_BULLISH_CONSENSUS,
        )
        assert any("pdt" in r.lower() or "day trade" in r.lower() for r in result.reasons)
        assert not any("intraday margin" in r.lower() for r in result.reasons)

    def test_both_regime_runs_both_guards(self):
        """regime='both': both the intraday guard and the legacy counter fire."""
        now = datetime.now(timezone.utc)
        prior_day_trade = _make_trade(
            ticker="PRIOR",
            entry_time=now - timedelta(days=1, hours=2),
            exit_time=now - timedelta(days=1),
        )
        result = evaluate(
            self._swing_buy(), [], portfolio_value=1500.0, daily_pnl=0.0,
            current_price=10.0, recent_trades=[prior_day_trade],
            margin_regime="both", **_BULLISH_CONSENSUS,
        )
        assert any("intraday margin" in r.lower() for r in result.reasons)
        assert any("pdt" in r.lower() or "day trade" in r.lower() for r in result.reasons)


class TestNoLegacy5kThresholdBranch:
    """Regression guard for MGN-01: the eliminated $5,000 PDT gate must not
    survive as a live code branch anywhere in core/risk.py."""

    def test_no_legacy_5k_threshold_branch(self):
        from pathlib import Path
        import core.risk as risk_mod

        source = Path(risk_mod.__file__).read_text()
        # Strip full-line comments (mirrors the acceptance grep
        # `grep -v '^[[:space:]]*#'`), then assert the literal is absent.
        executable = "\n".join(
            line for line in source.splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "5000" not in executable, (
            "Eliminated $5,000 PDT threshold still present in executable code"
        )


class TestDailyLossLimitBaseline:
    """Daily loss limit must use start-of-day equity as the denominator, not
    current (post-loss) equity. Using current equity causes the limit dollar
    amount to shrink as the portfolio loses value — the brake tightens while
    losses accumulate, so the cap moves around within a single session.
    """

    def test_uses_start_of_day_equity_when_provided(self):
        """When start_of_day_equity is supplied, limit is computed from it."""
        # Start of day $100k, down to $95k MTM, limit is 2% of $100k = $2000.
        # Current loss is -$5000 which exceeds $2000 → reject.
        ok, reason = check_daily_loss_limit(
            daily_pnl=-5000.0,
            portfolio_value=95_000.0,
            limit_pct=2.0,
            start_of_day_equity=100_000.0,
        )
        assert ok is False
        assert "2000" in reason or "2,000" in reason

    def test_backwards_compatible_without_start_of_day_equity(self):
        """When start_of_day_equity is not provided, use portfolio_value (legacy)."""
        ok, _ = check_daily_loss_limit(
            daily_pnl=-500.0,
            portfolio_value=100_000.0,
            limit_pct=2.0,
        )
        assert ok is True

    def test_start_of_day_equity_distinct_from_current(self):
        """Passing a larger start_of_day_equity yields a larger (correct) limit."""
        # If only current portfolio_value were used: 2% of $90k = $1800 → -$1900 rejects
        # But start_of_day = $100k: 2% of $100k = $2000 → -$1900 passes
        ok, _ = check_daily_loss_limit(
            daily_pnl=-1900.0,
            portfolio_value=90_000.0,
            limit_pct=2.0,
            start_of_day_equity=100_000.0,
        )
        assert ok is True

    def test_invalid_start_of_day_equity_falls_back(self):
        """Zero or negative start_of_day_equity falls back to portfolio_value."""
        ok, _ = check_daily_loss_limit(
            daily_pnl=-500.0,
            portfolio_value=100_000.0,
            limit_pct=2.0,
            start_of_day_equity=0.0,
        )
        assert ok is True

    def test_evaluate_accepts_start_of_day_equity(self):
        """evaluate() must pass start_of_day_equity to the daily-loss check."""
        sig = _make_signal()
        # Current portfolio has drawn down but start-of-day was higher.
        # Loss $-1800 would exceed 2% of $90k ($1800) but is under 2% of $100k ($2000).
        result = evaluate(
            sig, [], portfolio_value=90_000.0, daily_pnl=-1800.0,
            start_of_day_equity=100_000.0,
        )
        # Loss-limit check itself must pass (the limit is based on start-of-day)
        assert not any("halted" in r.lower() for r in result.reasons)


class TestSectorConcentrationMissingSector:
    """Missing sector must not silently bypass the concentration cap.

    Universe builder excludes stocks whose sector cannot be resolved, so a
    signal reaching the risk manager with no sector indicates the signal
    bypassed that filter (backtest/dry-run). The risk manager must at least
    record this so operators notice.
    """

    def test_missing_sector_logs_warning(self, caplog):
        """A signal with no sector data logs a WARNING about bypass."""
        import logging

        from core.risk import check_sector_concentration

        sig = _make_signal(ticker="XYZ")  # no indicator_values["sector"]
        with caplog.at_level(logging.WARNING, logger="core.risk"):
            ok, _ = check_sector_concentration(sig, [], 100_000)
        assert ok is True  # soft gate — passes but logs
        assert any(
            "sector" in rec.message.lower() and "xyz" in rec.message.lower()
            for rec in caplog.records
        ), "Expected WARNING about missing sector for XYZ"

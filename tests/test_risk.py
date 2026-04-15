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


class TestFullEvaluation:
    def test_approved(self):
        sig = _make_signal()
        result = evaluate(sig, [], 100_000, 0)
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
        result = evaluate(sig, [], 100_000, 0)
        # Should still pass with no positions
        assert result.approved is True


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
        result = evaluate(sig, [], 100_000, 0, recent_trades=[])
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
        result = evaluate(sig, [], 100_000, 0, recent_trades=[])
        assert result.approved is True

    def test_evaluate_backward_compatible(self):
        """evaluate() still works without recent_trades (defaults to no trades)."""
        sig = _make_signal()
        result = evaluate(sig, [], 100_000, 0)
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

    def test_extreme_vol_floors_at_one_share(self):
        """Even in extreme volatility, should allow at least 1 share if base > 0."""
        sig = _make_signal(entry_price=50.0, stop_loss=45.0)
        qty = calculate_position_size(sig, 100_000, volatility=1.5)
        # With 150% annualized vol, position should be tiny but >= 1
        assert qty >= 1


class TestVolatilityInEvaluate:
    """evaluate() passes volatility through to position sizing."""

    def test_high_vol_reduces_approved_size(self):
        sig = _make_signal(entry_price=150.0, stop_loss=145.0)
        result_normal = evaluate(sig, [], 100_000, 0)
        result_high_vol = evaluate(sig, [], 100_000, 0, volatility=0.40)
        assert result_normal.approved is True
        assert result_high_vol.approved is True
        assert result_high_vol.position_size < result_normal.position_size

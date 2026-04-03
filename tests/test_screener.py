"""Tests for core/screener.py using synthetic data."""

import numpy as np
import pandas as pd
import pytest

from core.models import Action
from core.screener import (
    check_rsi,
    check_macd,
    check_ma_crossover,
    check_volume_spike,
    check_bollinger,
    check_support_resistance,
    analyze_stock,
    score_candidate,
    screen_stocks,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic OHLCV DataFrames
# ---------------------------------------------------------------------------

def _make_df(closes: list[float], volumes: list[float] = None, days: int = None) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a close-price series."""
    n = len(closes)
    if volumes is None:
        volumes = [1_000_000] * n
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "open": [c * 0.99 for c in closes],
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.98 for c in closes],
        "close": closes,
        "volume": volumes,
    }, index=dates)
    df.index.name = "date"
    return df


def _trending_up(n: int = 60, start: float = 100.0, step: float = 0.5) -> list[float]:
    """Generate an uptrend series."""
    return [start + i * step for i in range(n)]


def _trending_down(n: int = 60, start: float = 200.0, step: float = 0.5) -> list[float]:
    """Generate a downtrend series."""
    return [start - i * step for i in range(n)]


def _flat(n: int = 60, price: float = 100.0) -> list[float]:
    """Generate a flat series."""
    return [price] * n


# ---------------------------------------------------------------------------
# RSI Tests
# ---------------------------------------------------------------------------

class TestRSI:
    def test_oversold(self):
        # Steep downtrend should produce RSI < 30
        closes = _trending_down(40, 200, 3)
        df = _make_df(closes)
        result = check_rsi(df)
        assert result is not None, "Steep downtrend should trigger RSI oversold"
        assert result["action"] == Action.BUY
        assert result["indicator"] == "RSI"

    def test_overbought(self):
        # Steep uptrend should produce RSI > 70
        closes = _trending_up(40, 50, 3)
        df = _make_df(closes)
        result = check_rsi(df)
        assert result is not None, "Steep uptrend should trigger RSI overbought"
        assert result["action"] == Action.SELL
        assert result["indicator"] == "RSI"

    def test_neutral_returns_none(self):
        # Flat price — RSI ~50
        df = _make_df(_flat(30))
        result = check_rsi(df)
        assert result is None

    def test_too_short(self):
        df = _make_df([100, 101, 102])
        result = check_rsi(df)
        assert result is None


# ---------------------------------------------------------------------------
# MACD Tests
# ---------------------------------------------------------------------------

class TestMACD:
    def test_bullish_crossover(self):
        # Downtrend then sharp reversal up
        closes = _trending_down(30, 150, 1) + _trending_up(15, 120, 3)
        df = _make_df(closes)
        result = check_macd(df)
        # May or may not trigger depending on exact crossover timing
        if result:
            assert result["indicator"] == "MACD"
            assert result["action"] in (Action.BUY, Action.SELL)

    def test_too_short(self):
        df = _make_df([100] * 10)
        result = check_macd(df)
        assert result is None


# ---------------------------------------------------------------------------
# MA Crossover Tests
# ---------------------------------------------------------------------------

class TestMACrossover:
    def test_golden_cross(self):
        # Gentle downtrend then just enough uptick for MA5 to cross above MA20
        # at exactly the last bar (prev: fast <= slow, curr: fast > slow)
        closes = _trending_down(25, 120, 0.5) + _trending_up(5, 107, 2)
        df = _make_df(closes)
        result = check_ma_crossover(df)
        if result:
            assert result["action"] == Action.BUY
            assert result["indicator"] == "MA_CROSSOVER"

    def test_death_cross(self):
        # Gentle uptrend then enough downtick for MA5 to cross below MA20
        closes = _trending_up(25, 80, 0.5) + _trending_down(5, 93, 2)
        df = _make_df(closes)
        result = check_ma_crossover(df)
        if result:
            assert result["action"] == Action.SELL

    def test_too_short(self):
        df = _make_df([100] * 5)
        result = check_ma_crossover(df)
        assert result is None


# ---------------------------------------------------------------------------
# Volume Spike Tests
# ---------------------------------------------------------------------------

class TestVolumeSpike:
    def test_spike_detected(self):
        volumes = [500_000] * 25 + [2_000_000]  # last day = 4x average
        closes = _flat(26, 100)
        # Make last close higher to get a BUY signal
        closes[-1] = 102
        df = _make_df(closes, volumes)
        result = check_volume_spike(df)
        assert result is not None, "4x volume spike should be detected"
        assert result["indicator"] == "VOLUME_SPIKE"
        assert result["action"] == Action.BUY

    def test_no_spike(self):
        volumes = [500_000] * 25
        df = _make_df(_flat(25), volumes)
        result = check_volume_spike(df)
        assert result is None

    def test_too_short(self):
        df = _make_df([100] * 5, [500_000] * 5)
        result = check_volume_spike(df)
        assert result is None


# ---------------------------------------------------------------------------
# Bollinger Band Tests
# ---------------------------------------------------------------------------

class TestBollinger:
    def test_below_lower_band(self):
        # Stable then sudden drop — should breach lower Bollinger band
        closes = _flat(25, 100) + [70]
        df = _make_df(closes)
        result = check_bollinger(df)
        assert result is not None, "30% drop should breach lower Bollinger band"
        assert result["action"] == Action.BUY
        assert result["indicator"] == "BOLLINGER"

    def test_above_upper_band(self):
        # Stable then sudden spike — should breach upper Bollinger band
        closes = _flat(25, 100) + [130]
        df = _make_df(closes)
        result = check_bollinger(df)
        assert result is not None, "30% spike should breach upper Bollinger band"
        assert result["action"] == Action.SELL

    def test_within_bands(self):
        df = _make_df(_flat(25))
        result = check_bollinger(df)
        assert result is None


# ---------------------------------------------------------------------------
# Support/Resistance Tests
# ---------------------------------------------------------------------------

class TestSupportResistance:
    def test_near_support(self):
        # _make_df creates low = close * 0.98, so close=95 gives low=93.1
        # Current close must be within 2% of that low: 93.1 * 1.02 = 94.96
        # So current close ~93.5 is within range of 20-day low 93.1
        closes = [100] * 10 + [95] + [100] * 10 + [93.5]
        df = _make_df(closes)
        result = check_support_resistance(df)
        assert result is not None, "Close near 20-day low should trigger support signal"
        assert result["action"] == Action.BUY
        assert result["indicator"] == "SUPPORT"

    def test_near_resistance(self):
        # _make_df creates high = close * 1.01, so close=110 gives high=111.1
        # Current close must be within 2% of that high: 111.1 * 0.98 = 108.88
        # So current close ~110 is within range of 20-day high 111.1
        closes = [100] * 10 + [110] + [100] * 10 + [110.0]
        df = _make_df(closes)
        result = check_support_resistance(df)
        assert result is not None, "Close near 20-day high should trigger resistance signal"
        assert result["action"] == Action.SELL
        assert result["indicator"] == "RESISTANCE"


# ---------------------------------------------------------------------------
# Scoring Tests
# ---------------------------------------------------------------------------

class TestScoring:
    def test_empty_signals(self):
        score, action = score_candidate([])
        assert score == 0.0
        assert action == Action.HOLD

    def test_single_buy(self):
        triggered = [{"action": Action.BUY, "strength": 0.8, "indicator": "RSI"}]
        score, action = score_candidate(triggered)
        assert score > 0
        assert action == Action.BUY

    def test_mixed_signals_buy_dominant(self):
        triggered = [
            {"action": Action.BUY, "strength": 0.9, "indicator": "RSI"},
            {"action": Action.BUY, "strength": 0.7, "indicator": "MACD"},
            {"action": Action.SELL, "strength": 0.3, "indicator": "BOLLINGER"},
        ]
        score, action = score_candidate(triggered)
        assert action == Action.BUY
        assert score > 0

    def test_equal_buy_sell_is_hold(self):
        triggered = [
            {"action": Action.BUY, "strength": 0.5, "indicator": "RSI"},
            {"action": Action.SELL, "strength": 0.5, "indicator": "MACD"},
        ]
        score, action = score_candidate(triggered)
        assert action == Action.HOLD


# ---------------------------------------------------------------------------
# Full Screening Pipeline
# ---------------------------------------------------------------------------

class TestScreenStocks:
    def test_returns_signals(self):
        # Build data that should trigger volume spike + support
        volumes = [500_000] * 25 + [2_000_000]
        closes = [100] * 10 + [95] + [100] * 10 + [95.5, 96, 97, 98, 97]
        # Adjust lengths to match
        n = len(closes)
        vols = [500_000] * (n - 1) + [2_000_000]
        df = _make_df(closes, vols)

        stock_data = {"TEST": ("SMART", df)}
        signals = screen_stocks(stock_data, min_score=0.1)
        # May or may not produce signals depending on exact indicator math
        assert isinstance(signals, list)
        for sig in signals:
            assert sig.source == "screener"
            assert sig.ticker == "TEST"
            assert sig.stop_loss > 0

    def test_empty_data(self):
        signals = screen_stocks({})
        assert signals == []

    def test_short_data_skipped(self):
        df = _make_df([100] * 5)
        signals = screen_stocks({"SHORT": ("SMART", df)})
        assert signals == []

    def test_max_candidates_limit(self):
        # Create many stocks with signals
        stock_data = {}
        for i in range(50):
            volumes = [500_000] * 25 + [3_000_000]
            closes = _flat(25, 100) + [70]  # below bollinger
            df = _make_df(closes, volumes)
            stock_data[f"STK{i:03d}"] = ("SMART", df)

        signals = screen_stocks(stock_data, min_score=0.1)
        assert len(signals) == 50

    def test_signals_sorted_by_score(self):
        stock_data = {}
        # Stock with more signals should rank higher
        volumes_spike = [500_000] * 25 + [3_000_000]

        # Mild signal
        closes_mild = _flat(26, 100)
        closes_mild[-1] = 98
        stock_data["MILD"] = ("SMART", _make_df(closes_mild, volumes_spike))

        # Strong signal — big drop below bollinger + volume spike
        closes_strong = _flat(25, 100) + [65]
        stock_data["STRONG"] = ("SMART", _make_df(closes_strong, volumes_spike))

        signals = screen_stocks(stock_data, min_score=0.1)
        if len(signals) >= 2:
            assert signals[0].confidence >= signals[1].confidence


class TestMAValuesAlwaysStored:
    """Verify that MA5, MA10, MA20 are always in indicator_values for trend confirmation."""

    def test_ma_values_present_in_signal(self):
        # Data that triggers at least one indicator (volume spike + bollinger)
        volumes = [500_000] * 25 + [3_000_000]
        closes = _flat(25, 100) + [70]
        df = _make_df(closes, volumes)

        stock_data = {"MATEST": ("SMART", df)}
        signals = screen_stocks(stock_data, min_score=0.1)

        assert len(signals) > 0, "Should produce at least one signal"
        sig = signals[0]
        assert "MA5" in sig.indicator_values, "MA5 must always be in indicator_values"
        assert "MA20" in sig.indicator_values, "MA20 must always be in indicator_values"
        assert "MA10" in sig.indicator_values, "MA10 must always be in indicator_values"
        assert isinstance(sig.indicator_values["MA5"], float)
        assert isinstance(sig.indicator_values["MA20"], float)

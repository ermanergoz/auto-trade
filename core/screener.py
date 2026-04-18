"""Technical screener — pure functions that accept DataFrames as input.

The screener never fetches data itself. The caller provides OHLCV DataFrames
so that the backtester can feed historical data without code duplication.
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import pandas_ta as ta

from config.settings import (
    RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    MA_FAST, MA_SLOW,
    VOLUME_SPIKE_MULTIPLIER,
    BOLLINGER_PERIOD, BOLLINGER_STD,
    SUPPORT_RESISTANCE_PCT,
    DEFAULT_STOP_LOSS_PCT, DEFAULT_TAKE_PROFIT_PCT,
    INDICATOR_WEIGHTS,
    MAX_EXTENSION_OVER_MA20_PCT,
)
from core.models import Signal, Action, TradeType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual indicator checks — all pure functions on a DataFrame
# ---------------------------------------------------------------------------

def check_rsi(df: pd.DataFrame) -> Optional[dict]:
    """Check RSI(14). Flag if < 30 (oversold) or > 70 (overbought).

    Returns dict with signal info or None.
    """
    if len(df) < RSI_PERIOD + 1:
        return None

    rsi = ta.rsi(df["close"], length=RSI_PERIOD)
    if rsi is None or rsi.empty:
        return None

    current_rsi = rsi.iloc[-1]
    if pd.isna(current_rsi):
        return None

    if current_rsi < RSI_OVERSOLD:
        return {
            "indicator": "RSI",
            "action": Action.BUY,
            "detail": f"RSI={current_rsi:.1f} (oversold < {RSI_OVERSOLD})",
            "value": current_rsi,
            "strength": (RSI_OVERSOLD - current_rsi) / RSI_OVERSOLD,
        }
    elif current_rsi > RSI_OVERBOUGHT:
        return {
            "indicator": "RSI",
            "action": Action.SELL,
            "detail": f"RSI={current_rsi:.1f} (overbought > {RSI_OVERBOUGHT})",
            "value": current_rsi,
            "strength": (current_rsi - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT),
        }
    return None


def check_macd(df: pd.DataFrame) -> Optional[dict]:
    """Check MACD crossover (signal line cross).

    Returns dict with signal info or None.
    """
    min_len = MACD_SLOW + MACD_SIGNAL - 1
    if len(df) < min_len:
        return None

    macd_result = ta.macd(df["close"], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    if macd_result is None or macd_result.empty:
        return None

    # Column names from pandas_ta: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    macd_col = f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"
    signal_col = f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"

    if macd_col not in macd_result.columns or signal_col not in macd_result.columns:
        return None

    macd_line = macd_result[macd_col]
    signal_line = macd_result[signal_col]

    if len(macd_line) < 2:
        return None

    curr_macd, prev_macd = macd_line.iloc[-1], macd_line.iloc[-2]
    curr_signal, prev_signal = signal_line.iloc[-1], signal_line.iloc[-2]

    if pd.isna(curr_macd) or pd.isna(prev_macd) or pd.isna(curr_signal) or pd.isna(prev_signal):
        return None

    # Bullish crossover: MACD crosses above signal
    if prev_macd <= prev_signal and curr_macd > curr_signal:
        return {
            "indicator": "MACD",
            "action": Action.BUY,
            "detail": f"MACD bullish crossover (MACD={curr_macd:.3f}, Signal={curr_signal:.3f})",
            "value": curr_macd - curr_signal,
            "strength": min(abs(curr_macd - curr_signal) / (abs(curr_signal) + 1e-9), 1.0),
        }

    # Bearish crossover: MACD crosses below signal
    if prev_macd >= prev_signal and curr_macd < curr_signal:
        return {
            "indicator": "MACD",
            "action": Action.SELL,
            "detail": f"MACD bearish crossover (MACD={curr_macd:.3f}, Signal={curr_signal:.3f})",
            "value": curr_macd - curr_signal,
            "strength": min(abs(curr_macd - curr_signal) / (abs(curr_signal) + 1e-9), 1.0),
        }

    return None


def check_ma_crossover(df: pd.DataFrame) -> Optional[dict]:
    """Check MA5 crossing MA20.

    Returns dict with signal info or None.
    """
    if len(df) < MA_SLOW + 1:
        return None

    ma_fast = ta.sma(df["close"], length=MA_FAST)
    ma_slow = ta.sma(df["close"], length=MA_SLOW)

    if ma_fast is None or ma_slow is None:
        return None
    if len(ma_fast) < 2 or len(ma_slow) < 2:
        return None

    curr_fast, prev_fast = ma_fast.iloc[-1], ma_fast.iloc[-2]
    curr_slow, prev_slow = ma_slow.iloc[-1], ma_slow.iloc[-2]

    if any(pd.isna(v) for v in [curr_fast, prev_fast, curr_slow, prev_slow]):
        return None

    # Golden cross: fast MA crosses above slow MA
    if prev_fast <= prev_slow and curr_fast > curr_slow:
        return {
            "indicator": "MA_CROSSOVER",
            "action": Action.BUY,
            "detail": f"Golden cross: MA{MA_FAST}={curr_fast:.2f} > MA{MA_SLOW}={curr_slow:.2f}",
            "value": curr_fast - curr_slow,
            "strength": min((curr_fast - curr_slow) / curr_slow, 1.0),
        }

    # Death cross: fast MA crosses below slow MA
    if prev_fast >= prev_slow and curr_fast < curr_slow:
        return {
            "indicator": "MA_CROSSOVER",
            "action": Action.SELL,
            "detail": f"Death cross: MA{MA_FAST}={curr_fast:.2f} < MA{MA_SLOW}={curr_slow:.2f}",
            "value": curr_fast - curr_slow,
            "strength": min(abs(curr_fast - curr_slow) / curr_slow, 1.0),
        }

    return None


def check_volume_spike(df: pd.DataFrame) -> Optional[dict]:
    """Check if today's volume > 2x 20-day average.

    Returns dict with signal info or None.
    """
    if len(df) < 21:
        return None

    current_volume = df["volume"].iloc[-1]
    avg_volume = df["volume"].iloc[-21:-1].mean()

    if pd.isna(current_volume) or pd.isna(avg_volume) or avg_volume == 0:
        return None

    ratio = current_volume / avg_volume

    if ratio >= VOLUME_SPIKE_MULTIPLIER:
        # Volume spike — direction from intraday candle body (open-to-close)
        price_change = df["close"].iloc[-1] - df["open"].iloc[-1]
        if price_change > 0:
            action = Action.BUY
        elif price_change < 0:
            action = Action.SELL
        else:
            return None  # Doji candle — no directional information

        return {
            "indicator": "VOLUME_SPIKE",
            "action": action,
            "detail": f"Volume spike: {current_volume:,.0f} = {ratio:.1f}x avg ({avg_volume:,.0f})",
            "value": ratio,
            "strength": min((ratio - VOLUME_SPIKE_MULTIPLIER) / VOLUME_SPIKE_MULTIPLIER, 1.0),
        }

    return None


def check_bollinger(df: pd.DataFrame) -> Optional[dict]:
    """Check if price is outside Bollinger Bands (20, 2).

    Returns dict with signal info or None.
    """
    if len(df) < BOLLINGER_PERIOD + 1:
        return None

    bbands = ta.bbands(df["close"], length=BOLLINGER_PERIOD, std=BOLLINGER_STD)
    if bbands is None or bbands.empty:
        return None

    # pandas_ta column naming varies by version (e.g. "BBL_20_2.0" vs "BBL_20_2.0_2.0")
    lower_col = next((c for c in bbands.columns if c.startswith("BBL_")), None)
    upper_col = next((c for c in bbands.columns if c.startswith("BBU_")), None)
    mid_col = next((c for c in bbands.columns if c.startswith("BBM_")), None)

    if not lower_col or not upper_col or not mid_col:
        return None

    price = df["close"].iloc[-1]
    lower = bbands[lower_col].iloc[-1]
    upper = bbands[upper_col].iloc[-1]
    mid = bbands[mid_col].iloc[-1]

    if any(pd.isna(v) for v in [price, lower, upper, mid]):
        return None

    band_width = upper - lower
    if band_width == 0:
        return None

    # Price below lower band — oversold
    if price < lower:
        return {
            "indicator": "BOLLINGER",
            "action": Action.BUY,
            "detail": f"Below lower Bollinger Band: price={price:.2f} < BB_low={lower:.2f}",
            "value": (lower - price) / band_width,
            "strength": min((lower - price) / band_width, 1.0),
        }

    # Price above upper band — overbought
    if price > upper:
        return {
            "indicator": "BOLLINGER",
            "action": Action.SELL,
            "detail": f"Above upper Bollinger Band: price={price:.2f} > BB_high={upper:.2f}",
            "value": (price - upper) / band_width,
            "strength": min((price - upper) / band_width, 1.0),
        }

    return None


def check_support_resistance(df: pd.DataFrame) -> Optional[dict]:
    """Check if price is within 2% of recent pivot points (support/resistance).

    Uses 20-day high/low as simple support/resistance levels.
    Returns dict with signal info or None.
    """
    if len(df) < 21:
        return None

    price = df["close"].iloc[-1]
    recent = df.iloc[-21:-1]  # last 20 days excluding today

    high_20d = recent["high"].max()
    low_20d = recent["low"].min()

    if pd.isna(high_20d) or pd.isna(low_20d) or price == 0:
        return None

    threshold = SUPPORT_RESISTANCE_PCT / 100.0
    if threshold <= 0:
        return None

    today_low = df["low"].iloc[-1]
    today_high = df["high"].iloc[-1]

    # Near support (within 2% of 20-day low) — price must be ABOVE support
    # and today's intraday low must not have breached it (broken support)
    if low_20d > 0 and price > low_20d and today_low >= low_20d and abs(price - low_20d) / low_20d <= threshold:
        return {
            "indicator": "SUPPORT",
            "action": Action.BUY,
            "detail": f"Near 20d support: price={price:.2f}, support={low_20d:.2f} ({abs(price - low_20d) / low_20d * 100:.1f}%)",
            "value": abs(price - low_20d) / low_20d,
            "strength": 1.0 - abs(price - low_20d) / low_20d / threshold,
        }

    # Near resistance (within 2% of 20-day high) — price must be AT or BELOW resistance
    # and today's intraday high must not have breached it (broken resistance/breakout)
    if high_20d > 0 and price <= high_20d and today_high <= high_20d and abs(price - high_20d) / high_20d <= threshold:
        return {
            "indicator": "RESISTANCE",
            "action": Action.SELL,
            "detail": f"Near 20d resistance: price={price:.2f}, resistance={high_20d:.2f} ({abs(price - high_20d) / high_20d * 100:.1f}%)",
            "value": abs(price - high_20d) / high_20d,
            "strength": 1.0 - abs(price - high_20d) / high_20d / threshold,
        }

    return None


# ---------------------------------------------------------------------------
# Scoring and ranking
# ---------------------------------------------------------------------------

ALL_CHECKS = [
    check_rsi,
    check_macd,
    check_ma_crossover,
    check_volume_spike,
    check_bollinger,
    check_support_resistance,
]

# Canonical indicator names used by each check for weight lookups.
# support_resistance can emit either "SUPPORT" or "RESISTANCE" — since they
# are mutually exclusive (one check, one signal), use the max of the two
# weights when normalizing so the total stays stable regardless of which fires.
_INDICATOR_NAMES = ["RSI", "MACD", "MA_CROSSOVER", "VOLUME_SPIKE", "BOLLINGER"]


def analyze_stock(df: pd.DataFrame) -> list[dict]:
    """Run all indicator checks on a single stock's DataFrame.

    Returns list of triggered signals (dicts from check functions).
    """
    triggered = []
    for check_fn in ALL_CHECKS:
        try:
            result = check_fn(df)
            if result is not None:
                triggered.append(result)
        except Exception as e:
            logger.debug("Check %s failed: %s", check_fn.__name__, e)
    return triggered


def score_candidate(
    triggered: list[dict],
    weights: dict[str, float] | None = None,
) -> tuple[float, Action]:
    """Score a candidate based on how many indicators triggered.

    Returns (score, dominant_action).
    Score is 0-100 based on weighted number and strength of signals.

    Args:
        triggered: List of triggered indicator dicts.
        weights: Optional dict mapping indicator name -> weight multiplier.
                 Defaults to INDICATOR_WEIGHTS from settings. Weight of 0
                 effectively disables that indicator's contribution.
    """
    if not triggered:
        return 0.0, Action.HOLD

    w = weights if weights is not None else INDICATOR_WEIGHTS

    def _weight(signal: dict) -> float:
        return w.get(signal["indicator"], 1.0)

    # Count buy vs sell signals, weighted by indicator importance
    buy_signals = [t for t in triggered if t["action"] == Action.BUY]
    sell_signals = [t for t in triggered if t["action"] == Action.SELL]

    buy_score = sum(t.get("strength", 0.5) * _weight(t) for t in buy_signals)
    sell_score = sum(t.get("strength", 0.5) * _weight(t) for t in sell_signals)

    # Dominant direction
    if buy_score > sell_score:
        dominant = Action.BUY
        direction_signals = buy_signals
        direction_score = buy_score
    elif sell_score > buy_score:
        dominant = Action.SELL
        direction_signals = sell_signals
        direction_score = sell_score
    else:
        return 0.0, Action.HOLD

    # Score: base on weighted signal count + net weighted strength
    # Opposing signals reduce the score to penalize mixed-signal stocks
    opposing_score = sell_score if dominant == Action.BUY else buy_score
    net_score = direction_score - opposing_score
    weighted_count = sum(_weight(t) for t in direction_signals)
    # Total weight = sum of weights for all possible indicators (6 checks).
    # For support/resistance (mutually exclusive outputs from one check),
    # use the max of the two weights so the total is stable.
    total_weight = sum(w.get(name, 1.0) for name in _INDICATOR_NAMES)
    total_weight += max(w.get("SUPPORT", 1.0), w.get("RESISTANCE", 1.0))
    if total_weight == 0:
        return 0.0, Action.HOLD
    raw_score = (weighted_count / total_weight) * 50 + (max(net_score, 0) / total_weight) * 50
    score = min(raw_score, 100.0)

    return score, dominant


def _build_signal(
    ticker: str,
    exchange: str,
    df: pd.DataFrame,
    triggered: list[dict],
    score: float,
    action: Action,
) -> Signal:
    """Build a Signal object from screening results."""
    if df.empty:
        raise ValueError(f"Cannot build signal for {ticker}: DataFrame is empty")
    price = df["close"].iloc[-1]
    atr = _compute_atr(df)

    if action == Action.BUY:
        stop_loss = price * (1 - DEFAULT_STOP_LOSS_PCT / 100)
        take_profit = price * (1 + DEFAULT_TAKE_PROFIT_PCT / 100)
        # Use ATR-based stops if available
        if atr and atr > 0:
            stop_loss = max(price - 2 * atr, 0.01)
            take_profit = price + 3 * atr
    else:
        stop_loss = price * (1 + DEFAULT_STOP_LOSS_PCT / 100)
        take_profit = price * (1 - DEFAULT_TAKE_PROFIT_PCT / 100)
        if atr and atr > 0:
            stop_loss = price + 2 * atr
            take_profit = max(price - 3 * atr, 0.01)

    reasoning = "; ".join(t["detail"] for t in triggered)
    indicator_values = {t["indicator"]: t["value"] for t in triggered}

    # Always compute and store MA values for trend confirmation in risk manager,
    # even if MA crossover didn't trigger (risk.check_trend_confirmation needs these).
    if len(df) >= MA_SLOW + 1:
        ma_fast = ta.sma(df["close"], length=MA_FAST)
        ma_slow = ta.sma(df["close"], length=MA_SLOW)
        if ma_fast is not None and not pd.isna(ma_fast.iloc[-1]):
            indicator_values["MA5"] = float(ma_fast.iloc[-1])
        if ma_slow is not None and not pd.isna(ma_slow.iloc[-1]):
            indicator_values["MA20"] = float(ma_slow.iloc[-1])
        # Also compute MA10 if possible
        ma_mid = ta.sma(df["close"], length=10)
        if ma_mid is not None and not pd.isna(ma_mid.iloc[-1]):
            indicator_values["MA10"] = float(ma_mid.iloc[-1])

    return Signal(
        ticker=ticker,
        action=action,
        confidence=score,
        entry_price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        reasoning=reasoning,
        source="screener",
        exchange=exchange,
        indicator_values=indicator_values,
    )


def _compute_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Compute Average True Range for stop-loss sizing."""
    if len(df) < period + 1:
        return None
    atr = ta.atr(df["high"], df["low"], df["close"], length=period)
    if atr is None or atr.empty:
        return None
    val = atr.iloc[-1]
    return val if not pd.isna(val) else None


# ---------------------------------------------------------------------------
# Extension guard — drop tickers that are already too extended above MA20
# ---------------------------------------------------------------------------

def _is_extended(df: pd.DataFrame, max_pct: float = MAX_EXTENSION_OVER_MA20_PCT) -> bool:
    """True if the latest close sits more than `max_pct`% above MA20.

    Blocks the "chase the top of a parabolic move" pattern (e.g. XNDU ripping
    from $9 → $32 in a few days). At that point every BUY indicator fires and
    the AI analyst can flip an otherwise-SELL candidate into a BUY, so the
    only robust fix is to drop the ticker from the candidate pool before it
    reaches the AI layer.

    Returns False when MA20 is unavailable (too-short DataFrame) or non-positive.
    """
    if max_pct <= 0 or len(df) < MA_SLOW + 1:
        return False
    ma20 = ta.sma(df["close"], length=MA_SLOW)
    if ma20 is None or ma20.empty:
        return False
    ma20_last = ma20.iloc[-1]
    close_last = df["close"].iloc[-1]
    if pd.isna(ma20_last) or pd.isna(close_last) or ma20_last <= 0:
        return False
    extension_pct = ((close_last - ma20_last) / ma20_last) * 100
    return extension_pct > max_pct


# ---------------------------------------------------------------------------
# Main screening entry point (pure function)
# ---------------------------------------------------------------------------

def screen_stocks(
    stock_data: dict[str, tuple[str, pd.DataFrame]],
    min_score: float = 15.0,
    indicator_weights: dict[str, float] | None = None,
    max_extension_pct: float = MAX_EXTENSION_OVER_MA20_PCT,
) -> list[Signal]:
    """Screen multiple stocks and return all candidates above min_score.

    This is a PURE FUNCTION — it does not fetch any data.
    The caller provides all data so the backtester can reuse this.

    Args:
        stock_data: Dict mapping ticker -> (exchange, ohlcv_dataframe).
        min_score: Minimum screener score to include (0-100).
        indicator_weights: Optional dict of indicator name -> weight.
                           Defaults to INDICATOR_WEIGHTS from settings.
        max_extension_pct: Drop tickers whose close is more than this % above
                           MA20. Prevents chasing parabolic breakouts. Set to
                           0 or negative to disable.

    Returns:
        List of Signal objects sorted by score descending.
    """
    candidates: list[tuple[float, Signal]] = []
    extended_skipped = 0

    for ticker, (exchange, df) in stock_data.items():
        if df.empty or len(df) < MA_SLOW + 1:
            continue

        try:
            if _is_extended(df, max_pct=max_extension_pct):
                extended_skipped += 1
                logger.debug(
                    "Skipping %s: close $%.2f is >%.0f%% above MA20 (parabolic extension)",
                    ticker, df["close"].iloc[-1], max_extension_pct,
                )
                continue

            triggered = analyze_stock(df)
            if not triggered:
                continue

            score, action = score_candidate(triggered, weights=indicator_weights)
            if score < min_score or action == Action.HOLD:
                continue

            signal = _build_signal(ticker, exchange, df, triggered, score, action)
            candidates.append((score, signal))

        except Exception as e:
            logger.warning("Screening failed for %s: %s", ticker, e)

    candidates.sort(key=lambda x: x[0], reverse=True)
    result = [signal for _, signal in candidates]

    logger.info(
        "Screener found %d candidates (from %d stocks, min_score=%.0f, "
        "dropped %d extended)",
        len(result), len(stock_data), min_score, extended_skipped,
    )
    return result

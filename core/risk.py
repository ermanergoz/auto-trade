"""Risk manager — every trade must pass ALL checks before execution.

All functions are pure — they accept portfolio state as input,
never fetch data themselves. This allows the backtester to reuse them.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from core.models import Signal, Position, Trade, Action
import math
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import (
    MAX_POSITION_SIZE_PCT,
    DAILY_LOSS_LIMIT_PCT,
    MAX_OPEN_POSITIONS,
    DEFAULT_STOP_LOSS_PCT,
    MAX_SECTOR_CONCENTRATION_PCT,
    ANTI_MOMENTUM_PCT,
    TREND_CONFIRMATION,
    MIN_RISK_REWARD_RATIO,
    RISK_PER_TRADE_PCT,
    ALLOW_SHORT_SELLING,
    FINANCIAL_KEYWORDS,
    DEFENSE_KEYWORDS,
    EXCLUDED_TICKERS,
    CIRCUIT_BREAKER_LOSSES,
    CIRCUIT_BREAKER_WINDOW_MIN,
    VOLATILITY_BASELINE,
    CHECK_ANALYST_CONSENSUS,
)

logger = logging.getLogger(__name__)


@dataclass
class RiskResult:
    """Result of a risk evaluation."""
    approved: bool
    reasons: list[str]
    position_size: int = 0  # recommended quantity if approved


# ---------------------------------------------------------------------------
# Individual risk checks — all pure functions
# ---------------------------------------------------------------------------

def check_position_size(
    signal: Signal,
    portfolio_value: float,
    max_pct: float = MAX_POSITION_SIZE_PCT,
) -> tuple[bool, str]:
    """Single share price must not exceed max position value.

    This is a pre-check that rejects stocks too expensive for even one
    share to fit within the position limit. The actual quantity-based
    constraint is enforced by calculate_position_size() after approval.
    """
    if portfolio_value <= 0:
        return False, "Portfolio value is zero or negative"

    max_value = portfolio_value * (max_pct / 100)

    if signal.entry_price > max_value:
        return False, (
            f"Single share ${signal.entry_price:.2f} exceeds max position "
            f"${max_value:.2f} ({max_pct}% of ${portfolio_value:.2f})"
        )
    return True, ""


def check_daily_loss_limit(
    daily_pnl: float,
    portfolio_value: float,
    limit_pct: float = DAILY_LOSS_LIMIT_PCT,
) -> tuple[bool, str]:
    """Today's P&L must not have breached the daily loss limit."""
    if portfolio_value <= 0:
        return False, "Portfolio value is zero or negative"

    limit_amount = portfolio_value * (limit_pct / 100)
    if daily_pnl < -limit_amount:
        return False, (
            f"Daily loss ${daily_pnl:.2f} exceeds limit "
            f"-${limit_amount:.2f} ({limit_pct}% of ${portfolio_value:.2f}). "
            "Trading halted for today."
        )
    return True, ""


def check_max_positions(
    signal: Signal,
    open_positions: list[Position],
    max_positions: int = MAX_OPEN_POSITIONS,
) -> tuple[bool, str]:
    """Must have fewer than MAX_OPEN_POSITIONS open.

    Exit signals (SELL on existing long, BUY on existing short) are
    always allowed — they reduce positions, not add new ones.
    """
    # Allow exits through regardless of position count
    for pos in open_positions:
        if pos.ticker == signal.ticker:
            if (signal.action == Action.SELL and pos.quantity > 0) or \
               (signal.action == Action.BUY and pos.quantity < 0):
                return True, ""

    if len(open_positions) >= max_positions:
        return False, (
            f"Max open positions reached: {len(open_positions)}/{max_positions}"
        )
    return True, ""


def check_stop_loss(signal: Signal) -> tuple[bool, str]:
    """Signal must have a stop-loss set."""
    if signal.stop_loss <= 0:
        return False, "Signal has no stop-loss set"

    # Verify stop-loss is on the correct side
    if signal.action == Action.BUY and signal.stop_loss >= signal.entry_price:
        return False, (
            f"BUY stop-loss ${signal.stop_loss:.2f} must be below "
            f"entry ${signal.entry_price:.2f}"
        )
    if signal.action == Action.SELL and signal.stop_loss <= signal.entry_price:
        return False, (
            f"SELL stop-loss ${signal.stop_loss:.2f} must be above "
            f"entry ${signal.entry_price:.2f}"
        )
    return True, ""


def check_sector_concentration(
    signal: Signal,
    open_positions: list[Position],
    portfolio_value: float,
    max_pct: float = MAX_SECTOR_CONCENTRATION_PCT,
    proposed_value: float = 0.0,
) -> tuple[bool, str]:
    """Sector exposure must not exceed MAX_SECTOR_CONCENTRATION_PCT."""
    if portfolio_value <= 0:
        return True, ""

    # Sum value of existing positions in same sector
    # (using entry_price * quantity as estimate)
    sector = (getattr(signal, "indicator_values", None) or {}).get("sector", "")
    if not sector:
        return True, ""  # Unknown sector, let it through

    sector_value = sum(
        (p.current_price or p.entry_price) * abs(p.quantity)
        for p in open_positions
        if p.sector.lower() == sector.lower()
    )

    # Include the proposed new position's value
    sector_value += proposed_value

    max_sector_value = portfolio_value * (max_pct / 100)
    if sector_value > max_sector_value:
        return False, (
            f"Sector '{sector}' exposure ${sector_value:.2f} would exceed "
            f"limit ${max_sector_value:.2f} ({max_pct}%)"
        )
    return True, ""


def check_no_duplicate(
    signal: Signal,
    open_positions: list[Position],
) -> tuple[bool, str]:
    """No existing position in the same direction for this ticker.

    A BUY is blocked if we already hold a long position (duplicate entry).
    A SELL is allowed if we hold a long position (closing the position).
    Zero-quantity positions (closed but not cleaned up) are ignored.
    """
    for pos in open_positions:
        if pos.ticker == signal.ticker:
            # Skip zero-quantity positions (closed but not yet removed from DB)
            if pos.quantity == 0:
                continue
            # SELL signal on an existing long = closing the position, allow it
            if signal.action == Action.SELL and pos.quantity > 0:
                return True, ""
            # BUY signal on an existing short = closing the position, allow it
            if signal.action == Action.BUY and pos.quantity < 0:
                return True, ""
            return False, (
                f"Already holding position in {signal.ticker} "
                f"({pos.quantity} shares @ ${pos.entry_price:.2f})"
            )
    return True, ""


def check_short_selling(
    signal: Signal,
    open_positions: list[Position],
) -> tuple[bool, str]:
    """Block sell signals for stocks not currently held (short selling)."""
    if ALLOW_SHORT_SELLING:
        return True, ""

    if signal.action != Action.SELL:
        return True, ""

    # Check if we hold this stock (skip zero-quantity stale positions)
    for pos in open_positions:
        if pos.ticker == signal.ticker and pos.quantity != 0:
            return True, ""

    return False, f"Short selling blocked for {signal.ticker} (not currently held)"


def check_excluded_sector(signal: Signal) -> tuple[bool, str]:
    """Block financial/defense sector stocks and explicitly excluded tickers.

    The universe builder already filters these, but this catches any
    that slip through (e.g., missing sector data from IBKR, backtest
    injection, or stale cache).
    """
    # Check explicitly excluded tickers first
    if signal.ticker in EXCLUDED_TICKERS:
        return False, (
            f"Excluded ticker: '{signal.ticker}' is in the exclusion list"
        )

    indicator_values = getattr(signal, "indicator_values", None) or {}
    sector = indicator_values.get("sector", "")

    if sector:
        sector_lower = sector.lower()
        for kw in FINANCIAL_KEYWORDS:
            if kw in sector_lower:
                return False, (
                    f"Excluded sector: '{sector}' (financial/lending companies are blocked)"
                )
        for kw in DEFENSE_KEYWORDS:
            if kw in sector_lower:
                return False, (
                    f"Excluded sector: '{sector}' (defense/military stocks are blocked)"
                )

    # Also check company name for defense/financial keywords — catches
    # companies with generic sector labels like "Industrials"
    company_name = indicator_values.get("company_name", "")
    if company_name:
        name_lower = company_name.lower()
        for kw in DEFENSE_KEYWORDS:
            if kw in name_lower:
                return False, (
                    f"Excluded by name: '{company_name}' (defense/military keywords detected)"
                )
        for kw in FINANCIAL_KEYWORDS:
            if kw in name_lower:
                return False, (
                    f"Excluded by name: '{company_name}' (financial keywords detected)"
                )

    return True, ""


def check_cumulative_risk(
    signal: Signal,
    open_positions: list[Position],
    portfolio_value: float,
    limit_pct: float = DAILY_LOSS_LIMIT_PCT,
    position_size: int | None = None,
) -> tuple[bool, str]:
    """Ensure total open risk across all positions stays within daily loss limit.

    If all open positions hit their stop-losses simultaneously (correlated
    market move), total loss must not exceed the daily loss limit.

    Args:
        position_size: Actual calculated position size. When provided, uses
            this instead of re-deriving from RISK_PER_TRADE_PCT, so volatility
            scaling and other adjustments are reflected accurately.
    """
    if portfolio_value <= 0:
        return False, "Portfolio value is zero or negative"

    existing_risk = sum(
        abs(p.entry_price - p.stop_loss) * abs(p.quantity)
        for p in open_positions
        if p.stop_loss > 0
    )

    stop_distance = abs(signal.entry_price - signal.stop_loss)
    if stop_distance > 0:
        if position_size is not None:
            # Use actual sized quantity (reflects volatility scaling, etc.)
            new_risk = stop_distance * position_size
        else:
            # Fallback: estimate from config (legacy callers)
            risk_per_trade = portfolio_value * (RISK_PER_TRADE_PCT / 100)
            estimated_qty = int(risk_per_trade / stop_distance)
            new_risk = stop_distance * estimated_qty
    else:
        new_risk = 0

    total_risk = existing_risk + new_risk
    max_daily_risk = portfolio_value * (limit_pct / 100)

    if total_risk > max_daily_risk:
        return False, (
            f"Cumulative risk ${total_risk:.2f} would exceed daily loss limit "
            f"${max_daily_risk:.2f} ({limit_pct}% of ${portfolio_value:.2f})"
        )
    return True, ""


def check_anti_momentum(
    signal: Signal,
    current_price: float,
    max_pct: float = ANTI_MOMENTUM_PCT,
) -> tuple[bool, str]:
    """Reject if price already moved >X% toward the signal direction.

    Prevents chasing stocks that already ran. If the current price is
    significantly above the entry (for buys) or below (for sells), the
    move was missed.
    """
    if current_price <= 0 or signal.entry_price <= 0:
        return False, (
            f"Invalid prices for anti-momentum check "
            f"(current_price={current_price}, entry_price={signal.entry_price})"
        )

    pct_move = ((current_price - signal.entry_price) / signal.entry_price) * 100

    if signal.action == Action.BUY and pct_move > max_pct:
        return False, (
            f"Anti-chase: price already up {pct_move:.1f}% from entry "
            f"${signal.entry_price:.2f} (limit {max_pct}%)"
        )
    if signal.action == Action.SELL and pct_move < -max_pct:
        return False, (
            f"Anti-chase: price already down {abs(pct_move):.1f}% from entry "
            f"${signal.entry_price:.2f} (limit {max_pct}%)"
        )
    return True, ""


def check_trend_confirmation(
    signal: Signal,
    indicator_values: dict,
    require_confirmation: bool = TREND_CONFIRMATION,
) -> tuple[bool, str]:
    """Require MA5 > MA10 > MA20 alignment for buys (inverse for sells).

    Uses indicator values passed from the screener. If MAs are not
    available, the check passes (don't block on missing data).
    """
    if not require_confirmation:
        return True, ""

    ma5 = indicator_values.get("MA5") or indicator_values.get("ma5")
    ma20 = indicator_values.get("MA20") or indicator_values.get("ma20")

    if ma5 is None or ma20 is None:
        return True, ""  # Can't check without MA data

    # Try to get MA10 if available, otherwise just check MA5 vs MA20
    ma10 = indicator_values.get("MA10") or indicator_values.get("ma10")

    if signal.action == Action.BUY:
        if ma10 is not None:
            if not (ma5 > ma10 > ma20):
                return False, (
                    f"Trend not confirmed: MA5={ma5:.2f} MA10={ma10:.2f} "
                    f"MA20={ma20:.2f} (need MA5 > MA10 > MA20 for buy)"
                )
        else:
            if not (ma5 > ma20):
                return False, (
                    f"Trend not confirmed: MA5={ma5:.2f} MA20={ma20:.2f} "
                    f"(need MA5 > MA20 for buy)"
                )

    if signal.action == Action.SELL:
        if ma10 is not None:
            if not (ma5 < ma10 < ma20):
                return False, (
                    f"Trend not confirmed: MA5={ma5:.2f} MA10={ma10:.2f} "
                    f"MA20={ma20:.2f} (need MA5 < MA10 < MA20 for sell)"
                )
        else:
            if not (ma5 < ma20):
                return False, (
                    f"Trend not confirmed: MA5={ma5:.2f} MA20={ma20:.2f} "
                    f"(need MA5 < MA20 for sell)"
                )

    return True, ""


def check_risk_reward(
    signal: Signal,
    min_ratio: float = MIN_RISK_REWARD_RATIO,
) -> tuple[bool, str]:
    """Take-profit must give at least X:1 reward/risk ratio."""
    if signal.entry_price <= 0 or signal.stop_loss <= 0 or signal.take_profit <= 0:
        return False, (
            f"Invalid prices for risk/reward check "
            f"(entry={signal.entry_price}, SL={signal.stop_loss}, TP={signal.take_profit})"
        )

    if signal.action == Action.BUY:
        risk = signal.entry_price - signal.stop_loss
        reward = signal.take_profit - signal.entry_price
    elif signal.action == Action.SELL:
        risk = signal.stop_loss - signal.entry_price
        reward = signal.entry_price - signal.take_profit
    else:
        return True, ""

    if risk <= 0:
        return False, "Risk is zero or negative (stop-loss on wrong side)"

    ratio = reward / risk
    if ratio < min_ratio:
        return False, (
            f"Risk/reward {ratio:.2f}:1 below minimum {min_ratio}:1 "
            f"(risk=${risk:.2f}, reward=${reward:.2f})"
        )
    return True, ""


def check_circuit_breaker(
    recent_trades: list[Trade],
    max_losses: int = CIRCUIT_BREAKER_LOSSES,
    window_minutes: int = CIRCUIT_BREAKER_WINDOW_MIN,
) -> tuple[bool, str]:
    """Pause trading after N consecutive losing trades within a time window.

    Catches regime changes, stale data, or systematic issues early —
    before the daily loss limit is hit.
    """
    if not recent_trades or max_losses <= 0:
        return True, ""

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)

    def _make_aware(dt: datetime) -> datetime:
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    # Filter to trades within the window, sorted newest-first
    # Guard against trades missing exit_time (e.g. backtest deserialization)
    windowed = sorted(
        (t for t in recent_trades
         if hasattr(t, "exit_time") and t.exit_time and _make_aware(t.exit_time) >= cutoff),
        key=lambda t: _make_aware(t.exit_time),
        reverse=True,
    )

    # Count consecutive losses from the most recent trade backward
    consecutive = 0
    for trade in windowed:
        if trade.pnl < 0:
            consecutive += 1
        else:
            break  # win or breakeven resets the streak

    if consecutive >= max_losses:
        return False, (
            f"Circuit breaker: {consecutive} consecutive losses in the last "
            f"{window_minutes} minutes. Trading paused — review manually."
        )
    return True, ""


def check_analyst_consensus(
    signal: Signal,
    consensus: str | None,
    enabled: bool = CHECK_ANALYST_CONSENSUS,
) -> tuple[bool, str]:
    """Block BUY when analyst consensus is sell or strong sell.

    Args:
        consensus: "strong_buy", "buy", "hold", "sell", "strong_sell", or None.
        enabled: Feature toggle (from settings.CHECK_ANALYST_CONSENSUS).

    Pass-through when: disabled, not a BUY signal, or no data available.
    """
    if not enabled or signal.action != Action.BUY or consensus is None:
        return True, ""

    if consensus in ("sell", "strong_sell"):
        return False, (
            f"Analyst consensus is '{consensus}' — blocking BUY. "
            "Analysts recommend selling this stock."
        )
    return True, ""


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def calculate_realized_volatility(
    closes: pd.Series,
    window: int = 20,
) -> Optional[float]:
    """Calculate annualized realized volatility from a close-price series.

    Uses log returns over the given window, annualized by sqrt(252).
    Returns None if the series is too short.
    """
    if len(closes) < window + 1:
        return None

    ratios = closes / closes.shift(1)
    log_returns = np.log(ratios.replace(0, np.nan)).dropna()
    if len(log_returns) < window:
        return None

    daily_vol = log_returns.iloc[-window:].std()
    if pd.isna(daily_vol):
        return None

    return float(daily_vol * math.sqrt(252))


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def calculate_position_size(
    signal: Signal,
    portfolio_value: float,
    max_pct: float = MAX_POSITION_SIZE_PCT,
    volatility: float | None = None,
) -> int:
    """Calculate number of shares based on max position size and stop-loss distance.

    Uses the smaller of:
    1. Max position size (% of portfolio)
    2. Risk-based sizing (1% risk per trade using stop-loss distance)

    When volatility is provided, scales the position inversely to the
    volatility regime: high vol → smaller position, low vol → base size
    (never increases beyond base to avoid leverage).
    """
    if signal.entry_price <= 0 or portfolio_value <= 0:
        return 0

    # Method 1: Max position size
    max_position_value = portfolio_value * (max_pct / 100)
    qty_by_size = int(max_position_value / signal.entry_price)

    # Method 2: Risk-based (risk RISK_PER_TRADE_PCT% of portfolio per trade)
    risk_per_trade = portfolio_value * (RISK_PER_TRADE_PCT / 100)
    stop_distance = abs(signal.entry_price - signal.stop_loss)
    if stop_distance > 0:
        qty_by_risk = int(risk_per_trade / stop_distance)
    else:
        qty_by_risk = qty_by_size

    # Take the smaller
    quantity = min(qty_by_size, qty_by_risk)

    # Volatility regime adjustment: scale down when vol > baseline
    if volatility is not None and volatility > 0 and VOLATILITY_BASELINE > 0:
        vol_scale = min(VOLATILITY_BASELINE / volatility, 1.0)  # cap at 1.0 (no leverage)
        quantity = max(int(quantity * vol_scale), min(1, quantity))

    return max(quantity, 0)


# ---------------------------------------------------------------------------
# Main evaluation — runs all checks
# ---------------------------------------------------------------------------

def evaluate(
    signal: Signal,
    open_positions: list[Position],
    portfolio_value: float,
    daily_pnl: float,
    current_price: float = 0.0,
    recent_trades: list[Trade] | None = None,
    volatility: float | None = None,
    analyst_consensus: str | None = None,
) -> RiskResult:
    """Run all risk checks on a signal. Returns RiskResult.

    Pure function — all state passed in as arguments.

    Args:
        volatility: Current annualized market volatility. When provided,
                    position sizes are scaled inversely (high vol → smaller).
        analyst_consensus: Analyst recommendation consensus string
                          ("buy", "sell", "hold", etc.) or None if unavailable.
    """
    reasons = []

    # Use entry_price as current_price fallback
    price = current_price if current_price > 0 else signal.entry_price

    # Get indicator values for trend check
    indicator_values = getattr(signal, "indicator_values", {}) or {}

    # Detect if this signal is closing an existing position (exit signal).
    # Exit signals must not be blocked by discipline checks (trend, anti-momentum,
    # risk/reward, analyst consensus) — those only apply to new entries. Blocking
    # an exit can trap the trader in a losing position indefinitely.
    is_exit = False
    for pos in open_positions:
        if pos.ticker == signal.ticker and pos.quantity != 0:
            # SELL on existing long = exit, BUY on existing short = exit
            if (signal.action == Action.SELL and pos.quantity > 0) or \
               (signal.action == Action.BUY and pos.quantity < 0):
                is_exit = True
                break

    # Pre-compute position size for sector concentration check
    # so we use the actual risk-sized value, not a fixed max estimate
    estimated_size = calculate_position_size(signal, portfolio_value, volatility=volatility)
    proposed_value = signal.entry_price * estimated_size if estimated_size > 0 else 0.0

    checks = [
        check_short_selling(signal, open_positions),
        check_position_size(signal, portfolio_value),
        check_daily_loss_limit(daily_pnl, portfolio_value),
        check_cumulative_risk(signal, open_positions, portfolio_value,
                              position_size=estimated_size),
        check_max_positions(signal, open_positions),
        check_stop_loss(signal),
        check_sector_concentration(signal, open_positions, portfolio_value,
                                    proposed_value=proposed_value),
        check_no_duplicate(signal, open_positions),
        check_excluded_sector(signal),
        check_circuit_breaker(recent_trades or []),
    ]

    # Discipline checks only apply to new entries, not exits.
    # Blocking an exit with trend/momentum/R:R checks would prevent
    # closing losing positions when the market moves against us.
    if not is_exit:
        checks.extend([
            check_risk_reward(signal),
            check_anti_momentum(signal, price),
            check_trend_confirmation(signal, indicator_values),
            check_analyst_consensus(signal, analyst_consensus),
        ])

    for passed, reason in checks:
        if not passed:
            reasons.append(reason)

    approved = len(reasons) == 0

    position_size = 0
    if approved:
        position_size = estimated_size
        if position_size <= 0:
            approved = False
            reasons.append("Calculated position size is 0 (portfolio too small or price too high)")

    if approved:
        logger.info(
            "APPROVED: %s %s %d shares @ $%.2f (SL: $%.2f, TP: $%.2f)",
            signal.action.value.upper(), signal.ticker, position_size,
            signal.entry_price, signal.stop_loss, signal.take_profit,
        )
    else:
        logger.info(
            "REJECTED: %s %s — %s",
            signal.action.value.upper(), signal.ticker, "; ".join(reasons),
        )

    return RiskResult(
        approved=approved,
        reasons=reasons,
        position_size=position_size,
    )

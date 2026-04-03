"""Risk manager — every trade must pass ALL checks before execution.

All functions are pure — they accept portfolio state as input,
never fetch data themselves. This allows the backtester to reuse them.
"""

import logging
from dataclasses import dataclass

from core.models import Signal, Position, Action
from config.settings import (
    MAX_POSITION_SIZE_PCT,
    DAILY_LOSS_LIMIT_PCT,
    MAX_OPEN_POSITIONS,
    DEFAULT_STOP_LOSS_PCT,
    MAX_SECTOR_CONCENTRATION_PCT,
    ANTI_MOMENTUM_PCT,
    TREND_CONFIRMATION,
    MIN_RISK_REWARD_RATIO,
    ALLOW_SHORT_SELLING,
    FINANCIAL_KEYWORDS,
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
    open_positions: list[Position],
    max_positions: int = MAX_OPEN_POSITIONS,
) -> tuple[bool, str]:
    """Must have fewer than MAX_OPEN_POSITIONS open."""
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
) -> tuple[bool, str]:
    """Sector exposure must not exceed MAX_SECTOR_CONCENTRATION_PCT."""
    if not signal.exchange or portfolio_value <= 0:
        return True, ""  # Can't check without sector info

    # Sum value of existing positions in same sector
    # (using entry_price * quantity as estimate)
    sector = getattr(signal, "indicator_values", {}).get("sector", "")
    if not sector:
        return True, ""  # Unknown sector, let it through

    sector_value = sum(
        p.entry_price * p.quantity
        for p in open_positions
        if p.sector.lower() == sector.lower()
    )

    max_sector_value = portfolio_value * (max_pct / 100)
    if sector_value >= max_sector_value:
        return False, (
            f"Sector '{sector}' exposure ${sector_value:.2f} would exceed "
            f"limit ${max_sector_value:.2f} ({max_pct}%)"
        )
    return True, ""


def check_no_duplicate(
    signal: Signal,
    open_positions: list[Position],
) -> tuple[bool, str]:
    """No existing position in this ticker."""
    for pos in open_positions:
        if pos.ticker == signal.ticker:
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

    # Check if we hold this stock
    for pos in open_positions:
        if pos.ticker == signal.ticker:
            return True, ""

    return False, f"Short selling blocked for {signal.ticker} (not currently held)"


def check_excluded_sector(signal: Signal) -> tuple[bool, str]:
    """Block financial sector stocks as a safety net.

    The universe builder already filters these, but this catches any
    that slip through (e.g., missing sector data from IBKR).
    """
    sector = getattr(signal, "indicator_values", {}).get("sector", "")
    if not sector:
        return True, ""

    sector_lower = sector.lower()
    for kw in FINANCIAL_KEYWORDS:
        if kw in sector_lower:
            return False, (
                f"Excluded sector: '{sector}' (financial/lending companies are blocked)"
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
        return True, ""

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
        return True, ""  # Can't check without prices

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


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def calculate_position_size(
    signal: Signal,
    portfolio_value: float,
    max_pct: float = MAX_POSITION_SIZE_PCT,
) -> int:
    """Calculate number of shares based on max position size and stop-loss distance.

    Uses the smaller of:
    1. Max position size (% of portfolio)
    2. Risk-based sizing (2% risk per trade using stop-loss distance)
    """
    if signal.entry_price <= 0 or portfolio_value <= 0:
        return 0

    # Method 1: Max position size
    max_position_value = portfolio_value * (max_pct / 100)
    qty_by_size = int(max_position_value / signal.entry_price)

    # Method 2: Risk-based (risk 1% of portfolio per trade)
    risk_per_trade = portfolio_value * 0.01
    stop_distance = abs(signal.entry_price - signal.stop_loss)
    if stop_distance > 0:
        qty_by_risk = int(risk_per_trade / stop_distance)
    else:
        qty_by_risk = qty_by_size

    # Take the smaller
    quantity = min(qty_by_size, qty_by_risk)
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
) -> RiskResult:
    """Run all risk checks on a signal. Returns RiskResult.

    Pure function — all state passed in as arguments.
    """
    reasons = []

    # Use entry_price as current_price fallback
    price = current_price if current_price > 0 else signal.entry_price

    # Get indicator values for trend check
    indicator_values = getattr(signal, "indicator_values", {}) or {}

    checks = [
        check_short_selling(signal, open_positions),
        check_position_size(signal, portfolio_value),
        check_daily_loss_limit(daily_pnl, portfolio_value),
        check_max_positions(open_positions),
        check_stop_loss(signal),
        check_sector_concentration(signal, open_positions, portfolio_value),
        check_no_duplicate(signal, open_positions),
        # Discipline rules
        check_excluded_sector(signal),
        check_risk_reward(signal),
        check_anti_momentum(signal, price),
        check_trend_confirmation(signal, indicator_values),
    ]

    for passed, reason in checks:
        if not passed:
            reasons.append(reason)

    approved = len(reasons) == 0

    position_size = 0
    if approved:
        position_size = calculate_position_size(signal, portfolio_value)
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

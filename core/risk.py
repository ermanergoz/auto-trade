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
    """Proposed position must be <= MAX_POSITION_SIZE_PCT of portfolio."""
    if portfolio_value <= 0:
        return False, "Portfolio value is zero or negative"

    max_value = portfolio_value * (max_pct / 100)
    position_value = signal.entry_price  # per share; actual check uses calculated qty

    if position_value > max_value:
        return False, (
            f"Single share ${position_value:.2f} exceeds max position "
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
) -> RiskResult:
    """Run all risk checks on a signal. Returns RiskResult.

    Pure function — all state passed in as arguments.
    """
    reasons = []

    checks = [
        check_position_size(signal, portfolio_value),
        check_daily_loss_limit(daily_pnl, portfolio_value),
        check_max_positions(open_positions),
        check_stop_loss(signal),
        check_sector_concentration(signal, open_positions, portfolio_value),
        check_no_duplicate(signal, open_positions),
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

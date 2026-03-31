"""Structured logging, CSV trade journal, and rich terminal dashboard."""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

from config.settings import LOG_DIR, TIMEZONE
from core.models import Position, Trade, DailySummary, Signal

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# CSV Trade Journal
# ---------------------------------------------------------------------------

_CSV_HEADERS = [
    "timestamp", "ticker", "exchange", "action", "quantity",
    "entry_price", "exit_price", "pnl", "pnl_pct",
    "trade_type", "sector", "reasoning", "duration_hours",
]


def _trades_csv_path() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    return LOG_DIR / f"trades_{date_str}.csv"


def log_trade_to_csv(trade: Trade) -> None:
    """Append a completed trade to the daily CSV journal."""
    path = _trades_csv_path()
    write_header = not path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_HEADERS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": trade.exit_time.isoformat(),
            "ticker": trade.ticker,
            "exchange": trade.exchange,
            "action": "BUY",  # entry action
            "quantity": trade.quantity,
            "entry_price": f"{trade.entry_price:.2f}",
            "exit_price": f"{trade.exit_price:.2f}",
            "pnl": f"{trade.pnl:.2f}",
            "pnl_pct": f"{trade.pnl_pct:.2f}",
            "trade_type": trade.trade_type.value,
            "sector": trade.sector,
            "reasoning": trade.reasoning[:200],
            "duration_hours": f"{trade.duration:.1f}",
        })

    logger.info("Trade logged to %s", path)


# ---------------------------------------------------------------------------
# Rich Terminal Dashboard
# ---------------------------------------------------------------------------

def display_positions(positions: list[Position]) -> None:
    """Display open positions as a rich table."""
    table = Table(title="Open Positions", show_lines=True)
    table.add_column("Ticker", style="cyan bold")
    table.add_column("Exchange", style="dim")
    table.add_column("Qty", justify="right")
    table.add_column("Entry", justify="right", style="yellow")
    table.add_column("Current", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("P&L %", justify="right")
    table.add_column("SL", justify="right", style="red")
    table.add_column("TP", justify="right", style="green")
    table.add_column("Type", style="dim")

    for pos in positions:
        pnl = pos.unrealized_pnl
        pnl_pct = pos.unrealized_pnl_pct
        pnl_str = f"${pnl:+,.2f}" if pnl is not None else "N/A"
        pnl_pct_str = f"{pnl_pct:+.1f}%" if pnl_pct is not None else "N/A"
        pnl_style = "green" if (pnl and pnl >= 0) else "red"
        current = f"${pos.current_price:.2f}" if pos.current_price else "N/A"

        table.add_row(
            pos.ticker,
            pos.exchange,
            str(pos.quantity),
            f"${pos.entry_price:.2f}",
            current,
            Text(pnl_str, style=pnl_style),
            Text(pnl_pct_str, style=pnl_style),
            f"${pos.stop_loss:.2f}",
            f"${pos.take_profit:.2f}",
            pos.trade_type.value,
        )

    if not positions:
        table.add_row("—", "—", "—", "—", "—", "—", "—", "—", "—", "—")

    console.print(table)


def display_recent_trades(trades: list[Trade], limit: int = 10) -> None:
    """Display recent completed trades."""
    table = Table(title=f"Recent Trades (last {limit})", show_lines=True)
    table.add_column("Time", style="dim")
    table.add_column("Ticker", style="cyan bold")
    table.add_column("Qty", justify="right")
    table.add_column("Entry", justify="right")
    table.add_column("Exit", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("P&L %", justify="right")
    table.add_column("Duration", justify="right")

    for trade in trades[:limit]:
        pnl_style = "green" if trade.pnl >= 0 else "red"
        table.add_row(
            trade.exit_time.strftime("%H:%M"),
            trade.ticker,
            str(trade.quantity),
            f"${trade.entry_price:.2f}",
            f"${trade.exit_price:.2f}",
            Text(f"${trade.pnl:+,.2f}", style=pnl_style),
            Text(f"{trade.pnl_pct:+.1f}%", style=pnl_style),
            f"{trade.duration:.1f}h",
        )

    if not trades:
        table.add_row("—", "—", "—", "—", "—", "—", "—", "—")

    console.print(table)


def display_portfolio_summary(
    portfolio_value: float,
    daily_pnl: float,
    positions_count: int,
    trades_today: int,
) -> None:
    """Display portfolio summary panel."""
    daily_pnl_pct = (daily_pnl / portfolio_value * 100) if portfolio_value > 0 else 0
    pnl_style = "green" if daily_pnl >= 0 else "red"

    summary = (
        f"[bold]Portfolio Value:[/bold] ${portfolio_value:,.2f}\n"
        f"[bold]Daily P&L:[/bold] [{pnl_style}]${daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%)[/{pnl_style}]\n"
        f"[bold]Open Positions:[/bold] {positions_count}\n"
        f"[bold]Trades Today:[/bold] {trades_today}"
    )
    console.print(Panel(summary, title="Portfolio Summary", border_style="blue"))


def display_scan_results(candidates: list[Signal]) -> None:
    """Display screener candidates."""
    table = Table(title="Screener Candidates", show_lines=True)
    table.add_column("Ticker", style="cyan bold")
    table.add_column("Action", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Source", style="dim")
    table.add_column("Reasoning")

    for sig in candidates:
        action_style = "green" if sig.action.value == "buy" else "red"
        table.add_row(
            sig.ticker,
            Text(sig.action.value.upper(), style=action_style),
            f"{sig.confidence:.0f}",
            f"${sig.entry_price:.2f}",
            sig.source,
            sig.reasoning[:60] + "..." if len(sig.reasoning) > 60 else sig.reasoning,
        )

    console.print(table)


def display_full_dashboard(
    portfolio_value: float,
    daily_pnl: float,
    positions: list[Position],
    recent_trades: list[Trade],
    candidates: Optional[list[Signal]] = None,
) -> None:
    """Display the complete terminal dashboard."""
    console.clear()
    now = datetime.now()
    console.print(
        f"[bold blue]Auto-Trader Dashboard[/bold blue] | "
        f"{now.strftime('%Y-%m-%d %H:%M:%S')}",
        justify="center",
    )
    console.print()

    display_portfolio_summary(
        portfolio_value, daily_pnl,
        len(positions), len(recent_trades),
    )
    console.print()

    display_positions(positions)
    console.print()

    display_recent_trades(recent_trades)

    if candidates:
        console.print()
        display_scan_results(candidates)

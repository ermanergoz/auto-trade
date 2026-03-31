"""Backtest performance reporting and metrics."""

import json
import logging
import math
from datetime import date
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from config.settings import RISK_FREE_RATE
from core.models import Trade

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Metrics calculation
# ---------------------------------------------------------------------------

def calculate_metrics(
    trades: list[Trade],
    equity_curve: list[tuple[date, float]],
    initial_capital: float,
) -> dict:
    """Calculate comprehensive backtest performance metrics."""
    if not trades:
        return {
            "total_return_pct": 0, "annualized_return_pct": 0,
            "sharpe_ratio": 0, "max_drawdown_pct": 0,
            "win_rate_pct": 0, "profit_factor": 0,
            "avg_trade_duration_hours": 0, "num_trades": 0,
            "total_pnl": 0, "avg_pnl_per_trade": 0,
            "best_trade_pnl": 0, "worst_trade_pnl": 0,
            "final_value": initial_capital,
        }

    # Basic PnL
    pnls = [t.pnl for t in trades]
    total_pnl = sum(pnls)
    final_value = equity_curve[-1][1] if equity_curve else initial_capital + total_pnl

    total_return = (final_value / initial_capital - 1) * 100

    # Annualized return
    if equity_curve and len(equity_curve) > 1:
        days = (equity_curve[-1][0] - equity_curve[0][0]).days
        if days > 0:
            annualized = ((final_value / initial_capital) ** (365 / days) - 1) * 100
        else:
            annualized = 0
    else:
        annualized = 0

    # Win rate
    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl <= 0]
    win_rate = len(winners) / len(trades) * 100

    # Profit factor
    gross_profit = sum(t.pnl for t in winners) if winners else 0
    gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Sharpe ratio (daily returns)
    sharpe = _calculate_sharpe(equity_curve, RISK_FREE_RATE)

    # Max drawdown
    max_dd = _calculate_max_drawdown(equity_curve)

    # Average trade duration
    durations = [t.duration for t in trades]
    avg_duration = sum(durations) / len(durations) if durations else 0

    return {
        "total_return_pct": round(total_return, 2),
        "annualized_return_pct": round(annualized, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "avg_trade_duration_hours": round(avg_duration, 1),
        "num_trades": len(trades),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl_per_trade": round(total_pnl / len(trades), 2),
        "best_trade_pnl": round(max(pnls), 2),
        "worst_trade_pnl": round(min(pnls), 2),
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "final_value": round(final_value, 2),
        "initial_capital": initial_capital,
    }


def _calculate_sharpe(
    equity_curve: list[tuple[date, float]],
    risk_free_rate: float,
) -> float:
    """Calculate annualized Sharpe ratio from daily equity curve."""
    if len(equity_curve) < 2:
        return 0.0

    values = [v for _, v in equity_curve]
    daily_returns = [
        (values[i] - values[i - 1]) / values[i - 1]
        for i in range(1, len(values))
        if values[i - 1] != 0
    ]

    if not daily_returns:
        return 0.0

    avg_return = sum(daily_returns) / len(daily_returns)
    daily_rf = risk_free_rate / 252

    excess = avg_return - daily_rf
    std = _std(daily_returns)

    if std == 0:
        return 0.0

    return (excess / std) * math.sqrt(252)


def _calculate_max_drawdown(equity_curve: list[tuple[date, float]]) -> float:
    """Calculate maximum drawdown percentage (peak to trough)."""
    if not equity_curve:
        return 0.0

    values = [v for _, v in equity_curve]
    peak = values[0]
    max_dd = 0.0

    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    return max_dd


def _std(values: list[float]) -> float:
    """Standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_metrics(metrics: dict) -> None:
    """Display backtest metrics as a rich table."""
    pnl_style = "green" if metrics["total_pnl"] >= 0 else "red"

    table = Table(title="Backtest Results", show_lines=True)
    table.add_column("Metric", style="cyan bold")
    table.add_column("Value", justify="right")

    rows = [
        ("Initial Capital", f"${metrics['initial_capital']:,.2f}"),
        ("Final Value", f"${metrics['final_value']:,.2f}"),
        ("Total P&L", Text(f"${metrics['total_pnl']:+,.2f}", style=pnl_style)),
        ("Total Return", Text(f"{metrics['total_return_pct']:+.2f}%", style=pnl_style)),
        ("Annualized Return", f"{metrics['annualized_return_pct']:+.2f}%"),
        ("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}"),
        ("Max Drawdown", Text(f"-{metrics['max_drawdown_pct']:.2f}%", style="red")),
        ("", ""),
        ("Total Trades", str(metrics["num_trades"])),
        ("Win Rate", f"{metrics['win_rate_pct']:.1f}%"),
        ("Profit Factor", f"{metrics['profit_factor']:.2f}"),
        ("Avg P&L/Trade", f"${metrics['avg_pnl_per_trade']:+,.2f}"),
        ("Best Trade", Text(f"${metrics['best_trade_pnl']:+,.2f}", style="green")),
        ("Worst Trade", Text(f"${metrics['worst_trade_pnl']:+,.2f}", style="red")),
        ("Avg Duration", f"{metrics['avg_trade_duration_hours']:.1f}h"),
        ("", ""),
        ("Winning Trades", str(metrics.get("winning_trades", 0))),
        ("Losing Trades", str(metrics.get("losing_trades", 0))),
        ("Gross Profit", f"${metrics.get('gross_profit', 0):,.2f}"),
        ("Gross Loss", f"${metrics.get('gross_loss', 0):,.2f}"),
    ]

    for label, value in rows:
        if label == "":
            table.add_row("─" * 20, "─" * 15)
        else:
            table.add_row(label, value if isinstance(value, Text) else str(value))

    console.print(table)


def compare_configs(results_list: list[tuple[str, dict]]) -> None:
    """Side-by-side comparison of multiple backtest runs."""
    table = Table(title="Backtest Comparison", show_lines=True)
    table.add_column("Metric", style="cyan bold")

    for name, _ in results_list:
        table.add_column(name, justify="right")

    key_metrics = [
        ("Total Return", "total_return_pct", ".2f", "%"),
        ("Sharpe Ratio", "sharpe_ratio", ".2f", ""),
        ("Max Drawdown", "max_drawdown_pct", ".2f", "%"),
        ("Win Rate", "win_rate_pct", ".1f", "%"),
        ("Profit Factor", "profit_factor", ".2f", ""),
        ("Trades", "num_trades", "d", ""),
        ("Avg P&L/Trade", "avg_pnl_per_trade", "+,.2f", "$"),
    ]

    for label, key, fmt, prefix in key_metrics:
        row = [label]
        for _, metrics in results_list:
            val = metrics.get(key, 0)
            if prefix == "$":
                row.append(f"${val:{fmt}}")
            elif prefix == "%":
                row.append(f"{val:{fmt}}%")
            else:
                row.append(f"{val:{fmt}}")
        table.add_row(*row)

    console.print(table)


def export_metrics_json(metrics: dict, filepath: str) -> None:
    """Export metrics to JSON file."""
    import json
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Metrics exported to %s", filepath)

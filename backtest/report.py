"""Backtest performance reporting and metrics."""

import json
import logging
import math
from datetime import date
from typing import Optional

import numpy as np
from scipy import stats

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from config.settings import (
    RISK_FREE_RATE,
    BACKTEST_SLIPPAGE_PCT, BACKTEST_SPREAD_BPS,
    BACKTEST_COMMISSION, BACKTEST_COMMISSION_PER_SHARE,
)
from core.models import Trade

logger = logging.getLogger(__name__)
console = Console()

# Survivorship-bias caveat printed on EVERY backtest report. yfinance returns
# only tickers that still exist today, so delisted/acquired names are silently
# absent and absolute returns are inflated — judge edge by alpha vs SPY.
SURVIVORSHIP_CAVEAT = (
    "Universe is survivorship-biased: yfinance returns only tickers that exist "
    "today; delisted names are silently absent — judge edge by alpha vs SPY, "
    "not absolute return. The universe is a fixed point-in-time snapshot."
)


# ---------------------------------------------------------------------------
# Cost decomposition
# ---------------------------------------------------------------------------

def _commission_for(quantity: int, commission: float, commission_per_share: float) -> float:
    """Replicate SimulatedPortfolio._commission_for for the report side.

    IBKR-style commission: max(min, per_share * qty). Mirrors the engine's
    __post_init__ rule — disabling the minimum also zeroes the per-share fee.
    """
    if commission <= 0:
        commission_per_share = 0.0
    if commission <= 0 and commission_per_share <= 0:
        return 0.0
    per_share = abs(quantity) * max(commission_per_share, 0.0)
    return max(commission, per_share)


def _decompose_trade_costs(
    trade: Trade,
    slippage_pct: float,
    spread_bps: float,
    commission: float,
    commission_per_share: float,
) -> dict:
    """Decompose a single round trip into gross/total-cost/net per the EXACT
    engine fill mechanics (see core/models.py:91-93 and backtest/engine.py).

    Trade.pnl = (exit_price - entry_price) * quantity, where entry_price and
    exit_price are the slippage- AND spread-adjusted FILL prices. So Trade.pnl
    already embeds slippage+spread but NOT commission (commission only touches
    cash). We reconstruct the pre-friction raw prices from the stored fills,
    then:
        gross_pnl  = Trade.pnl + slippage + spread   (add back embedded frictions)
        total_cost = commissions + slippage + spread
        net_pnl    = Trade.pnl - commissions          (only commission is left)
    """
    q = trade.quantity
    absq = abs(q)
    slip_frac = slippage_pct / 100.0
    spread_frac = spread_bps / 10_000.0
    frac = slip_frac + spread_frac

    # Invert the engine's per-leg adjustment to recover raw (pre-friction) prices.
    if q > 0:  # long: entry crossed up, exit crossed down
        raw_entry = trade.entry_price / (1 + frac)
        raw_exit = trade.exit_price / (1 - frac)
    else:  # short: entry crossed down, exit crossed up
        raw_entry = trade.entry_price / (1 - frac)
        raw_exit = trade.exit_price / (1 + frac)

    slippage_cost = (raw_entry + raw_exit) * slip_frac * absq
    spread_cost = (raw_entry + raw_exit) * spread_frac * absq

    commission_entry = _commission_for(q, commission, commission_per_share)
    commission_exit = _commission_for(q, commission, commission_per_share)
    commissions = commission_entry + commission_exit

    gross_pnl = trade.pnl + slippage_cost + spread_cost
    total_cost = commissions + slippage_cost + spread_cost
    net_pnl = trade.pnl - commissions

    return {
        "gross_pnl": gross_pnl,
        "slippage_cost": slippage_cost,
        "spread_cost": spread_cost,
        "commission_cost": commissions,
        "total_cost": total_cost,
        "net_pnl": net_pnl,
    }


# ---------------------------------------------------------------------------
# Metrics calculation
# ---------------------------------------------------------------------------

def calculate_metrics(
    trades: list[Trade],
    equity_curve: list[tuple[date, float]],
    initial_capital: float,
    slippage_pct: float = BACKTEST_SLIPPAGE_PCT,
    spread_bps: float = BACKTEST_SPREAD_BPS,
    commission: float = BACKTEST_COMMISSION,
    commission_per_share: float = BACKTEST_COMMISSION_PER_SHARE,
    benchmark_curve: Optional[list[tuple[date, float]]] = None,
) -> dict:
    """Calculate comprehensive backtest performance metrics.

    Cost params (slippage_pct, spread_bps, commission, commission_per_share)
    default to the BACKTEST_* settings — the same values run_backtest uses — so
    the gross/total-cost/net decomposition reconstructs friction from the stored
    fill prices, not by re-deriving from cash. Pass the portfolio's actual
    params when they differ from the defaults.

    When ``benchmark_curve`` (the SPY buy-and-hold curve aligned to the
    strategy's warmup-trimmed window) is supplied, the result also carries
    ``benchmark_total_return``, ``benchmark_sharpe``, and risk-free-adjusted
    CAPM ``alpha`` / ``beta`` of the strategy vs the benchmark.
    """
    # CAPM / benchmark block — folded into whichever result we return below.
    benchmark_metrics = (
        calculate_capm_metrics(equity_curve, benchmark_curve)
        if benchmark_curve
        else {}
    )

    if not trades:
        return {
            "total_return_pct": 0, "annualized_return_pct": 0,
            "sharpe_ratio": 0, "max_drawdown_pct": 0,
            "win_rate_pct": 0, "profit_factor": 0,
            "avg_trade_duration_hours": 0, "num_trades": 0,
            "total_pnl": 0, "avg_pnl_per_trade": 0,
            "best_trade_pnl": 0, "worst_trade_pnl": 0,
            "gross_pnl": 0, "net_pnl": 0, "total_cost": 0,
            "cost_pct_of_gross_pnl": 0, "breakeven_edge_per_trade": 0,
            "final_value": initial_capital,
            **benchmark_metrics,
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

    # Cost decomposition — aggregate the exact gross/total-cost/net split over
    # every round trip. gross adds back the slippage+spread embedded in fills;
    # net subtracts only commission (slippage/spread are already in Trade.pnl).
    total_gross_pnl = 0.0
    total_cost = 0.0
    total_net_pnl = 0.0
    for t in trades:
        d = _decompose_trade_costs(
            t, slippage_pct, spread_bps, commission, commission_per_share,
        )
        total_gross_pnl += d["gross_pnl"]
        total_cost += d["total_cost"]
        total_net_pnl += d["net_pnl"]

    # Cost as % of gross P&L (guard gross == 0). The fraction of the strategy's
    # gross edge eaten by friction — the headline realism check.
    if total_gross_pnl != 0:
        cost_pct_of_gross = total_cost / total_gross_pnl * 100
    else:
        cost_pct_of_gross = 0.0

    # Breakeven edge per trade: the round-trip friction (in $) each trade must
    # clear just to break even. Same currency basis as the cost columns.
    breakeven_edge_per_trade = total_cost / len(trades) if trades else 0.0

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
        "gross_pnl": round(total_gross_pnl, 2),
        "net_pnl": round(total_net_pnl, 2),
        "total_cost": round(total_cost, 2),
        "cost_pct_of_gross_pnl": round(cost_pct_of_gross, 2),
        "breakeven_edge_per_trade": round(breakeven_edge_per_trade, 2),
        "final_value": round(final_value, 2),
        "initial_capital": initial_capital,
        **benchmark_metrics,
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
# CAPM benchmark (SPY) — alpha / beta on risk-free-adjusted excess returns
# ---------------------------------------------------------------------------

def _daily_returns_from_curve(
    curve: list[tuple[date, float]],
) -> list[float]:
    """Simple daily returns from an equity curve (skips zero-value denominators)."""
    values = [v for _, v in curve]
    return [
        (values[i] - values[i - 1]) / values[i - 1]
        for i in range(1, len(values))
        if values[i - 1] != 0
    ]


def calculate_capm_metrics(
    strategy_curve: list[tuple[date, float]],
    benchmark_curve: list[tuple[date, float]],
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict:
    """CAPM alpha/beta of the strategy vs the benchmark (SPY).

    The regression is run on RISK-FREE-ADJUSTED EXCESS daily returns — the
    risk-free rate is subtracted EXPLICITLY (per STACK.md / standard CAPM), so
    a strategy that merely matches the benchmark shows alpha ~= 0 / beta ~= 1
    rather than fabricating alpha by omitting the risk-free term:

        excess_strat = strategy_daily_returns - RISK_FREE_RATE/252
        excess_bench = benchmark_daily_returns - RISK_FREE_RATE/252
        slope, intercept = linregress(excess_bench, excess_strat)
        beta            = slope
        annualized_alpha = intercept * 252

    Both curves must be sampled on the SAME trading days (the engine aligns the
    benchmark curve to the strategy's warmup-trimmed window) so the excess
    return series line up bar-for-bar. Returns zeros when there is too little
    data or the benchmark has degenerate (zero) variance.
    """
    result = {
        "benchmark_total_return": 0.0,
        "benchmark_sharpe": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
    }
    if not benchmark_curve or len(benchmark_curve) < 2:
        return result

    bench_values = [v for _, v in benchmark_curve]
    if bench_values[0]:
        result["benchmark_total_return"] = round(
            (bench_values[-1] / bench_values[0] - 1) * 100, 2
        )
    result["benchmark_sharpe"] = round(
        _calculate_sharpe(benchmark_curve, risk_free_rate), 2
    )

    strat_r = _daily_returns_from_curve(strategy_curve)
    bench_r = _daily_returns_from_curve(benchmark_curve)
    n = min(len(strat_r), len(bench_r))
    if n < 2:
        return result

    daily_rf = risk_free_rate / 252
    excess_strat = np.asarray(strat_r[:n], dtype=float) - daily_rf
    excess_bench = np.asarray(bench_r[:n], dtype=float) - daily_rf

    # A flat benchmark (zero variance) makes the regression undefined — leave
    # alpha/beta at zero rather than emitting NaN/inf.
    if np.std(excess_bench) == 0:
        return result

    slope, intercept = stats.linregress(excess_bench, excess_strat)[:2]
    result["beta"] = round(float(slope), 4)
    result["alpha"] = round(float(intercept) * 252, 4)
    return result


def benchmark_column_metrics(
    benchmark_curve: list[tuple[date, float]],
    initial_capital: float,
) -> dict:
    """A metrics-shaped dict for rendering the SPY column in compare_configs.

    The benchmark has no trades, so trade-based fields are zeroed; total return,
    Sharpe and max drawdown come straight off the buy-and-hold curve. Beta vs
    itself is 1.0 and alpha 0.0 by construction.
    """
    if not benchmark_curve or len(benchmark_curve) < 2:
        return {
            "total_return_pct": 0.0, "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0, "win_rate_pct": 0.0,
            "profit_factor": 0.0, "num_trades": 0,
            "avg_pnl_per_trade": 0.0, "benchmark_total_return": 0.0,
            "alpha": 0.0, "beta": 0.0,
            "final_value": initial_capital, "initial_capital": initial_capital,
        }

    values = [v for _, v in benchmark_curve]
    total_return = (values[-1] / values[0] - 1) * 100 if values[0] else 0.0
    return {
        "total_return_pct": round(total_return, 2),
        "sharpe_ratio": round(_calculate_sharpe(benchmark_curve, RISK_FREE_RATE), 2),
        "max_drawdown_pct": round(_calculate_max_drawdown(benchmark_curve), 2),
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "num_trades": 0,
        "avg_pnl_per_trade": 0.0,
        "benchmark_total_return": round(total_return, 2),
        "alpha": 0.0,
        "beta": 1.0,
        "final_value": round(values[-1], 2),
        "initial_capital": initial_capital,
    }


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
        ("SPY Return", f"{metrics.get('benchmark_total_return', 0):+.2f}%"),
        ("Alpha (annualized)", f"{metrics.get('alpha', 0) * 100:+.2f}%"),
        ("Beta", f"{metrics.get('beta', 0):.2f}"),
        ("Cost % of Gross P&L", f"{metrics.get('cost_pct_of_gross_pnl', 0):.2f}%"),
        ("Breakeven Edge/Trade", f"${metrics.get('breakeven_edge_per_trade', 0):,.2f}"),
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
    # Survivorship caveat on EVERY report — absolute return is inflated by
    # delisted names that yfinance silently omits; judge edge by alpha vs SPY.
    console.print(Text(SURVIVORSHIP_CAVEAT, style="yellow"))


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

    # Benchmark / CAPM rows. alpha is stored as an annualized return fraction,
    # rendered here as a percentage. Present per labeled column ("Strategy",
    # "Random-Entry", "SPY") so the survival-vs-edge comparison is explicit.
    bench_rows = [
        ("SPY Return", lambda m: f"{m.get('benchmark_total_return', 0):+.2f}%"),
        ("Alpha (ann.)", lambda m: f"{m.get('alpha', 0) * 100:+.2f}%"),
        ("Beta", lambda m: f"{m.get('beta', 0):.2f}"),
    ]
    for label, fmt in bench_rows:
        table.add_row(label, *[fmt(m) for _, m in results_list])

    console.print(table)


# ---------------------------------------------------------------------------
# Walk-forward report — per-segment, IS→OOS degradation, WFE with fail flag
# ---------------------------------------------------------------------------

# Walk-Forward Efficiency pass bars (STACK.md:22): WFE = annualized_OOS_return /
# annualized_IS_return. < 0.5 is treated as overfit/regime-bound (FAIL); >= 0.7
# is robust; the 0.5–0.7 band passes but is not yet robust.
WFE_FAIL_THRESHOLD = 0.5
WFE_ROBUST_THRESHOLD = 0.7


def walk_forward_wfe_status(wfe: Optional[float]) -> tuple[str, bool]:
    """Classify a WFE into (status_label, is_pass).

    None (undefined: non-positive in-sample return) is UNDEFINED and not a pass.
    WFE < 0.5 FAILs; >= 0.7 is ROBUST; the 0.5–0.7 band is a (non-robust) PASS.
    """
    if wfe is None:
        return ("UNDEFINED", False)
    if wfe < WFE_FAIL_THRESHOLD:
        return ("FAIL", False)
    if wfe >= WFE_ROBUST_THRESHOLD:
        return ("ROBUST", True)
    return ("PASS", True)


def _pooled_oos_tstat(trades: list[Trade]) -> float:
    """Per-trade-return t-stat over the pooled OOS trades (0.0 if undefined).

    Uses each trade's pnl_pct (as a fraction). The t-stat needs at least two
    trades and non-degenerate variance; otherwise it is reported as 0.0 rather
    than NaN so the report never crashes on a thin/identical OOS sample.
    """
    from backtest.stats import per_trade_tstat

    returns = [t.pnl_pct / 100.0 for t in trades]
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return round(per_trade_tstat(returns), 3)


def display_walk_forward(result) -> dict:
    """Render a multi-fold rolling walk-forward and return its gating status.

    Prints a per-fold table (IS vs OOS return, Sharpe, trades, WFE), the
    aggregate pooled-OOS metrics, and a headline aggregate WFE that is visibly
    flagged FAIL when WFE < 0.5 (and noted ROBUST at >= 0.7). The aggregate-OOS
    statistical context from backtest.stats — the per-trade t-stat over pooled
    OOS trades and whether the >=30-OOS-trade gate passes — is surfaced too.

    Returns a status dict (so tests assert on a value, not stdout): aggregate
    WFE, its status label / pass flag, pooled-OOS trade count, the >=30-trade
    gate result, and the pooled-OOS per-trade t-stat.
    """
    from backtest.stats import min_trade_gate

    folds = result.folds
    agg_wfe = result.aggregate_wfe
    status_label, wfe_pass = walk_forward_wfe_status(agg_wfe)

    num_oos_trades = len(result.aggregate_oos_trades)
    trade_gate_pass = min_trade_gate(num_oos_trades)
    oos_tstat = _pooled_oos_tstat(result.aggregate_oos_trades)

    # Per-fold table.
    table = Table(title="Walk-Forward — Per-Fold IS→OOS", show_lines=True)
    table.add_column("Fold", style="cyan bold", justify="right")
    table.add_column("OOS Window", justify="center")
    table.add_column("IS Ret%", justify="right")
    table.add_column("OOS Ret%", justify="right")
    table.add_column("IS Sharpe", justify="right")
    table.add_column("OOS Sharpe", justify="right")
    table.add_column("OOS Trades", justify="right")
    table.add_column("WFE", justify="right")

    for fold in folds:
        ism = fold.in_sample_metrics
        oosm = fold.out_of_sample_metrics
        wfe_txt = "n/a" if fold.wfe is None else f"{fold.wfe:.2f}"
        table.add_row(
            str(fold.index),
            f"{fold.out_of_sample_start} → {fold.out_of_sample_end}",
            f"{ism.get('annualized_return_pct', 0):+.2f}",
            f"{oosm.get('annualized_return_pct', 0):+.2f}",
            f"{ism.get('sharpe_ratio', 0):.2f}",
            f"{oosm.get('sharpe_ratio', 0):.2f}",
            str(oosm.get("num_trades", 0)),
            wfe_txt,
        )
    console.print(table)

    # Aggregate pooled-OOS metrics.
    agg = result.aggregate_oos_metrics
    agg_table = Table(title="Walk-Forward — Aggregate Out-of-Sample", show_lines=True)
    agg_table.add_column("Metric", style="cyan bold")
    agg_table.add_column("Value", justify="right")
    agg_rows = [
        ("Pooled OOS Trades", str(num_oos_trades)),
        ("OOS Total Return", f"{agg.get('total_return_pct', 0):+.2f}%"),
        ("OOS Annualized Return", f"{agg.get('annualized_return_pct', 0):+.2f}%"),
        ("OOS Sharpe", f"{agg.get('sharpe_ratio', 0):.2f}"),
        ("OOS Max Drawdown", f"-{agg.get('max_drawdown_pct', 0):.2f}%"),
        ("OOS Win Rate", f"{agg.get('win_rate_pct', 0):.1f}%"),
        ("Per-Trade t-stat", f"{oos_tstat:.3f}"),
        (">=30-Trade Gate", "PASS" if trade_gate_pass else "FAIL"),
    ]
    for label, value in agg_rows:
        agg_table.add_row(label, value)
    console.print(agg_table)

    # Headline WFE verdict.
    wfe_str = "n/a (undefined IS return)" if agg_wfe is None else f"{agg_wfe:.2f}"
    if status_label == "FAIL":
        style = "bold red"
        verdict = (
            f"WFE = {wfe_str} — FAIL (< {WFE_FAIL_THRESHOLD}): "
            "treat as overfit / regime-bound."
        )
    elif status_label == "ROBUST":
        style = "bold green"
        verdict = f"WFE = {wfe_str} — ROBUST (>= {WFE_ROBUST_THRESHOLD})."
    elif status_label == "PASS":
        style = "yellow"
        verdict = (
            f"WFE = {wfe_str} — PASS (>= {WFE_FAIL_THRESHOLD}) but not yet "
            f"robust (< {WFE_ROBUST_THRESHOLD})."
        )
    else:
        style = "bold red"
        verdict = (
            f"WFE = {wfe_str} — UNDEFINED: no fold had a positive in-sample "
            "return, so edge survival cannot be judged."
        )
    if not trade_gate_pass:
        verdict += (
            f"\nWARNING: only {num_oos_trades} pooled OOS trades (< 30) — "
            "per-trade conclusions are statistically unreliable."
        )
    console.print(Panel(Text(verdict, style=style), title="Walk-Forward Efficiency"))
    console.print(Text(SURVIVORSHIP_CAVEAT, style="yellow"))

    return {
        "aggregate_wfe": agg_wfe,
        "wfe_status": status_label,
        "wfe_pass": wfe_pass,
        "num_oos_trades": num_oos_trades,
        "oos_trade_gate_pass": trade_gate_pass,
        "oos_per_trade_tstat": oos_tstat,
        "num_folds": len(folds),
    }


def compare_ai_value_add(
    screener_metrics: dict,
    ai_metrics: dict,
) -> dict:
    """Compare screener-only vs screener+AI backtest results.

    Returns a dict with both sets of metrics and alpha measurements
    to determine whether the AI analyst adds or destroys value.
    """
    s_trades = screener_metrics.get("num_trades", 0)
    a_trades = ai_metrics.get("num_trades", 0)

    return_alpha = ai_metrics.get("total_return_pct", 0) - screener_metrics.get("total_return_pct", 0)
    sharpe_alpha = ai_metrics.get("sharpe_ratio", 0) - screener_metrics.get("sharpe_ratio", 0)
    pnl_alpha = ai_metrics.get("total_pnl", 0) - screener_metrics.get("total_pnl", 0)

    if s_trades > 0:
        filter_rate = (1 - a_trades / s_trades) * 100
    else:
        filter_rate = 0.0

    return {
        "screener_only": screener_metrics,
        "screener_plus_ai": ai_metrics,
        "alpha": {
            "return_alpha_pct": return_alpha,
            "sharpe_alpha": sharpe_alpha,
            "pnl_alpha": pnl_alpha,
            "ai_filter_rate_pct": filter_rate,
            "ai_adds_value": return_alpha > 0,
        },
    }


def export_metrics_json(metrics: dict, filepath: str) -> None:
    """Export metrics to JSON file."""
    import json
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Metrics exported to %s", filepath)

"""Sweep MAX_EXTENSION_OVER_MA20_PCT across several values and compare.

Runs the same backtest config at each threshold and prints a comparison
table of return, trade count, win rate, and max drawdown.

Pre-downloads YFinance data once and monkey-patches core.data.get_historical_data_yfinance
so all 6 runs read the exact same in-memory DataFrames. Without this, the
built-in 5-minute TTL cache expires mid-sweep.

Usage:
    .venv/bin/python scripts/sweep_extension_pct.py [--sample N] [--days D]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# Allow running from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backtest.engine import BacktestConfig, run_backtest
from backtest.report import calculate_metrics
import core.data as core_data
import backtest.engine as backtest_engine


UNIVERSE_FILE = REPO_ROOT / "data" / "universe_us_2026-04-17.json"

THRESHOLDS = [0.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]  # 0 disables the filter


def load_tickers(sample_size: int | None) -> list[str]:
    data = json.loads(UNIVERSE_FILE.read_text())
    tickers = [s["ticker"] for s in data]
    # Guarantee XNDU and ARTV are included — they're the whole point.
    for must_have in ("XNDU", "ARTV"):
        if must_have not in tickers:
            tickers.append(must_have)
    if sample_size and sample_size < len(tickers):
        head = tickers[:sample_size]
        for must_have in ("XNDU", "ARTV"):
            if must_have not in head:
                head.append(must_have)
        tickers = head
    return tickers


def prefetch_data(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Download one year of history for every ticker up front.

    The built-in core.data._cache has a 5-minute TTL which is too short for
    a multi-threshold sweep over hundreds of tickers. We hold the data in
    this process's memory for the whole run instead.
    """
    print(f"Prefetching 1y daily data for {len(tickers)} tickers...")
    t0 = time.time()
    cache: dict[str, pd.DataFrame] = {}
    for i, ticker in enumerate(tickers, 1):
        try:
            df = core_data.get_historical_data_yfinance(
                ticker, period="1y", interval="1d", market="US",
            )
        except Exception as e:  # pragma: no cover
            print(f"  [{i}/{len(tickers)}] {ticker}: fetch failed — {e}")
            continue
        if df is not None and not df.empty:
            cache[ticker] = df
        if i % 25 == 0 or i == len(tickers):
            print(f"  [{i}/{len(tickers)}] fetched, {len(cache)} with data "
                  f"({time.time() - t0:.1f}s elapsed)")
    print(f"Prefetch complete: {len(cache)}/{len(tickers)} have data "
          f"({time.time() - t0:.1f}s)")
    return cache


def run_sweep(tickers: list[str], days: int) -> list[dict]:
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    # Prefetch once and monkey-patch the backtest's data loader so every
    # threshold run reads the exact same in-memory DataFrames.
    prefetched = prefetch_data(tickers)

    def _patched_fetch(ticker, period="1y", interval="1d", market="US"):
        return prefetched.get(ticker, pd.DataFrame()).copy()

    backtest_engine.get_historical_data_yfinance = _patched_fetch  # type: ignore

    base = BacktestConfig(
        tickers=list(prefetched.keys()),
        market="US",
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        initial_capital=100_000.0,
        use_ai=False,
    )

    rows = []
    for threshold in THRESHOLDS:
        print(f"\n=== Running backtest with MAX_EXTENSION_OVER_MA20_PCT = "
              f"{'DISABLED' if threshold <= 0 else f'{threshold:.0f}%'} ===")
        t0 = time.time()
        cfg = replace(base, max_extension_pct=threshold)
        portfolio = run_backtest(cfg)
        metrics = calculate_metrics(
            portfolio.trades, portfolio.equity_curve, cfg.initial_capital,
        )
        rows.append({
            "threshold": threshold,
            "trades": metrics["num_trades"],
            "win_rate": metrics["win_rate_pct"],
            "total_return": metrics["total_return_pct"],
            "sharpe": metrics["sharpe_ratio"],
            "max_dd": metrics["max_drawdown_pct"],
            "avg_pnl": metrics["avg_pnl_per_trade"],
            "final_value": metrics["final_value"],
            "xndu_bought": any(t.ticker == "XNDU" and t.quantity > 0 for t in portfolio.trades),
            "artv_bought": any(t.ticker == "ARTV" and t.quantity > 0 for t in portfolio.trades),
        })
        print(f"  -> {metrics['num_trades']} trades, "
              f"return {metrics['total_return_pct']:+.2f}%, "
              f"maxDD {metrics['max_drawdown_pct']:.2f}%  "
              f"({time.time() - t0:.1f}s)")
    return rows


def print_table(rows: list[dict]) -> None:
    header = [
        "threshold", "trades", "win%", "return%", "sharpe",
        "maxDD%", "avg$/trade", "final$", "XNDU buy?", "ARTV buy?",
    ]
    widths = [12, 8, 7, 9, 8, 8, 12, 13, 11, 11]
    print()
    print(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        t = "DISABLED" if r["threshold"] <= 0 else f"{r['threshold']:.0f}%"
        line = [
            t,
            str(r["trades"]),
            f"{r['win_rate']:.1f}",
            f"{r['total_return']:+.2f}",
            f"{r['sharpe']:.2f}",
            f"{r['max_dd']:.2f}",
            f"{r['avg_pnl']:+.2f}",
            f"${r['final_value']:,.0f}",
            "YES" if r["xndu_bought"] else "no",
            "YES" if r["artv_bought"] else "no",
        ]
        print(" | ".join(c.ljust(w) for c, w in zip(line, widths)))


def write_csv(rows: list[dict], days: int, n_tickers: int) -> Path:
    """Persist sweep results so future runs can compare against this baseline."""
    out_path = REPO_ROOT / "data" / f"sweep_extension_pct_{date.today().isoformat()}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "threshold", "trades", "win_rate", "total_return",
        "sharpe", "max_dd", "avg_pnl", "final_value",
        "xndu_bought", "artv_bought",
    ]
    with out_path.open("w", newline="") as f:
        f.write(f"# sweep date={date.today().isoformat()} window_days={days} tickers={n_tickers}\n")
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fields})
    print(f"\nResults written to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Only use first N tickers (default: all)")
    parser.add_argument("--days", type=int, default=180,
                        help="Length of backtest window in days")
    args = parser.parse_args()

    tickers = load_tickers(args.sample)
    print(f"Running sweep with {len(tickers)} tickers over {args.days} days "
          f"at thresholds: {THRESHOLDS}")

    rows = run_sweep(tickers, args.days)
    print_table(rows)
    write_csv(rows, days=args.days, n_tickers=len(tickers))


if __name__ == "__main__":
    main()

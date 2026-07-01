"""Walk-forward OOS sweep of MAX_EXTENSION_OVER_MA20_PCT with trial correction.

This REPLACES the previous in-sample curve-fit (which optimised and evaluated on
the same 180-day window and picked the 15% threshold on ~6 in-sample trades).
Instead, for every candidate threshold it runs the multi-fold rolling
walk-forward (backtest.engine.rolling_walk_forward), pools the out-of-sample
trades/equity across folds, and judges each threshold on AGGREGATE OOS only.

Three things every honest parameter sweep must do (PITFALLS.md Pitfall 2 /
STACK.md decision rules) are wired in here:

  1. Report the number of configurations tested (the trial count = len(THRESHOLDS)).
  2. Correct for multiple testing with the Deflated Sharpe Ratio
     (backtest.stats.deflated_sharpe_ratio) using that trial count.
  3. Select a PLATEAU (the middle of the widest stable band of validated
     thresholds), never the single spiking maximum — overfit cliffs vs robust
     plateaus.

A threshold is only "validated" when it clears the full statistical floor:
DSR > 0.95 AND |per-trade t| > 2 AND >= 30 pooled-OOS trades. If nothing clears
the floor, the verdict is INSUFFICIENT EVIDENCE — no threshold is blessed.

Holdout safety: the whole sweep window stays strictly BEFORE the locked holdout
(backtest.holdout.HOLDOUT_START = 2025-07-01). Each fold delegates to
run_backtest, whose assert_range_excludes_holdout preflight refuses any slice
that overlaps the reserved test set.

Pre-downloads daily history once and monkey-patches the backtest's data loader so
every threshold reads the EXACT same in-memory DataFrames (the built-in 5-minute
TTL cache would otherwise expire mid-sweep).

Usage:
    .venv/bin/python scripts/sweep_extension_pct.py [--sample N]
        [--is-days D] [--oos-days D] [--step-days D] [--start YYYY-MM-DD]
        [--end YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# Allow running from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backtest.engine import (
    BacktestConfig,
    DEFAULT_WF_IS_DAYS,
    DEFAULT_WF_OOS_DAYS,
    DEFAULT_WF_STEP_DAYS,
    rolling_walk_forward,
)
from backtest.holdout import HOLDOUT_START
from backtest.stats import (
    deflated_sharpe_ratio,
    min_trade_gate,
    per_trade_tstat,
)
import core.data as core_data
import backtest.engine as backtest_engine


UNIVERSE_FILE = REPO_ROOT / "data" / "universe_us_2026-04-17.json"

# The K configurations under test. The trial count reported by the sweep and fed
# to the Deflated Sharpe Ratio as n_trials is exactly len(THRESHOLDS). Keep this
# list small and defensible — every extra threshold is another trial the DSR must
# deflate against (more trials => harder to clear DSR > 0.95).
THRESHOLDS = [0.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]  # 0 disables the filter

# Statistical floor (STACK.md "Statistical floor" rule). A threshold is only
# "validated" when it clears ALL THREE simultaneously.
DSR_GATE = 0.95          # P(true SR > 0) after trial correction
TSTAT_GATE = 2.0         # |per-trade-return t-stat|
MIN_OOS_TRADES = 30      # >= 30 pooled-OOS trades

# Fallback universe used only when the snapshot file is missing, so the script
# still runs end-to-end for a smoke test. The real sweep loads the frozen
# point-in-time universe snapshot.
_FALLBACK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "NFLX",
    "CRM", "ADBE", "INTC", "QCOM", "AVGO", "TXN", "MU", "ORCL", "CSCO",
    "PEP", "COST",
]


def _holdout_safe_end() -> date:
    """The latest end date that stays strictly BEFORE the locked holdout.

    The sweep must never peek at the reserved single-use holdout, so the default
    backtest window ends the day before HOLDOUT_START. run_backtest's preflight
    would refuse any later range while the holdout is locked.
    """
    return date.fromisoformat(HOLDOUT_START) - timedelta(days=1)


def load_tickers(sample_size: int | None) -> list[str]:
    """Load the frozen universe snapshot (or a small fallback if it is missing).

    The XNDU/ARTV must-include hack from the old in-sample sweep is intentionally
    GONE: hand-picking the two tickers that motivated the threshold is itself a
    form of selection bias and makes no sense for a multi-year OOS sweep judged on
    aggregate statistics. We log the universe size and move on.
    """
    if UNIVERSE_FILE.exists():
        data = json.loads(UNIVERSE_FILE.read_text())
        tickers = [s["ticker"] for s in data]
    else:
        print(
            f"NOTE: universe snapshot {UNIVERSE_FILE.name} not found — "
            f"falling back to a {len(_FALLBACK_TICKERS)}-ticker smoke-test list."
        )
        tickers = list(_FALLBACK_TICKERS)
    if sample_size and sample_size < len(tickers):
        tickers = tickers[:sample_size]
    return tickers


def prefetch_data(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Download multi-year daily history for every ticker up front.

    A walk-forward OOS sweep needs several years of history (multiple ~2y-IS +
    ~1y-OOS folds), so we pull the maximum available window once and hold it in
    this process's memory for the whole run. The backtest's per-fold date filters
    then carve the IS/OOS slices out of these identical DataFrames — every
    threshold reads the exact same data, and the built-in 5-minute TTL cache
    (too short for a long sweep) is bypassed entirely.
    """
    print(f"Prefetching 10y daily data for {len(tickers)} tickers...")
    t0 = time.time()
    cache: dict[str, pd.DataFrame] = {}
    for i, ticker in enumerate(tickers, 1):
        try:
            df = core_data.get_historical_data_yfinance(
                ticker, period="10y", interval="1d", market="US",
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


def _evaluate_threshold(
    base: BacktestConfig,
    threshold: float,
    is_days: int,
    oos_days: int,
    step_days: int,
    n_trials: int,
) -> dict:
    """Run the rolling walk-forward at one threshold and compute its OOS gates.

    Returns a row dict carrying the AGGREGATE-OOS metrics, the pooled-OOS daily
    series length (n_obs), the multiple-testing-corrected DSR, the per-trade
    t-stat, the >=30-trade gate, and the overall validated flag.
    """
    cfg = replace(base, max_extension_pct=threshold)
    result = rolling_walk_forward(
        cfg, is_days=is_days, oos_days=oos_days, step_days=step_days,
    )

    agg = result.aggregate_oos_metrics
    oos_sharpe = float(agg.get("sharpe_ratio", 0.0))
    oos_trades = int(agg.get("num_trades", 0))
    oos_return = float(agg.get("total_return_pct", 0.0))
    oos_maxdd = float(agg.get("max_drawdown_pct", 0.0))
    wfe = result.aggregate_wfe

    # n_obs is LOAD-BEARING: it MUST be the number of OOS trading days (length of
    # the pooled/compounded OOS daily equity series), NOT the trade count. Passing
    # the trade count here would understate the sample size and corrupt the DSR
    # (backtest/stats.py n_obs contract, pinned by 02-05's N=252 known-answer test).
    n_obs = len(result.aggregate_oos_equity)

    # Per-trade return series over the POOLED OOS trades — distinct from n_obs.
    per_trade_returns = [t.pnl_pct for t in result.aggregate_oos_trades]
    if len(per_trade_returns) >= 2:
        tstat = per_trade_tstat(per_trade_returns)
    else:
        tstat = float("nan")

    # Deflated Sharpe Ratio: correct the selected Sharpe for having tried K=n_trials
    # configurations. observed_sr = this threshold's aggregate-OOS Sharpe;
    # n_trials = len(THRESHOLDS); n_obs = pooled-OOS trading-day count (NOT trades).
    if n_obs >= 2:
        dsr = deflated_sharpe_ratio(
            observed_sr=oos_sharpe, n_trials=n_trials, n_obs=n_obs,
        )
    else:
        dsr = 0.0

    dsr_pass = dsr > DSR_GATE
    tstat_pass = (not math.isnan(tstat)) and abs(tstat) > TSTAT_GATE
    trade_pass = min_trade_gate(oos_trades, MIN_OOS_TRADES)
    validated = dsr_pass and tstat_pass and trade_pass

    return {
        "threshold": threshold,
        "oos_trades": oos_trades,
        "oos_return": oos_return,
        "oos_sharpe": oos_sharpe,
        "oos_maxdd": oos_maxdd,
        "wfe": wfe,
        "n_obs": n_obs,
        "dsr": dsr,
        "tstat": tstat,
        "dsr_pass": dsr_pass,
        "tstat_pass": tstat_pass,
        "trade_pass": trade_pass,
        "validated": validated,
    }


def run_sweep(
    tickers: list[str],
    start_date: date,
    end_date: date,
    is_days: int,
    oos_days: int,
    step_days: int,
) -> list[dict]:
    # Prefetch once and monkey-patch the backtest's data loader so every threshold
    # (and every fold within it) reads the exact same in-memory DataFrames.
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

    n_trials = len(THRESHOLDS)
    print(
        f"\nSweeping {n_trials} configurations (trial count = {n_trials}) over "
        f"walk-forward OOS [{start_date} .. {end_date}] "
        f"(IS={is_days}d, OOS={oos_days}d, step={step_days}d trading days)."
    )

    rows = []
    for threshold in THRESHOLDS:
        label = "DISABLED" if threshold <= 0 else f"{threshold:.0f}%"
        print(f"\n=== Walk-forward OOS for MAX_EXTENSION_OVER_MA20_PCT = {label} ===")
        t0 = time.time()
        row = _evaluate_threshold(
            base, threshold, is_days, oos_days, step_days, n_trials,
        )
        rows.append(row)
        print(
            f"  -> aggregate OOS: {row['oos_trades']} trades, "
            f"return {row['oos_return']:+.2f}%, Sharpe {row['oos_sharpe']:.2f}, "
            f"DSR {row['dsr']:.3f}, |t| "
            f"{abs(row['tstat']):.2f} ({time.time() - t0:.1f}s)  "
            f"[{'VALIDATED' if row['validated'] else 'rejected'}]"
        )
    return rows


def select_plateau(
    rows: list[dict],
    metric: str = "oos_sharpe",
    *,
    validated_key: str = "validated",
    rel_tol: float = 0.20,
) -> float | None:
    """Pick the threshold in the MIDDLE of the widest stable band of validated rows.

    Plateau-not-peak selection (STACK.md:28, PITFALLS.md:84). A robust parameter
    sits inside a broad band of similarly-good neighbours; an overfit one is a lone
    spike that collapses if nudged. This helper:

      1. Keeps only rows that pass the full statistical floor (``validated_key``).
         If none are validated it returns ``None`` (INSUFFICIENT EVIDENCE).
      2. Sorts the survivors by threshold and splits them into maximal contiguous
         "flat" bands — consecutive thresholds whose ``metric`` values are within
         ``rel_tol`` of each other (a non-validated gap also breaks a band).
      3. Chooses the WIDEST band (ties broken by higher mean metric, then lower
         threshold) and returns its middle threshold.

    A single spiking maximum forms a width-1 band and therefore loses to any
    genuine multi-point plateau. Returns ``None`` when no validated rows exist.
    """
    if not any(r.get(validated_key) for r in rows):
        return None

    ordered = sorted(rows, key=lambda r: r["threshold"])

    def _flat(a: float, b: float) -> bool:
        scale = max(abs(a), abs(b), 1e-9)
        return abs(a - b) <= rel_tol * scale

    # Build maximal contiguous bands of validated, mutually-flat neighbours.
    # A non-validated row is a GAP that breaks the current band (so a lone peak
    # flanked by rejected neighbours can never be part of a wider plateau); a
    # non-flat metric jump also breaks it.
    bands: list[list[dict]] = []
    current: list[dict] = []
    for r in ordered:
        if not r.get(validated_key):
            if current:
                bands.append(current)
                current = []
            continue
        if current and _flat(current[-1][metric], r[metric]):
            current.append(r)
        else:
            if current:
                bands.append(current)
            current = [r]
    if current:
        bands.append(current)

    def _band_key(band: list[dict]) -> tuple:
        mean_metric = sum(r[metric] for r in band) / len(band)
        # Widest first; then highest mean metric; then lowest starting threshold.
        return (len(band), mean_metric, -band[0]["threshold"])

    best_band = max(bands, key=_band_key)
    middle = best_band[len(best_band) // 2]
    return float(middle["threshold"])


def print_table(rows: list[dict]) -> None:
    header = [
        "threshold", "OOS trades", "OOS ret%", "OOS sharpe", "OOS maxDD%",
        "WFE", "n_obs", "DSR", "|t|", "30+", "verdict",
    ]
    widths = [10, 11, 9, 11, 11, 6, 6, 7, 6, 4, 10]
    print()
    print(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        t = "DISABLED" if r["threshold"] <= 0 else f"{r['threshold']:.0f}%"
        wfe = "n/a" if r["wfe"] is None else f"{r['wfe']:.2f}"
        tstat = "nan" if math.isnan(r["tstat"]) else f"{abs(r['tstat']):.2f}"
        line = [
            t,
            str(r["oos_trades"]),
            f"{r['oos_return']:+.2f}",
            f"{r['oos_sharpe']:.2f}",
            f"{r['oos_maxdd']:.2f}",
            wfe,
            str(r["n_obs"]),
            f"{r['dsr']:.3f}",
            tstat,
            "YES" if r["trade_pass"] else "no",
            "VALIDATED" if r["validated"] else "rejected",
        ]
        print(" | ".join(c.ljust(w) for c, w in zip(line, widths)))


def print_verdict(rows: list[dict]) -> float | None:
    """Print the trial count, the validated band, and the plateau verdict."""
    n_trials = len(THRESHOLDS)
    validated = [r for r in rows if r["validated"]]
    chosen = select_plateau(rows, metric="oos_sharpe")

    print("\n" + "=" * 64)
    print("VERDICT")
    print("=" * 64)
    print(f"Configurations tested (trial count): {n_trials}")
    print(
        "Gates required to validate: "
        f"DSR > {DSR_GATE}, |t| > {TSTAT_GATE:.0f}, "
        f">= {MIN_OOS_TRADES} OOS trades"
    )

    if validated:
        band = ", ".join(
            ("DISABLED" if r["threshold"] <= 0 else f"{r['threshold']:.0f}%")
            for r in validated
        )
        print(f"Validated thresholds (passed all gates): {band}")
    else:
        print("Validated thresholds (passed all gates): NONE")

    if chosen is None:
        print(
            "\nResult: INSUFFICIENT EVIDENCE — no threshold cleared the "
            "statistical floor out-of-sample. Do NOT deploy a tuned extension "
            "threshold on this evidence."
        )
    else:
        label = "DISABLED" if chosen <= 0 else f"{chosen:.0f}%"
        print(
            f"\nResult: selected MAX_EXTENSION_OVER_MA20_PCT = {label} "
            "(middle of the widest validated plateau — chosen for stability, "
            "not peak performance)."
        )
    print("=" * 64)
    return chosen


def write_csv(rows: list[dict], n_tickers: int, chosen: float | None) -> Path:
    """Persist sweep results (with gates + verdict) for future baseline comparison."""
    out_path = REPO_ROOT / "data" / f"sweep_extension_pct_{date.today().isoformat()}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "threshold", "oos_trades", "oos_return", "oos_sharpe", "oos_maxdd",
        "wfe", "n_obs", "dsr", "tstat", "dsr_pass", "tstat_pass",
        "trade_pass", "validated",
    ]
    with out_path.open("w", newline="") as f:
        f.write(
            f"# sweep date={date.today().isoformat()} tickers={n_tickers} "
            f"trial_count={len(THRESHOLDS)} "
            f"selected={'none' if chosen is None else chosen}\n"
        )
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
    parser.add_argument("--is-days", type=int, default=DEFAULT_WF_IS_DAYS,
                        help="In-sample length in TRADING days (default ~2y)")
    parser.add_argument("--oos-days", type=int, default=DEFAULT_WF_OOS_DAYS,
                        help="Out-of-sample length in TRADING days (default ~12mo)")
    parser.add_argument("--step-days", type=int, default=DEFAULT_WF_STEP_DAYS,
                        help="Fold step in TRADING days (default = OOS length)")
    parser.add_argument("--start", type=str, default=None,
                        help="Window start YYYY-MM-DD (default: end - 5y)")
    parser.add_argument("--end", type=str, default=None,
                        help="Window end YYYY-MM-DD (default: day before the "
                             "locked holdout; MUST stay < %s)" % HOLDOUT_START)
    args = parser.parse_args()

    end_date = date.fromisoformat(args.end) if args.end else _holdout_safe_end()
    start_date = (
        date.fromisoformat(args.start) if args.start
        else end_date - timedelta(days=5 * 365)
    )

    tickers = load_tickers(args.sample)
    print(
        f"Walk-forward extension sweep: {len(tickers)} tickers, "
        f"thresholds {THRESHOLDS} (trial count {len(THRESHOLDS)})."
    )

    rows = run_sweep(
        tickers, start_date, end_date,
        is_days=args.is_days, oos_days=args.oos_days, step_days=args.step_days,
    )
    print_table(rows)
    chosen = print_verdict(rows)
    write_csv(rows, n_tickers=len(tickers), chosen=chosen)


if __name__ == "__main__":
    main()

# Risk Parameter Tuning — Paper to Small Account

## Context

The original risk parameters were designed for a well-funded paper trading account with many positions open simultaneously. When switching to a small live account ($500), these defaults blocked every single trade from executing. This document explains what changed and why.

## Parameter Comparison

| Parameter | Paper Default | Small Account | Reasoning |
|---|---|---|---|
| `MAX_POSITION_SIZE_PCT` | 5% ($25) | 15% ($75) | Hard ceiling on a single position. `RISK_PER_TRADE_PCT` (below) is the primary sizing control; `MAX_POSITION_SIZE_PCT` only bites when the stop-loss distance is very tight. 15% is wide enough for a $500 account to enter most stocks but tight enough to prevent a single bad position from dominating the portfolio. |
| `DAILY_LOSS_LIMIT_PCT` | 2% ($10) | 10% ($50) | A $10 daily limit meant any single trade's stop-loss would trip the circuit breaker. 10% gives room for 2-3 trades to hit stops before halting. |
| `MAX_OPEN_POSITIONS` | 10 | 3 | With $500, spreading across 10 positions means ~$50 each — too small for meaningful trades. 3 positions at ~$165 each is more practical. |
| `RISK_PER_TRADE_PCT` | 1% ($5) | 5% ($25) | $5 risk per trade produced tiny position sizes (often 0-1 shares). $25 risk allows proper sizing while still limiting downside. |
| `MAX_SECTOR_CONCENTRATION_PCT` | 25% ($125) | 50% ($250) | With 10 positions, 25% sector cap ensures diversification across 4+ sectors. With 3 positions, 25% is impossible to satisfy — you can't diversify 3 positions across 4 sectors. 50% allows 2 positions in the same sector. |
| `ANTI_MOMENTUM_PCT` | 5% | 8% | In volatile markets (tariff news, macro uncertainty), many stocks move 5-8% intraday. The 5% limit rejected valid entries that had already moved. 8% still prevents chasing runaway moves while allowing normal volatility. |

## Unchanged Parameters

These were kept as-is because they work regardless of account size:

| Parameter | Value | Why Unchanged |
|---|---|---|
| `AI_CONFIDENCE_THRESHOLD` | 65 | Quality filter — no reason to lower the bar on trade conviction. |
| `DEFAULT_STOP_LOSS_PCT` | 3% | Per-trade protection scales with position size, not account size. |
| `DEFAULT_TAKE_PROFIT_PCT` | 6% | Same reasoning as stop-loss. |
| `MIN_RISK_REWARD_RATIO` | 1.5 | Mathematical edge requirement — independent of account size. |
| `ALLOW_SHORT_SELLING` | False | We don't hold shares to sell short. Not an account size issue. |
| `TREND_CONFIRMATION` | True | MA alignment check — pure signal quality, not sizing. |
| `CIRCUIT_BREAKER_LOSSES` | 3 | 3 consecutive losses should pause trading regardless of account size. |

## Why the Paper Defaults Failed

The core issue: risk parameters designed for percentage-based limits assume the account is large enough that those percentages produce tradeable dollar amounts.

With a $10,000 account:
- 5% max position = $500 (can buy most stocks)
- 2% daily loss = $200 (room for several stops)
- 25% sector cap = $2,500 (plenty of room)

With a $500 account:
- 5% max position = $25 (can't buy a single share of INTC at $62)
- 2% daily loss = $10 (one stop-loss and you're done)
- 25% sector cap = $125 (one position in any sector and it's full)

## Ghost Positions Bug (April 2026)

Two stale positions (HTZ: 8,282 shares, RYDE: 8,282 shares) were stuck in the database from April 8th. These were never actually executed but inflated cumulative risk to $43,816 — blocking every trade with "Cumulative risk would exceed daily loss limit." Cleared manually from the SQLite database.

## Scaling Back Up

When the account grows, these parameters should be tightened back toward the paper defaults:

- **$2,000+**: Consider `MAX_POSITION_SIZE_PCT=10`, `MAX_OPEN_POSITIONS=5`
- **$5,000+**: Consider `MAX_SECTOR_CONCENTRATION_PCT=30`, `DAILY_LOSS_LIMIT_PCT=5`
- **$10,000+**: Return to original paper trading defaults

---

## Portfolio Heat Cap (`MAX_PORTFOLIO_HEAT_PCT`)

`MAX_PORTFOLIO_HEAT_PCT` (default 6%) caps the total open at-risk capital across all positions as a percentage of equity. "At-risk capital" for each position is `(entry_price - stop_loss) * quantity` — the most you can lose if every stop fires simultaneously.

Key properties:
- **Entry-only**: This check gates new entries only. It never blocks exits. If the heat cap is already exceeded because existing positions moved against you, you can still close them.
- **Tighter than the daily-loss limit**: The daily loss limit halts trading after actual losses; the heat cap prevents you from committing to losses before they happen. At 6% heat cap with a 10% daily loss limit, you can sustain roughly 1–2 simultaneous stop-outs before hitting the daily halt.
- **Complements `RISK_PER_TRADE_PCT`**: `RISK_PER_TRADE_PCT` controls per-trade risk; `MAX_PORTFOLIO_HEAT_PCT` controls the aggregate. Both must pass for a new entry.

Tuning guidance:
- At `MAX_OPEN_POSITIONS=3`, `RISK_PER_TRADE_PCT=5%`, heat will typically land around 15% if all three positions use full risk — well above the 6% cap. Tighten `RISK_PER_TRADE_PCT` to 2–3% or leave `MAX_PORTFOLIO_HEAT_PCT` higher (e.g., 15%) if you want 3 simultaneous full-risk positions.
- For a $500 account running 1–2 positions at a time, 6% is a reasonable default.

---

## Intraday-Margin Parameters & `MARGIN_REGIME`

The FINRA PDT rule was eliminated 2026-06-04. The margin protection model was updated in Phase 1. Key parameters:

| Parameter | Value | Description |
|---|---|---|
| `REG_T_MIN_EQUITY_USD` | `2000.0` | Reg-T minimum equity to trade on margin. New entries that would leave the account below $2,000 equity are blocked. |
| `INTRADAY_MAINTENANCE_MARGIN_PCT` | `25.0` | Intraday maintenance margin floor (25%). Entries that would leave margin below this threshold are blocked to prevent uncured deficits that trigger a 90-day broker restriction. |
| `MARGIN_REGIME` | `"both"` | Selects the active margin model. Valid values: `"intraday"` (new rules only), `"legacy_pdt"` (old $25,000 gate only), `"both"` (run both checks simultaneously). |
| `LEGACY_PDT_THRESHOLD_USD` | `25000.0` | The correct legacy PDT threshold. Used only when `MARGIN_REGIME` is `"legacy_pdt"` or `"both"`. |

**When to change `MARGIN_REGIME`**: Leave it at `"both"` until your IBKR account has fully migrated to the new intraday-margin regime (IBKR phase-in runs through 2027-10-20). Once your account is confirmed on the new regime, set `MARGIN_REGIME=intraday` to disable the legacy gate entirely. The env var form is `MARGIN_REGIME=intraday` in your `.env` file.

---

## Swing Default & `DAY_TRADE_ENABLED`

As of Phase 1, swing trading is the default trade cadence:

| Parameter | Default | Description |
|---|---|---|
| `DEFAULT_TRADE_TYPE` | `"swing"` | All signals default to SWING unless overridden |
| `DAY_TRADE_ENABLED` | `False` | When false, the day-trading path is disabled. AI signals that return `"day"` are downgraded to `"swing"`. End-of-day auto-close only runs when this is True. |

**Why swing-default?** Day-trading suitability will be determined by the Phase-2 backtest harness running out-of-sample validation. Until those results confirm day-trading adds value, swing is the safer default — it avoids intraday margin and PDT concerns and gives positions time to develop.

To enable day-trading: set `DAY_TRADE_ENABLED=True` in `.env`. Do this only after reviewing the Phase-2 backtest output.

---

## Edge-Validation Harness (Phase 2)

Tuning a parameter on the same data you then report results on produces a number that looks great and predicts nothing. Phase 2 replaced ad-hoc, single-window tuning with an **out-of-sample, benchmarked, statistically-gated** harness. The rule is simple: **no parameter is "validated" unless it beats a passive index *and* a coin-flip control, out-of-sample, net of realistic costs.** No edge has been demonstrated yet — the harness exists to test for one honestly.

### What "net of realistic costs" means

Every backtest number is charged slippage (0.1%), a per-leg bid-ask spread (`BACKTEST_SPREAD_BPS`, default 5 bps crossed on each side, so a round trip pays it twice), and commission ($1/trade). Gaps *through* a stop fill at the bar open, never the stop price. The report surfaces **Cost % of Gross P&L** and **Breakeven Edge/Trade** so a thin edge that frictions would erase is visible immediately. Reported absolute returns also carry a **survivorship caveat** (the universe is a point-in-time snapshot excluding delisted names), so the alpha-vs-SPY figure matters more than the raw return.

### Pass bars (all must hold, out-of-sample, net of costs)

| Gate | Bar | Where |
|---|---|---|
| **CAPM alpha vs SPY** | `> 0` (risk-free subtracted from both legs) | `calculate_capm_metrics` in `backtest/report.py` |
| **Beats random-entry control** | OOS return & Sharpe exceed the seeded Bernoulli(0.5) coin-flip that uses the *same* sizing/exits/costs | `run_strategy_with_controls` |
| **OOS net return & Sharpe ≥ SPY** | strategy must clear the passive benchmark | benchmark columns in the report |
| **Walk-Forward Efficiency (WFE)** | `≥ 0.50` (target `0.70`); FAIL below 0.5 | `rolling_walk_forward` / `walk_forward_wfe_status` |
| **Out-of-sample sample size** | `≥ 30` OOS trades (else *insufficient evidence*) | `min_trade_gate` in `backtest/stats.py` |
| **Deflated Sharpe Ratio (DSR)** | `> 0.95`, trial-count corrected | `deflated_sharpe_ratio` |
| **Per-trade t-statistic** | `|t| > 2` | `per_trade_tstat` |

These thresholds are **pre-registered and locked** in the repo-root `ACCEPTANCE-CRITERIA.md` (and mirrored for planning convenience at `.planning/ACCEPTANCE-CRITERIA.md`). They were frozen *before* any holdout-touching run to prevent p-hacking; do not restate or "adjust" them here — point operators at the locked file so the bar cannot drift after the fact. If any gate fails → no edge demonstrated → do not deploy real money (index instead).

### Plateau-not-peak parameter selection

When you sweep a parameter, the single best-performing value is almost always an overfit cliff that collapses out-of-sample. The harness instead:

1. Evaluates every candidate on **aggregate out-of-sample** folds (`rolling_walk_forward`), never on the window it was tuned on.
2. Reports the **trial count** (number of configurations tested) so the DSR can correct for multiple testing.
3. Keeps only thresholds that pass the full statistical floor (DSR > 0.95 **and** |t| > 2 **and** ≥ 30 OOS trades).
4. Picks the **middle of the widest contiguous validated band** (`select_plateau`), not the peak. A non-validated row breaks a band, so a lone spike cannot be glued into a plateau.
5. Emits either a single validated threshold or an explicit **INSUFFICIENT EVIDENCE** verdict.

### Re-validating the extension threshold (`MAX_EXTENSION_OVER_MA20_PCT`)

The parabolic-breakout guard ships at **15%** as a conservative default. The earlier framing — a "15% sweet spot" chosen from a 180-day in-sample window on ~6 trades — was a textbook curve-fit and has been **retired**: it optimized and evaluated the threshold on the *same* window. The replacement, `scripts/sweep_extension_pct.py`, now drives `rolling_walk_forward` per threshold over ~5 years of multi-regime history (ending strictly before the locked holdout), applies the plateau selection above, and prints the trial count, per-threshold DSR / t-stat / 30-trade gate, and the plateau verdict.

```bash
# Re-run the walk-forward OOS extension sweep against fresh data
.venv/bin/python scripts/sweep_extension_pct.py --sample 20
```

Treat the printed plateau (or INSUFFICIENT EVIDENCE) as the only basis for changing the threshold — never a single-window result.

### Dow filter & SPY market-regime gate (DOW-01 / DOW-02 / DOW-03)

A pure `dow_trend` classifier (DOW-01) feeds two **toggleable, default-OFF** filters on `BacktestConfig`:

| Toggle | Effect | Status |
|---|---|---|
| `use_dow_filter` (DOW-02) | Per-ticker entry filter: drops candidates whose own Dow trend is not up | **OFF — unproven OOS** |
| `use_market_regime_filter` (DOW-03) | SPY market-regime gate: blocks long entries while SPY is not in an uptrend (reads the raw full-history SPY close, sliced strictly `< current_date` for no look-ahead) | **OFF — unproven OOS** |

Both ship OFF and are adopted into the live path **only if they beat their no-filter baseline net of costs, out-of-sample.** Their Phase-2 with-vs-without walk-forward (8-ticker universe, 2018-01-01 → 2025-06-30) did **not** beat baseline: the per-ticker filter lowered OOS net return and Sharpe and collapsed WFE; the regime gate was a no-op because the risk manager's existing MA5>MA10>MA20 trend-confirmation already suppresses buys in falling markets. They remain available-but-off so they can be re-evaluated against other universes without code changes. **Do not enable them as a tuning lever** until a fresh with-vs-without OOS run shows they beat baseline.

### Pre-registration & single-use holdout discipline

- **Pre-registration.** The acceptance gates above were locked in `ACCEPTANCE-CRITERIA.md` before tuning. Every parameter sweep counts as a multiple-testing trial and must report its trial count + DSR.
- **Single-use holdout.** The window `2025-07-01 → 2026-06-29` is reserved and mechanically protected: `run_backtest` raises `PermissionError` on any overlapping range until a Phase-4 unlock (`BORSA_HOLDOUT_UNLOCKED=1`). All tuning and sweeps must set `--backtest-end` before `2025-07-01`. The holdout is touched **exactly once**, after all tuning is frozen — re-tuning against it would destroy its value as an honest out-of-sample test.

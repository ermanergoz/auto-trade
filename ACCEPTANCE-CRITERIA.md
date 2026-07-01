# Edge-Validation Acceptance Criteria (pre-registered)

**Status: LOCKED** — committed 2026-06-28, before any holdout-touching run.

These thresholds were chosen *before* tuning to prevent data-snooping / p-hacking.
The strategy is considered to have a real edge only if it clears **every** gate below
on out-of-sample data, net of realistic costs (commission + spread + gap-through slippage).

## Go / No-Go gate (all must hold, OOS, net of costs)

1. **Beats SPY** — CAPM annualized alpha vs SPY buy-and-hold is **> 0** (risk-free subtracted from both legs).
2. **Beats the random-entry control** — strategy OOS return/Sharpe exceeds the coin-flip control that uses the *same* sizing, exits, and costs.
3. **Walk-Forward Efficiency** — WFE **≥ 0.50** (target 0.70); IS→OOS degradation within bounds.
4. **Sample size** — **≥ 30** out-of-sample trades.
5. **Deflated Sharpe Ratio** — **DSR > 0.95** (trial-count corrected for the number of configurations tested).

If any gate fails → **no edge demonstrated → do not deploy real money** (index instead).

## Windows

- **Tuning / walk-forward history:** ~5 years (≈2021-06 → 2025-06), multiple regimes incl. the 2022 drawdown.
- **Single-use holdout (LOCKED):** **2025-07-01 → 2026-06-29**. Touched exactly once, in Phase 4, after all tuning is frozen. Enforced by a `run_backtest` preflight that raises if a run's range overlaps the holdout while locked.
- **Account size for backtests:** realistic sub-$25k `--capital` (exercises the swing/margin logic).

## Rules

- No re-tuning against the holdout. Each parameter sweep counts as a multiple-testing trial and must report its trial count + DSR.
- Parameter selection uses the **plateau** (stable middle of an OOS band), never the single peak.
- Costs modeled pessimistically; the edge must survive them (report "cost as % of gross P&L" and "breakeven edge per trade").

---
*Pre-registered for the Auto-Trade (borsa) swing-validation milestone, Phase 2.*

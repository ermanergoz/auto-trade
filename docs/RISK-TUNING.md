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

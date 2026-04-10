# Risk Parameter Tuning — Paper to Small Account

## Context

The original risk parameters were designed for a well-funded paper trading account with many positions open simultaneously. When switching to a small live account ($500), these defaults blocked every single trade from executing. This document explains what changed and why.

## Parameter Comparison

| Parameter | Paper Default | Small Account | Reasoning |
|---|---|---|---|
| `MAX_POSITION_SIZE_PCT` | 5% ($25) | 50% ($250) | At 5%, max position was $25 — cheaper than a single share of most US stocks. At 50%, we can buy 1-2 shares of stocks up to $250. |
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

- **$2,000+**: Consider `MAX_POSITION_SIZE_PCT=25`, `MAX_OPEN_POSITIONS=5`
- **$5,000+**: Consider `MAX_SECTOR_CONCENTRATION_PCT=30`, `DAILY_LOSS_LIMIT_PCT=5`
- **$10,000+**: Return to original paper trading defaults

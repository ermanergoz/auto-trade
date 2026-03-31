# Auto Trader - AI Instructions

This project is an automated stock trading system. Read these docs before starting any work:

1. `docs/DESIGN.md` — Full design spec (architecture, components, tech stack, configuration)
2. `docs/IMPLEMENTATION-PLAN.md` — Step-by-step implementation plan with 8 milestones, file creation order, verification steps, and machine setup guide

## Key Context

- **What**: Automated day/swing trading bot for US and Turkish (BIST) stocks
- **Broker**: Interactive Brokers (IBKR) via `ib_insync` Python library
- **Data**: IBKR is the primary data source (historical + real-time). YFinance is fallback for backtesting only.
- **Strategy**: Technical screener filters thousands of stocks -> AI (Claude/GPT) analyzes top ~20 candidates -> Risk manager gates trades -> IBKR executes
- **Excludes**: Financial sector stocks (banks, insurance, lending companies)
- **Modes**: paper (default), live, backtest, dry-run
- **Notifications**: Telegram bot
- **Database**: SQLite
- **Language**: Python 3.11+

## Implementation Order

Follow the milestones in `docs/IMPLEMENTATION-PLAN.md` strictly in order:
1. Core Infrastructure (IBKR connection, data, portfolio, models)
2. Technical Screener (universe builder, indicators)
3. AI Analyst (LLM integration)
4. Risk Manager + Execution Engine
5. Notifications + Logging (can parallel with M6)
6. Backtesting Engine (can parallel with M5)
7. Paper Trading Shakedown
8. Options Support (future)

## Architecture Rule

The screener and risk manager must be written as **pure functions** that accept data as input (not fetch it themselves). This allows the backtester to feed them historical data without code duplication.

## Safety

- Paper trading is the default. Live mode requires `--mode live` flag.
- Every trade must have a stop-loss.
- Daily loss limit halts all trading automatically.
- Never commit `.env` files.

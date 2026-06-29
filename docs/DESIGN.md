# Auto Stock Trader - Design Spec

## Overview

An automated stock trading system that trades US equities through Interactive Brokers (swing trading is the default cadence; day trading is gated behind `DAY_TRADE_ENABLED`, which defaults to false). Uses technical indicators for broad market screening and LLM-powered analysis for final trade decisions. Excludes financial sector stocks.

## Broker & Account

- **Broker**: Interactive Brokers (IBKR)
- **API**: `ib_insync` Python library connecting to TWS or IB Gateway
- **Paper trading first**, then small real money
- **Single account** for US markets
- Paper vs live toggle: same code, different IBKR connection port (7497 paper, 7496 live)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Scheduler                            │
│  (Runs during US: 16:30-23:00 TRT)                         │
└─────────┬───────────────────────────────────┬───────────────┘
          │                                   │
  ┌───────▼────────┐                 ┌────────▼───────┐
  │  Stock Universe │                 │  Portfolio     │
  │  Builder        │                 │  Tracker       │
  │  (IBKR scanner  │                 │  (SQLite)      │
  │   + filters)    │                 └────────┬───────┘
  └───────┬────────┘                          │
          │                                   │
  ┌───────▼────────┐                          │
  │  Market Data    │                          │
  │  (YFinance +    │                          │
  │   News API)     │                          │
  └───────┬────────┘                          │
          │                                   │
  ┌───────▼────────┐                          │
  │  Technical      │                          │
  │  Screener       │  Candidates (~10-20)    │
  │  (RSI, MACD,    ├──────────┐              │
  │   MA, Volume)   │          │              │
  └────────────────┘          │              │
                       ┌──────▼──────┐        │
                       │  AI Analyst  │        │
                       │  (Gemini /   │        │
                       │   Ollama +   │        │
                       │    News)     │        │
                       └──────┬──────┘        │
                              │               │
                       ┌──────▼──────┐        │
                       │  Risk       │◄───────┘
                       │  Manager    │
                       └──────┬──────┘
                              │
                       ┌──────▼──────┐
                       │  Execution  │
                       │  (IBKR)     │
                       └──────┬──────┘
                              │
                       ┌──────▼──────┐
                       │  Logger +   │
                       │  Telegram   │
                       └─────────────┘
```

## Components

### 1. Scheduler

Orchestrates the trading loop. Runs on the local machine.

- Detects which markets are open based on current time
- Runs the full pipeline (screen -> analyze -> trade) on a configurable interval (e.g., every 15 minutes)
- Handles graceful shutdown, market close procedures
- Schedule: US 16:30-23:00 TRT

### 2. Stock Universe Builder

Builds the tradeable stock list, updated daily.

- Pulls all available tickers from IBKR for US (NYSE, NASDAQ) exchanges
- Filters OUT financial sector stocks (GICS sector "Financials" — banks, insurance, capital markets, consumer finance, mortgage/lending)
- Applies liquidity filters: minimum average daily volume, minimum market cap
- Caches the universe daily (doesn't change intraday)

### 3. Market Data Service

Provides price data and news.

- **Price data (primary)**: IBKR historical data via `ib_insync` `reqHistoricalData()` — no extra API needed
- **Price data (backtest fallback)**: YFinance for bulk historical downloads when IBKR connection isn't available (backtest mode)
- **Real-time quotes**: IBKR streaming market data via `reqMktData()` for active positions and screener hits
- **News**: Tavily API for stock news, yfinance as free fallback
- Caches aggressively to avoid redundant requests within a scan interval

### 4. Technical Screener

Fast, cheap pass over the full stock universe.

Flags stocks that match any of these patterns:
- RSI oversold (<30) or overbought (>70)
- MACD crossover (bullish or bearish)
- Moving average crossover (MA5 crosses MA20)
- Volume spike (>2x average daily volume)
- Bollinger Band breakout
- Price near support/resistance levels

Output: ~10-20 candidates per market per scan interval.

### 5. AI Analyst

Deep analysis on screener candidates only.

For each candidate:
- Gathers: technical indicator values, recent price action, news headlines, sector performance
- Sends structured prompt through the provider router (`core/analyst._call_llm`): Gemini (`gemini-2.5-flash-lite` by default) when `GEMINI_API_KEY` is set and the process-lifetime exhaustion flag is clear, otherwise (or on Gemini transport failure / credits depleted) Ollama + Qwen 2.5 7B locally
- Receives structured response: `{action: buy|sell|hold, confidence: 0-100, entry_price, stop_loss, take_profit, trade_type, reasoning}`
- Confidence threshold: only act on signals with confidence >= 65 (configurable)
- Fallback semantics mirror the Tavily→yfinance news path: permanent exhaustion (401/403, depleted credits) latches a process-wide flag; transient failures (5xx, network, per-minute 429) fall back for just this call

### 6. Risk Manager

Every trade must pass through risk checks before execution.

Rules:
- **Position size**: Max 5% of portfolio per position (configurable)
- **Daily loss limit**: Stop trading if daily P&L drops below -2% of portfolio
- **Max open positions**: 10 concurrent positions (configurable)
- **Stop-loss required**: Every trade has a stop-loss (set by AI Analyst or default 3%)
- **Sector concentration**: Max 25% of portfolio in any one sector
- **No duplicate positions**: Can't buy more of a stock you already hold (unless scaling in is enabled)

### 7. Execution Engine

Interfaces with IBKR to place and manage orders.

- Places market or limit orders via `ib_insync`
- Attaches stop-loss orders (bracket orders)
- Monitors order fills and partial fills
- Handles connection drops and reconnection
- For day trades: closes all intraday positions before market close (applies only when `DAY_TRADE_ENABLED` is True)
- For swing trades: keeps positions open, manages trailing stops

### 8. Portfolio Tracker

Persistent state in SQLite.

Tables:
- `positions` — open positions with entry price, quantity, stop-loss, take_profit
- `trades` — completed trades with entry/exit prices, P&L, reasoning
- `daily_summary` — daily portfolio value, P&L, number of trades
- `signals` — all signals generated (for backtesting comparison)

### 9. Backtesting Engine

Replays historical data through the same Strategy Engine code.

- Downloads historical data via IBKR or YFinance (fallback for bulk downloads)
- Runs Technical Screener + AI Analyst (or cached signals) on historical data
- Simulates order execution with configurable slippage and commission
- Calculates: total return, Sharpe ratio, max drawdown, win rate, profit factor
- Compares multiple strategy configurations side by side
- Uses the SAME signal generation code as live trading (no code duplication)

### 10. Logger & Notifications

- **Trade log**: Every trade with full context (signal, indicators, news, reasoning, outcome)
- **Terminal dashboard**: Real-time display of positions, P&L, recent trades
- **Telegram bot**: Sends alerts for trades executed, daily P&L summary, risk warnings
- **CSV export**: For external analysis

## Tech Stack

- **Language**: Python 3.11+
- **Broker API**: `ib_insync` (or `ib_async`, maintained fork)
- **Market data (primary)**: IBKR via `ib_insync` (historical + real-time)
- **Market data (backtest fallback)**: `yfinance` for bulk historical downloads
- **Technical analysis**: `pandas-ta` or `ta-lib`
- **AI**: Gemini (primary) via direct `urllib.request` calls to the Generative Language API; Ollama (fallback) via local HTTP
- **Database**: SQLite via `sqlite3`
- **Notifications**: `python-telegram-bot`
- **News**: Tavily API
- **Scheduling**: `schedule` or `APScheduler`
- **CLI**: `rich` for terminal dashboard

## Project Structure

```
auto-trader/
├── config/
│   ├── settings.py          # All configurable parameters
│   └── .env                 # API keys (gitignored)
├── core/
│   ├── scheduler.py         # Main loop and market hours
│   ├── universe.py          # Stock universe builder
│   ├── data.py              # Market data service
│   ├── screener.py          # Technical screener
│   ├── analyst.py           # AI analyst (LLM integration)
│   ├── risk.py              # Risk manager
│   ├── executor.py          # IBKR order execution
│   ├── portfolio.py         # Portfolio tracker (SQLite)
│   └── models.py            # Data classes (Signal, Position, Trade)
├── backtest/
│   ├── engine.py            # Backtesting engine
│   └── report.py            # Backtest results and comparison
├── notifications/
│   └── telegram.py          # Telegram bot notifications
├── tests/
│   └── ...
├── main.py                  # Entry point
├── requirements.txt
└── README.md
```

## Trading Modes

- **Paper mode** (default): Connects to IBKR paper trading account. No real money.
- **Live mode**: Connects to IBKR live account. Requires explicit `--live` flag.
- **Backtest mode**: Runs historical simulation. No broker connection needed.
- **Dry-run mode**: Runs full pipeline but logs what it WOULD trade without placing orders.

## Configuration (settings.py)

```python
# Broker
IBKR_HOST = "127.0.0.1"
IBKR_PORT = 7497  # 7497=paper, 7496=live
IBKR_CLIENT_ID = 1

# Markets
MARKETS = ["US"]
EXCLUDED_SECTORS = ["Financials"]
MIN_DAILY_VOLUME = 100_000
MIN_MARKET_CAP = 50_000_000  # $50M

# Strategy
SCAN_INTERVAL_MINUTES = 15
AI_CONFIDENCE_THRESHOLD = 65
AI_PROVIDER = "gemini"                  # "gemini" (auto-falls back to Ollama) or "ollama"
GEMINI_API_KEY = ""                     # leave blank to disable Gemini and use Ollama only
GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_HOST = "https://generativelanguage.googleapis.com"
AI_MODEL = "qwen2.5:7b"                 # Ollama fallback model
OLLAMA_HOST = "http://localhost:11434"

# Risk
MAX_POSITION_SIZE_PCT = 5.0
DAILY_LOSS_LIMIT_PCT = 2.0
MAX_OPEN_POSITIONS = 10
DEFAULT_STOP_LOSS_PCT = 3.0
MAX_SECTOR_CONCENTRATION_PCT = 25.0

# Trade Type / Day-Trade Settings
DEFAULT_TRADE_TYPE = "swing"    # "swing" (default) or "day"
DAY_TRADE_ENABLED = False       # gate day-trading path; when False AI "day" signals are downgraded to "swing"
CLOSE_DAY_TRADES_BEFORE_MARKET_CLOSE = True  # applies only when DAY_TRADE_ENABLED = True
CLOSE_MINUTES_BEFORE = 15

# Notifications
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""
```

## Milestones

1. **Core infrastructure**: Project setup, IBKR connection, market data, portfolio tracker, SQLite
2. **Technical screener**: Implement indicators, build stock universe, run scans
3. **AI analyst**: LLM integration, structured prompts, signal generation
4. **Risk manager + execution**: Risk rules, order placement, stop-losses
5. **Notifications + logging**: Telegram bot, trade journal, terminal dashboard
6. **Backtesting**: Historical replay, performance metrics
7. **Paper trading shakedown**: Run on paper for 1-2 weeks, tune parameters
8. **Options support** (future): Add options trading as a later milestone

## Key Decisions

- **Build from scratch** rather than forking `daily_stock_analysis` — that repo's architecture is built for notifications, not execution, and has lots of Chinese-market-specific code
- **IBKR as single broker** for US markets
- **IBKR as primary data source** — already connected for trading, provides both historical and real-time data for US stocks. YFinance only as backtest fallback for bulk downloads. This eliminates an external dependency and avoids YFinance reliability issues.
- **Screener-then-AI pipeline** to keep LLM costs minimal (Gemini Flash-Lite is cheap-to-free at this volume; Ollama fallback is free)
- **Gemini-primary, Ollama-fallback LLM routing** — reuses the same process-lifetime exhaustion-flag pattern as Tavily→yfinance news fallback; no general multi-provider abstraction. Transport failures on Gemini (HTTP 5xx, network, credits depleted) fall straight through to Ollama rather than burning retries on stateless server errors; content-level failures (malformed JSON) retry Gemini up to 3 times because re-prompting can yield a parseable response
- **SQLite** instead of PostgreSQL — simpler for a local single-user system
- **Skip options for now** — add as a future milestone once stock trading is stable
- **Python** — best ecosystem for trading (ib_insync, pandas, yfinance, ta-lib)

## Safety

- Paper trading mode is the default; live mode requires explicit opt-in
- Daily loss limit halts all trading automatically
- Every trade has a mandatory stop-loss
- Dry-run mode lets you observe without executing
- All trades are logged with full reasoning for review

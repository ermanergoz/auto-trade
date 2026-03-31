# Auto Trade

An automated stock trading system that day-trades and swing-trades US (NYSE/NASDAQ) and Turkish (BIST) equities through Interactive Brokers. The system uses a two-stage pipeline: a fast technical screener filters thousands of stocks down to ~20 candidates, then an LLM (Claude or GPT) performs deep analysis on only those candidates. A risk manager gates every trade before execution through IBKR.

Financial sector stocks (banks, insurance, lending companies) are automatically excluded from all trading.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Trading Pipeline](#trading-pipeline)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Trading Modes](#trading-modes)
- [Technical Screener](#technical-screener)
- [AI Analyst](#ai-analyst)
- [Risk Management](#risk-management)
- [Backtesting](#backtesting)
- [Notifications](#notifications)
- [Database Schema](#database-schema)
- [Testing](#testing)
- [Development Guide](#development-guide)
- [Safety Features](#safety-features)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

- **Multi-market support**: Trades both US (NYSE/NASDAQ) and Turkish (BIST) equities through a single IBKR account
- **Two-stage screening pipeline**: Technical screener (fast, free) filters thousands of stocks, then AI analyst (Claude/GPT) performs deep analysis on ~20 candidates per market
- **AI-powered analysis**: Structured LLM integration with Claude and GPT-4 support, using tool_use/function_calling for reliable JSON output
- **Comprehensive risk management**: Position sizing, daily loss limits, sector concentration limits, mandatory stop-losses, and duplicate position prevention
- **Bracket order execution**: Automatic stop-loss and take-profit orders attached to every trade via IBKR bracket orders
- **Day and swing trading**: Automatic end-of-day position closing for day trades, trailing stops for swing trades
- **Backtesting engine**: Replay historical data through the exact same strategy code with configurable slippage and commission modeling
- **Real-time notifications**: Telegram bot alerts for trades, daily summaries, risk warnings, and system errors
- **Rich terminal dashboard**: Live position tracking, P&L display, scan results, and portfolio summary using Rich
- **SQLite persistence**: Full audit trail of positions, trades, signals, and daily summaries
- **Paper trading by default**: Live mode requires explicit opt-in with confirmation prompt
- **Market hours awareness**: Respects BIST (10:00-18:00 TRT) and US (16:30-23:00 TRT) trading hours with overlap handling
- **Connection resilience**: Automatic reconnection to IBKR on connection drops with retry logic
- **Cost tracking**: Monitors daily LLM API spending with alerts when costs exceed thresholds

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Scheduler                            │
│  (Runs during BIST: 10:00-18:00 TRT, US: 16:30-23:00 TRT) │
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
  │  (IBKR primary  │                          │
  │   YFinance      │                          │
  │   fallback)     │                          │
  └───────┬────────┘                          │
          │                                   │
  ┌───────▼────────┐                          │
  │  Technical      │                          │
  │  Screener       │  Candidates (~10-20)    │
  │  (RSI, MACD,    ├──────────┐              │
  │   MA, Volume,   │          │              │
  │   Bollinger)    │          │              │
  └────────────────┘          │              │
                       ┌──────▼──────┐        │
                       │  AI Analyst  │        │
                       │  (Claude/GPT │        │
                       │   + News)    │        │
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

### Key Architectural Principle

The screener and risk manager are written as **pure functions** that accept data as input (they never fetch data themselves). This allows the backtesting engine to feed them historical data without any code duplication. The same `screener.py` and `risk.py` code runs in both live trading and backtesting.

---

## Trading Pipeline

Each scan cycle (every 15 minutes by default) executes the following pipeline:

1. **Market hours check** -- Determine which markets (US, BIST, or both) are currently open
2. **Universe building** -- Build/load the tradeable stock list (cached daily), filtering out financial sector stocks and applying liquidity thresholds
3. **Data fetching** -- Fetch historical OHLCV data for all stocks in the universe from IBKR (or YFinance fallback)
4. **Technical screening** -- Run 6 technical indicators on every stock, score candidates, and select the top ~10-20 per market
5. **AI analysis** -- Send each candidate to Claude or GPT with price action, indicators, and news context; receive structured trade recommendations with confidence scores
6. **Risk evaluation** -- Pass every AI-approved signal through 6 risk checks (position size, daily loss, max positions, stop-loss, sector concentration, no duplicates)
7. **Order execution** -- Place bracket orders (entry + stop-loss + take-profit) through IBKR for approved signals
8. **Logging and notifications** -- Record everything to SQLite and CSV, send Telegram alerts, update terminal dashboard

---

## Project Structure

```
auto-trade/
├── config/
│   ├── __init__.py
│   └── settings.py              # All configurable parameters
├── core/
│   ├── __init__.py
│   ├── models.py                # Data classes (Signal, Position, Trade, etc.)
│   ├── connection.py            # IBKR connection manager
│   ├── data.py                  # Market data service (IBKR + YFinance + Tavily)
│   ├── portfolio.py             # SQLite portfolio tracker
│   ├── universe.py              # Stock universe builder
│   ├── screener.py              # Technical indicator screener (pure functions)
│   ├── analyst.py               # LLM-powered trade analyst
│   ├── risk.py                  # Risk manager (pure functions)
│   ├── executor.py              # IBKR order execution
│   ├── scheduler.py             # Main orchestration loop
│   └── logger.py                # Structured logging + Rich dashboard
├── backtest/
│   ├── __init__.py
│   ├── engine.py                # Backtesting engine
│   └── report.py                # Performance metrics and reporting
├── notifications/
│   ├── __init__.py
│   └── telegram.py              # Telegram bot notifications
├── tests/
│   ├── __init__.py
│   ├── test_models.py           # Data class tests
│   ├── test_connection.py       # IBKR connection tests
│   ├── test_data.py             # Market data tests
│   ├── test_portfolio.py        # Database operation tests
│   ├── test_universe.py         # Universe builder tests
│   ├── test_screener.py         # Technical screener tests
│   ├── test_analyst.py          # AI analyst tests
│   ├── test_risk.py             # Risk manager tests
│   └── test_backtest.py         # Backtesting engine tests
├── docs/
│   ├── DESIGN.md                # Full design specification
│   └── IMPLEMENTATION-PLAN.md   # Step-by-step implementation plan
├── data/                        # Runtime data (gitignored)
│   ├── portfolio.db             # SQLite database
│   └── universe_*.json          # Cached stock universes
├── logs/                        # Runtime logs (gitignored)
│   ├── trader_YYYY-MM-DD.log    # Daily log files
│   └── trades_YYYY-MM-DD.csv    # Daily trade journals
├── main.py                      # Entry point
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
├── .gitignore                   # Git ignore rules
├── CLAUDE.md                    # AI assistant instructions
└── README.md                    # This file
```

---

## Prerequisites

### 1. Python 3.11+

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install python3.11 python3.11-venv python3-pip

# macOS (via Homebrew)
brew install python@3.11

# Or via pyenv (any platform)
curl https://pyenv.run | bash
pyenv install 3.11.8 && pyenv global 3.11.8
```

### 2. Interactive Brokers TWS or IB Gateway

You need either TWS (Trader Workstation) or IB Gateway running on the same machine (or accessible via network).

1. **Download TWS**: https://www.interactivebrokers.com/en/trading/tws.php
   - OR download **IB Gateway** (lighter, headless): https://www.interactivebrokers.com/en/trading/ibgateway-stable.php

2. **Create a paper trading account**: https://www.interactivebrokers.com/en/trading/free-trial.php

3. **Configure TWS/Gateway API settings** (required):
   - Open TWS > Edit > Global Configuration > API > Settings
   - **Check** "Enable ActiveX and Socket Clients"
   - **Set Socket Port**: `7497` (paper trading) or `7496` (live trading)
   - **Check** "Allow connections from localhost only" (security)
   - **Uncheck** "Read-Only API" (the system needs to place orders)
   - Click OK/Apply

4. **TWS/Gateway must be running** whenever the trading system is active. The system will fail to start if it cannot connect.

### 3. Telegram Bot (for notifications)

1. Open Telegram and message [@BotFather](https://t.me/BotFather)
2. Send `/newbot` and follow the prompts to create a bot
3. Save the **bot token** (format: `123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11`)
4. Send any message to your new bot, then visit:
   ```
   https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
   ```
5. Find your **chat_id** in the response JSON under `result[0].message.chat.id`

### 4. API Keys

| Service | Purpose | Where to get |
|---------|---------|-------------|
| **Anthropic (Claude)** | AI analyst (primary) | https://console.anthropic.com/ |
| **OpenAI (GPT-4)** | AI analyst (alternative) | https://platform.openai.com/ |
| **Tavily** | News headlines for AI context | https://tavily.com/ (free tier: 1000 searches/month) |

You only need **one** of Anthropic or OpenAI. The system auto-selects based on which API key is configured.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/ermanergoz/auto-trade.git
cd auto-trade

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your actual API keys and configuration
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `ib_insync` | >= 0.9.86 | Interactive Brokers API wrapper |
| `yfinance` | >= 0.2.31 | Fallback historical data (backtesting) |
| `pandas` | >= 2.1.0 | Data manipulation and analysis |
| `pandas-ta` | >= 0.3.14b | Technical indicators (RSI, MACD, Bollinger, etc.) |
| `anthropic` | >= 0.39.0 | Claude API for AI analysis |
| `openai` | >= 1.50.0 | GPT-4 API for AI analysis (alternative) |
| `python-telegram-bot` | >= 20.7 | Telegram notification bot |
| `python-dotenv` | >= 1.0.0 | Environment variable loading |
| `apscheduler` | >= 3.10.4 | Job scheduling for scan cycles |
| `rich` | >= 13.7.0 | Terminal UI, tables, and dashboard |
| `tavily-python` | >= 0.3.0 | News API integration |
| `pytest` | >= 7.4.0 | Test framework |
| `pytest-asyncio` | >= 0.23.0 | Async test support |

---

## Configuration

### Environment Variables (.env)

```bash
# IBKR Connection (no API key needed - connects via socket)
IBKR_HOST=127.0.0.1
IBKR_PORT=7497                    # 7497 = paper trading, 7496 = live
IBKR_CLIENT_ID=1

# AI Model (configure at least one)
ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...           # Uncomment to use GPT-4 instead

# Telegram Notifications
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_CHAT_ID=123456789

# News
TAVILY_API_KEY=tvly-...
```

### Trading Parameters (config/settings.py)

All trading parameters are configured in `config/settings.py`. Key settings:

#### Broker Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `IBKR_HOST` | `127.0.0.1` | TWS/Gateway host |
| `IBKR_PORT` | `7497` | Socket port (7497=paper, 7496=live) |
| `IBKR_CLIENT_ID` | `1` | Client ID for API connection |

#### Market Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `MARKETS` | `["US", "BIST"]` | Active markets |
| `TIMEZONE` | `Europe/Istanbul` | Time zone for market hours |
| `EXCLUDED_SECTORS` | `["Financials"]` | Sectors to exclude from trading |
| `MIN_DAILY_VOLUME` | `100,000` | Minimum average daily volume |
| `MIN_MARKET_CAP` | `$50,000,000` | Minimum market capitalization |

#### Strategy Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `SCAN_INTERVAL_MINUTES` | `15` | Minutes between scan cycles |
| `AI_CONFIDENCE_THRESHOLD` | `70` | Minimum AI confidence to act (0-100) |
| `AI_MODEL` | `claude-sonnet-4-6` | LLM model for analysis |

#### Risk Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_POSITION_SIZE_PCT` | `5.0` | Max portfolio % per position |
| `DAILY_LOSS_LIMIT_PCT` | `2.0` | Daily loss % that halts trading |
| `MAX_OPEN_POSITIONS` | `10` | Maximum concurrent positions |
| `DEFAULT_STOP_LOSS_PCT` | `3.0` | Default stop-loss percentage |
| `MAX_SECTOR_CONCENTRATION_PCT` | `25.0` | Max portfolio % in one sector |

#### Technical Indicator Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `RSI_PERIOD` | `14` | RSI calculation period |
| `RSI_OVERSOLD` | `30` | RSI oversold threshold |
| `RSI_OVERBOUGHT` | `70` | RSI overbought threshold |
| `MACD_FAST` | `12` | MACD fast period |
| `MACD_SLOW` | `26` | MACD slow period |
| `MACD_SIGNAL` | `9` | MACD signal period |
| `MA_FAST` | `5` | Fast moving average period |
| `MA_SLOW` | `20` | Slow moving average period |
| `BB_PERIOD` | `20` | Bollinger Band period |
| `BB_STD` | `2.0` | Bollinger Band standard deviations |
| `VOLUME_SPIKE_MULTIPLIER` | `2.0` | Volume spike threshold (x avg) |

#### Day Trading Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `CLOSE_DAY_TRADES_BEFORE_MARKET_CLOSE` | `True` | Auto-close day trades |
| `CLOSE_MINUTES_BEFORE` | `15` | Minutes before close to start closing |

---

## Usage

### Basic Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Paper trading - single scan cycle (for testing)
python main.py --once

# Paper trading - continuous during market hours
python main.py

# Trade only US market
python main.py --market us

# Trade only BIST market
python main.py --market bist

# Dry-run mode (full pipeline, logs what it would trade, no actual orders)
python main.py --mode dry-run

# Backtest mode (no IBKR connection needed)
python main.py --mode backtest

# Backtest with specific tickers and date range
python main.py --mode backtest --backtest-tickers AAPL MSFT GOOGL --backtest-start 2025-01-01 --backtest-end 2025-06-30

# Backtest with custom initial capital
python main.py --mode backtest --capital 50000

# Live trading (requires explicit confirmation)
python main.py --mode live
```

### Command-Line Arguments

| Argument | Values | Default | Description |
|----------|--------|---------|-------------|
| `--mode` | `paper`, `live`, `backtest`, `dry-run` | `paper` | Trading mode |
| `--market` | `us`, `bist`, `all` | `all` | Markets to trade |
| `--once` | flag | off | Run single scan then exit |
| `--backtest-tickers` | space-separated tickers | default list | Tickers for backtesting |
| `--backtest-start` | `YYYY-MM-DD` | 1 year ago | Backtest start date |
| `--backtest-end` | `YYYY-MM-DD` | today | Backtest end date |
| `--capital` | float | `100000` | Initial capital for backtesting |

---

## Trading Modes

### Paper Mode (default)

Connects to IBKR paper trading account (port 7497). Executes real orders on the paper account, which simulates market conditions without using real money. This is the recommended mode for testing and tuning.

```bash
python main.py                    # continuous
python main.py --once             # single cycle
```

### Live Mode

Connects to IBKR live account (port 7496). **Uses real money.** Requires explicit confirmation at startup.

```bash
python main.py --mode live
# Prompts: Type 'CONFIRM LIVE' to proceed
```

### Backtest Mode

Replays historical data through the exact same strategy code. Does not connect to IBKR. Downloads data via YFinance.

```bash
python main.py --mode backtest
python main.py --mode backtest --backtest-tickers AAPL TSLA --backtest-start 2025-01-01
```

### Dry-Run Mode

Runs the full pipeline (screener, AI analyst, risk manager) but logs what it **would** trade without placing any orders. Useful for observing the system's decisions in real-time without execution.

```bash
python main.py --mode dry-run
```

---

## Technical Screener

The screener runs 6 technical indicators on every stock in the universe and scores candidates based on how many patterns trigger simultaneously.

### Indicators

| Indicator | Bullish Signal | Bearish Signal |
|-----------|---------------|----------------|
| **RSI(14)** | RSI < 30 (oversold) | RSI > 70 (overbought) |
| **MACD(12,26,9)** | MACD crosses above signal line | MACD crosses below signal line |
| **MA Crossover (5,20)** | MA5 crosses above MA20 (golden cross) | MA5 crosses below MA20 (death cross) |
| **Volume Spike** | Volume > 2x 20-day average (confirms moves) | Volume > 2x 20-day average (confirms moves) |
| **Bollinger Bands (20,2)** | Price below lower band (oversold) | Price above upper band (overbought) |
| **Support/Resistance** | Price within 2% of support level | Price within 2% of resistance level |

### Scoring

Each triggered indicator contributes to the candidate's score. The screener counts buy signals vs sell signals, determines the dominant direction, and calculates a confidence score (0-100). Stocks are ranked by score and the top candidates (typically 10-20 per market) are passed to the AI analyst.

Stop-loss and take-profit levels are calculated using ATR (Average True Range) for volatility-adjusted sizing.

---

## AI Analyst

The AI analyst performs deep analysis on each screener candidate using Claude or GPT-4.

### Context Provided to the LLM

For each candidate, the analyst gathers:
- **5-day price action**: Recent OHLCV data showing price movement
- **Technical indicator values**: All computed indicator values from the screener
- **News headlines**: Top 5 recent news articles via Tavily API
- **Sector context**: Current sector performance

### Structured Output

The LLM returns a structured JSON response via tool_use (Claude) or function_calling (GPT):

```json
{
  "action": "buy",
  "confidence": 82,
  "entry_price": 150.25,
  "stop_loss": 145.75,
  "take_profit": 160.00,
  "reasoning": "Strong bullish MACD crossover with volume confirmation..."
}
```

### Filtering

Only signals with `confidence >= AI_CONFIDENCE_THRESHOLD` (default: 70) are forwarded to the risk manager. Lower-confidence signals are logged but not acted upon.

### Cost Control

- The two-stage pipeline keeps daily LLM costs under ~$1 (only ~20 candidates per market analyzed)
- Token usage is tracked per call
- An alert triggers if daily AI spending exceeds $2

---

## Risk Management

Every trade must pass through **all 6 risk checks** before execution. If any check fails, the trade is rejected and the reasons are logged.

### Risk Checks

| Check | Rule | Default |
|-------|------|---------|
| **Position Size** | Single position cannot exceed X% of portfolio value | 5% |
| **Daily Loss Limit** | Halt all trading if daily P&L drops below -X% | 2% |
| **Max Open Positions** | Cannot exceed N concurrent open positions | 10 |
| **Stop-Loss Required** | Every trade must have a valid stop-loss order | Required |
| **Sector Concentration** | No sector can exceed X% of total portfolio | 25% |
| **No Duplicates** | Cannot open a second position in an already-held stock | Enforced |

### Position Sizing

Position size is calculated using the more conservative of two methods:
1. **Max position method**: `portfolio_value * MAX_POSITION_SIZE_PCT / entry_price`
2. **Risk-based method**: `(portfolio_value * 1%) / (entry_price - stop_loss)` -- limits risk to 1% of portfolio per trade using stop-loss distance

---

## Backtesting

The backtesting engine replays historical data through the **exact same** screener and risk manager code used in live trading.

### How It Works

1. Downloads 1 year of historical data for all specified tickers via YFinance
2. Skips the first 60 trading days for indicator warmup (moving averages, etc.)
3. Iterates day-by-day:
   - Checks stop-loss and take-profit exits for open positions
   - Builds historical data window up to the current day (**no look-ahead bias**)
   - Runs the technical screener on the windowed data
   - Passes candidates through the risk manager with simulated portfolio state
   - Simulates order execution with configurable slippage and commission
4. Closes all remaining positions at the last bar
5. Calculates comprehensive performance metrics

### Backtest Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| Slippage | 0.1% | Simulated execution slippage |
| Commission | $1/trade | Per-trade commission cost |
| Initial capital | $100,000 | Starting portfolio value |
| Warmup period | 60 days | Days skipped for indicator stabilization |

### Performance Metrics

The backtest report includes:

| Metric | Description |
|--------|-------------|
| **Total Return** | Overall portfolio return percentage |
| **Annualized Return** | Return normalized to a yearly basis |
| **Sharpe Ratio** | Risk-adjusted return (assuming 5% risk-free rate, 252 trading days) |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Win Rate** | Percentage of profitable trades |
| **Profit Factor** | Gross profit divided by gross loss |
| **Average Trade Duration** | Mean holding period |
| **Best/Worst Trade** | Largest single gain and loss |
| **Total Trades** | Number of round-trip trades executed |

### Example

```bash
# Backtest 10 tech stocks over 6 months
python main.py --mode backtest \
  --backtest-tickers AAPL MSFT GOOGL AMZN NVDA TSLA META AMD NFLX CRM \
  --backtest-start 2025-07-01 \
  --backtest-end 2025-12-31 \
  --capital 50000
```

---

## Notifications

The Telegram bot sends real-time alerts for all trading activity.

### Notification Types

| Event | Content |
|-------|---------|
| **Trade Opened** | Ticker, action (BUY/SELL), quantity, price, stop-loss, take-profit, confidence score, AI reasoning |
| **Trade Closed** | Ticker, exit price, P&L percentage, profit/loss amount |
| **Daily Summary** | End-of-day report with portfolio value, daily P&L, number of trades, open positions |
| **Risk Warning** | Alerts when daily loss limit is approached, positions rejected by risk manager |
| **Scan Summary** | Per-cycle summary: candidates found, AI-approved signals, risk-approved trades, orders placed |
| **System Error** | Connection failures, API errors, unexpected exceptions |

### Setup

1. Create a Telegram bot via @BotFather (see [Prerequisites](#3-telegram-bot-for-notifications))
2. Add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` to your `.env` file
3. Notifications are fire-and-forget -- Telegram failures will not crash the trading system

---

## Database Schema

The system uses SQLite (stored at `data/portfolio.db`) with WAL mode enabled for concurrent read/write performance.

### Tables

#### `positions` -- Open Positions
| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment ID |
| `ticker` | TEXT | Stock ticker symbol |
| `exchange` | TEXT | Exchange (US/BIST) |
| `quantity` | INTEGER | Number of shares |
| `entry_price` | REAL | Entry price per share |
| `entry_time` | TEXT | ISO timestamp |
| `stop_loss` | REAL | Stop-loss price |
| `take_profit` | REAL | Take-profit price |
| `trade_type` | TEXT | DAY or SWING |
| `sector` | TEXT | Stock sector |

#### `trades` -- Completed Trades
| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment ID |
| `ticker` | TEXT | Stock ticker symbol |
| `exchange` | TEXT | Exchange (US/BIST) |
| `quantity` | INTEGER | Number of shares |
| `entry_price` | REAL | Entry price per share |
| `exit_price` | REAL | Exit price per share |
| `entry_time` | TEXT | Entry ISO timestamp |
| `exit_time` | TEXT | Exit ISO timestamp |
| `pnl` | REAL | Profit/Loss amount |
| `trade_type` | TEXT | DAY or SWING |
| `sector` | TEXT | Stock sector |
| `reasoning` | TEXT | AI analysis reasoning |

#### `daily_summary` -- End-of-Day Snapshots
| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment ID |
| `date` | TEXT | Date (YYYY-MM-DD) |
| `portfolio_value` | REAL | Total portfolio value |
| `daily_pnl` | REAL | Day's profit/loss |
| `num_trades` | INTEGER | Trades executed |
| `num_positions` | INTEGER | Open positions at EOD |

#### `signals` -- Signal Audit Trail
| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment ID |
| `timestamp` | TEXT | ISO timestamp |
| `ticker` | TEXT | Stock ticker symbol |
| `exchange` | TEXT | Exchange (US/BIST) |
| `action` | TEXT | BUY/SELL/HOLD |
| `confidence` | REAL | Confidence score (0-100) |
| `entry_price` | REAL | Suggested entry price |
| `stop_loss` | REAL | Suggested stop-loss |
| `take_profit` | REAL | Suggested take-profit |
| `source` | TEXT | Signal source (screener/ai) |
| `reasoning` | TEXT | Analysis reasoning |

---

## Testing

The project includes tests for all major modules.

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_screener.py -v
pytest tests/test_risk.py -v
pytest tests/test_backtest.py -v

# Run with coverage
pytest tests/ --cov=core --cov=backtest --cov=notifications
```

### Test Modules

| Module | Tests |
|--------|-------|
| `test_models.py` | Data class construction, properties, P&L calculations |
| `test_connection.py` | IBKR connection, reconnection, contract creation |
| `test_data.py` | Historical data fetching, caching, news API |
| `test_portfolio.py` | Database CRUD operations, position lifecycle |
| `test_universe.py` | Universe building, financial sector filtering, caching |
| `test_screener.py` | Technical indicator calculations, scoring, signal generation |
| `test_analyst.py` | LLM integration, response validation, cost tracking |
| `test_risk.py` | All 6 risk checks, position sizing calculations |
| `test_backtest.py` | Backtesting engine, simulated portfolio, no look-ahead bias |

---

## Development Guide

### Adding a New Technical Indicator

1. Add the indicator function to `core/screener.py` following the existing pattern:

```python
def check_my_indicator(df: pd.DataFrame) -> dict | None:
    """Check for my custom pattern. Returns signal dict or None."""
    # Calculate indicator using pandas-ta
    indicator_values = df.ta.my_indicator()

    if bullish_condition:
        return {
            "name": "my_indicator",
            "action": Action.BUY,
            "value": indicator_values.iloc[-1],
            "description": "Bullish signal detected",
        }
    return None
```

2. Add the new check to the `analyze_stock()` function's indicator list
3. Add corresponding settings to `config/settings.py`
4. Write tests in `tests/test_screener.py`

### Adding a New Risk Check

1. Add the check function to `core/risk.py`:

```python
def check_my_rule(signal: Signal, positions: list[Position], portfolio_value: float) -> tuple[bool, str]:
    """Returns (passed, reason)."""
    if violates_rule:
        return False, "Rejected: reason"
    return True, ""
```

2. Add the check to the `evaluate()` function's check list
3. Add related settings to `config/settings.py`
4. Write tests in `tests/test_risk.py`

### Adding a New Notification Channel

1. Create a new module in `notifications/` (e.g., `notifications/slack.py`)
2. Implement the same interface as `telegram.py`: `notify_trade()`, `notify_daily_summary()`, etc.
3. Register the notification channel in `core/scheduler.py`

### Data Flow

```
main.py
  └── core/scheduler.py         # Orchestrates the pipeline
       ├── core/universe.py     # Builds stock list
       ├── core/data.py         # Fetches market data
       ├── core/screener.py     # Screens candidates (pure function)
       ├── core/analyst.py      # AI analysis
       ├── core/risk.py         # Risk evaluation (pure function)
       ├── core/executor.py     # Places orders
       ├── core/portfolio.py    # Records trades
       ├── core/logger.py       # Logs and dashboard
       └── notifications/       # Alerts
```

---

## Safety Features

This system is designed with multiple layers of safety:

1. **Paper trading by default** -- The default mode connects to IBKR paper trading (port 7497). No real money is used unless you explicitly switch to live mode.

2. **Live mode confirmation** -- Starting in live mode requires typing `CONFIRM LIVE` at the prompt. There is no way to accidentally enter live mode.

3. **Mandatory stop-losses** -- Every trade placed through the system has a stop-loss order attached via IBKR bracket orders. Trades without valid stop-losses are rejected by the risk manager.

4. **Daily loss limit** -- If the portfolio's daily P&L drops below -2% (configurable), all trading is automatically halted for the remainder of the day.

5. **Position size limits** -- No single position can exceed 5% of portfolio value (configurable). Position sizing also accounts for stop-loss distance to limit risk to 1% of portfolio per trade.

6. **Sector concentration limits** -- No single sector can exceed 25% of the portfolio, preventing over-concentration.

7. **Duplicate position prevention** -- The system will not open a second position in a stock that is already held.

8. **Day trade auto-close** -- Day trade positions are automatically closed 15 minutes before market close to prevent unintended overnight exposure.

9. **Dry-run mode** -- Run the full pipeline and observe decisions without any orders being placed.

10. **Full audit trail** -- Every signal, trade, and risk decision is logged to SQLite and CSV for review.

11. **Financial sector exclusion** -- Banks, insurance companies, and lending institutions are permanently excluded from the trading universe.

---

## Troubleshooting

### Connection Issues

**"Connection failed" on startup**
- Ensure TWS or IB Gateway is running
- Verify API connections are enabled in TWS settings (Edit > Global Configuration > API > Settings)
- Check the socket port matches your `.env` (7497 for paper, 7496 for live)
- Ensure "Allow connections from localhost only" is checked and you're connecting from localhost

**"Connection dropped" during operation**
- The system automatically attempts reconnection (3 retries, 5s delay)
- TWS/Gateway may drop connections after inactivity -- this is normal IBKR behavior
- If TWS was restarted, the system will reconnect on the next scan cycle

### IBKR Data Issues

**"Pacing violation" errors**
- IBKR limits historical data requests to 60 per 10 minutes
- The system batches and caches requests, but very large universes may hit this limit
- Reduce universe size or increase `SCAN_INTERVAL_MINUTES`

**BIST tickers not resolving**
- BIST stocks use the "BIST" exchange with "TRY" currency in IBKR
- The `.IS` suffix is only used for YFinance (backtest fallback)
- Some tickers may need `qualifyContracts()` -- the connection module handles this automatically

### AI Analyst Issues

**"No API key configured"**
- Set either `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your `.env` file
- The system auto-selects based on which key is present (Anthropic takes priority)

**High API costs**
- The system tracks daily spending and alerts at $2
- Reduce `AI_CONFIDENCE_THRESHOLD` to filter more aggressively (fewer candidates reach AI)
- Or increase the screener score threshold in settings

### Backtest Issues

**"No data available" for tickers**
- YFinance may not have data for all tickers, especially BIST stocks
- Check the ticker symbol (BIST tickers auto-append `.IS` for YFinance)
- Try a different date range -- very recent data may have a delay

**Unrealistic backtest results**
- Check for look-ahead bias (should not exist with proper screener windowing)
- Increase slippage to model real execution costs more accurately
- Note: backtests without AI analysis (default) only use technical signals

### General Issues

**Database locked**
- SQLite WAL mode handles most concurrent access, but ensure only one instance is running
- Delete `data/portfolio.db` to reset (all history will be lost)

**Import errors**
- Ensure virtual environment is activated: `source .venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

---

## Market Hours

All times are in Turkey time (TRT / Europe/Istanbul):

| Market | Open | Close | Notes |
|--------|------|-------|-------|
| **BIST** | 10:00 | 18:00 | Turkish equities |
| **US (NYSE/NASDAQ)** | 16:30 | 23:00 | Adjusted for TRT |
| **Overlap** | 16:30 | 18:00 | Both markets active |

The scheduler automatically detects which markets are open and only scans active markets. Weekends and holidays are skipped.

---

## License

This project is for personal use. All rights reserved.

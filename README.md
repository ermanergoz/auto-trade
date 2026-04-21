# Auto Trade

An automated stock trading system that day-trades and swing-trades US (NYSE/NASDAQ) equities through Interactive Brokers. The system uses a two-stage pipeline: a fast technical screener filters hundreds of stocks, then an AI analyst (Gemini primary with Ollama as automatic fallback) performs deep analysis on all qualifying candidates. A risk manager gates every trade before execution through IBKR.

Financial sector stocks (banks, insurance, lending companies) and defense/military stocks (weapons, ammunition, combat systems) are automatically excluded from all trading.

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

- **US market focus**: Trades US (NYSE/NASDAQ) equities through IBKR with 10 different scanner types for broad market coverage
- **Two-stage screening pipeline**: Technical screener (fast, free) filters hundreds of stocks, then AI analyst performs deep analysis on all qualifying candidates
- **AI-powered analysis**: Gemini (primary) for fast, capable cloud-based reasoning; Ollama + Qwen 2.5 7B (fallback) for resilience when Gemini is unavailable, rate-limited, or unconfigured
- **Comprehensive risk management**: 14 risk checks including position sizing, daily loss limits, sector concentration limits, mandatory stop-losses, duplicate position prevention, defense/financial sector exclusion, circuit breaker, and analyst consensus. Exit signals bypass discipline checks so positions can always be closed
- **Bracket order execution**: Automatic stop-loss and take-profit orders attached to every trade via IBKR bracket orders
- **Day and swing trading**: Automatic end-of-day position closing for day trades, trailing stops for swing trades
- **Backtesting engine**: Replay historical data through the exact same strategy code with configurable slippage and commission modeling
- **Real-time notifications**: Telegram bot alerts for trades, daily summaries, risk warnings, and system errors
- **Rich terminal dashboard**: Live position tracking, P&L display, scan results, and portfolio summary using Rich
- **SQLite persistence**: Full audit trail of positions, trades, signals, and daily summaries
- **Paper trading by default**: Live mode requires explicit opt-in with confirmation prompt
- **Market hours awareness**: Respects US (16:30-23:00 TRT) trading hours
- **Connection resilience**: Automatic reconnection to IBKR on connection drops with retry logic; realtime subscriptions are properly cleaned up on disconnect to prevent callback leaks
- **IBC Watchdog mode**: Optional auto-start of IB Gateway with automatic reconnection after daily restarts via IBC
- **GTC bracket orders**: Orders placed outside market hours persist and execute at market open
- **Cost-efficient AI**: Gemini Flash-Lite is cheap-to-free on light workloads; on any Gemini error or quota depletion the bot transparently falls back to local Ollama (zero cost)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Scheduler                            │
│        (Runs during US: 16:30-23:00 TRT)                    │
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
  │  Screener       │  All candidates          │
  │  (RSI, MACD,    ├──────────┐              │
  │   MA, Volume,   │          │              │
  │   Bollinger)    │          │              │
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

### Key Architectural Principle

The screener and risk manager are written as **pure functions** that accept data as input (they never fetch data themselves). This allows the backtesting engine to feed them historical data without any code duplication. The same `screener.py` and `risk.py` code runs in both live trading and backtesting.

---

## Trading Pipeline

Each scan cycle (every 15 minutes by default) executes the following pipeline:

1. **Stale order re-evaluation** -- Check unfilled limit orders older than 24 hours, re-run the screener on each, and cancel orders that no longer pass technical screening
2. **Market hours check** -- Determine if the US market is currently open
3. **Universe building** -- Build/load the tradeable stock list (cached daily) using 10 IBKR scanner types, enrich each stock with sector data via a 4-tier fallback chain (IBKR contract details -> yfinance -> Gemini -> Ollama), classify ETFs by category (equity ETFs kept, bond/leveraged/commodity ETFs excluded), then filter out financial sector stocks and apply liquidity thresholds. Typical result: ~200-350 unique stocks
4. **Data fetching** -- Fetch historical OHLCV data for all stocks in the universe from IBKR (or YFinance fallback)
5. **Technical screening** -- Run 6 technical indicators on every stock, score candidates, inject sector data from the universe into each candidate's indicator values (so risk checks can enforce sector limits), and pass all qualifying stocks (above min_score) to AI analysis
6. **AI analysis** -- Send each candidate to the local LLM (via Ollama) with price action, indicators, and news context; receive structured trade recommendations with confidence scores
7. **Risk evaluation** -- Pass every AI-approved signal through 14 risk checks (short selling block, position size, daily loss, cumulative risk, max positions, stop-loss, sector concentration, no duplicates, excluded sector, circuit breaker, and for new entries only: risk/reward, anti-momentum, trend confirmation, analyst consensus). Exit signals (selling a held position) skip the discipline checks so positions can always be closed
8. **Order execution** -- Place bracket orders (entry + stop-loss + take-profit) through IBKR for approved signals
9. **Logging and notifications** -- Record everything to SQLite and CSV, send Telegram alerts, update terminal dashboard

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
│   ├── analyst.py               # LLM-powered trade analyst (Ollama)
│   ├── risk.py                  # Risk manager (pure functions)
│   ├── executor.py              # IBKR order execution
│   ├── scheduler.py             # Main orchestration loop
│   ├── logger.py                # Structured logging + Rich dashboard
│   └── state.py                 # Shared mutable state (shutdown flag)
├── backtest/
│   ├── __init__.py
│   ├── engine.py                # Backtesting engine
│   └── report.py                # Performance metrics and reporting
├── notifications/
│   ├── __init__.py
│   └── telegram.py              # Telegram bot notifications
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Shared test fixtures (make_signal, make_position)
│   ├── test_models.py           # Data class tests
│   ├── test_connection.py       # IBKR connection tests
│   ├── test_data.py             # Market data tests
│   ├── test_portfolio.py        # Database operation tests
│   ├── test_universe.py         # Universe builder tests
│   ├── test_screener.py         # Technical screener tests
│   ├── test_analyst.py          # AI analyst tests
│   ├── test_risk.py             # Risk manager tests
│   ├── test_scheduler.py        # Streaming pipeline tests
│   ├── test_telegram.py         # Telegram notification tests
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

4. **TWS/Gateway must be running** whenever the trading system is active. The system will fail to start if it cannot connect (unless using `--watchdog` mode, which starts the gateway automatically).

### 3. IBC (Optional — for unattended operation)

[IBC](https://github.com/IbcAlpha/IBC) automates IB Gateway login and handles daily restarts. Required only for `--watchdog` mode.

```bash
# Download and install to ~/ibc
curl -sL -o /tmp/IBCLinux.zip https://github.com/IbcAlpha/IBC/releases/latest/download/IBCLinux-3.23.0.zip
mkdir -p ~/ibc && unzip -o /tmp/IBCLinux.zip -d ~/ibc
chmod +x ~/ibc/scripts/*.sh ~/ibc/*.sh
```

Edit `~/ibc/config.ini` and set your IBKR credentials (`IbLoginId`, `IbPassword`), trading mode, and auto-restart time.

### 4. Telegram Bot (for notifications)

1. Open Telegram and message [@BotFather](https://t.me/BotFather)
2. Send `/newbot` and follow the prompts to create a bot
3. Save the **bot token** (format: `123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11`)
4. Send any message to your new bot, then visit:
   ```
   https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
   ```
5. Find your **chat_id** in the response JSON under `result[0].message.chat.id`

### 5. LLM Providers

The AI analyst routes through **Gemini** (primary) with **Ollama** as an automatic fallback. At least one of them must be reachable:

- **Gemini (recommended)** — set `GEMINI_API_KEY` in `.env`. Get a key at https://aistudio.google.com/apikey. On any Gemini error, rate limit, or credit exhaustion the bot transparently continues on Ollama for the rest of the process.
- **Ollama (fallback / legacy)** — runs locally, no API key. Install and pull the model:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:8b
```

Ollama must be running (as a system service or `ollama serve`) whenever it may be used as a fallback. Set `AI_PROVIDER=ollama` in `.env` to skip Gemini entirely.

### 6. API Keys (Optional)

| Service | Purpose | Where to get |
|---------|---------|-------------|
| **Gemini** | Primary LLM for trade analysis | https://aistudio.google.com/apikey |
| **Tavily** | News headlines for AI context | https://tavily.com/ (free tier: 1000 searches/month) |

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
# Edit .env with your configuration (Telegram token, Tavily key)
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `ib_insync` | >= 0.9.86 | Interactive Brokers API wrapper |
| `yfinance` | >= 0.2.31 | Fallback historical data (backtesting) |
| `pandas` | >= 2.1.0 | Data manipulation and analysis |
| `pandas-ta` | >= 0.3.14b | Technical indicators (RSI, MACD, Bollinger, etc.) |
| `python-telegram-bot` | >= 20.7 | Telegram notification bot |
| `python-dotenv` | >= 1.0.0 | Environment variable loading |
| `rich` | >= 13.7.0 | Terminal UI, tables, and Rich dashboard |
| `tavily-python` | >= 0.3.0 | News API integration |
| `pytest` | >= 7.4.0 | Test framework |
| `pytest-asyncio` | >= 0.23.0 | Async test support |

---

## Configuration

### Environment Variables (.env)

```bash
# IBKR Connection (no API key needed - connects via socket to running TWS/Gateway)
IBKR_HOST=127.0.0.1
IBKR_PORT=7497                    # 7497 = paper trading, 7496 = live
IBKR_CLIENT_ID=1

# LLM provider — "gemini" (primary, auto-falls back to Ollama on error) or "ollama"
AI_PROVIDER=gemini

# Gemini (primary) — leave GEMINI_API_KEY blank to skip Gemini entirely
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash-lite
GEMINI_HOST=https://generativelanguage.googleapis.com

# Ollama (fallback) — no API key needed
AI_MODEL=qwen3:8b
OLLAMA_HOST=http://localhost:11434

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

#### IBC Settings (for `--watchdog` mode)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `IBC_PATH` | `~/ibc` | Path to IBC installation |
| `IBC_INI` | `~/ibc/config.ini` | Path to IBC configuration file |
| `TWS_PATH` | `~/Jts` | Path to TWS/Gateway installation |
| `TWS_VERSION` | `1037` | IB Gateway major version number |
| `IBC_USERID` | (empty) | IBKR username (overrides config.ini) |
| `IBC_PASSWORD` | (empty) | IBKR password (overrides config.ini) |

#### Market Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `MARKETS` | `["US"]` | Active markets |
| `TIMEZONE` | `Europe/Istanbul` | Time zone for market hours |
| `EXCLUDED_SECTORS` | `["Financials"]` | Sectors to exclude from trading |
| `MIN_DAILY_VOLUME` | `100,000` | Minimum average daily volume |
| `MIN_MARKET_CAP` | `$50,000,000` | Minimum market capitalization |

#### Strategy Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `SCAN_INTERVAL_MINUTES` | `15` | Minutes between scan cycles |
| `AI_CONFIDENCE_THRESHOLD` | `65` | Minimum AI confidence to act (0-100) |
| `AI_MAX_CANDIDATES` | `0` | Max candidates sent to AI per cycle (0 = unlimited) |
| `AI_PROVIDER` | `gemini` | Primary LLM provider. `gemini` falls back to Ollama on error; `ollama` skips Gemini entirely |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | Gemini model; `gemini-2.5-flash` is smarter but occasionally rate-limited |
| `GEMINI_HOST` | `https://generativelanguage.googleapis.com` | Gemini API host (override for testing) |
| `AI_MODEL` | `qwen3:8b` | Ollama fallback model |

#### Risk Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_POSITION_SIZE_PCT` | `5.0` | Max portfolio % per position |
| `DAILY_LOSS_LIMIT_PCT` | `2.0` | Daily loss % that halts trading |
| `MAX_OPEN_POSITIONS` | `10` | Maximum concurrent positions |
| `DEFAULT_STOP_LOSS_PCT` | `3.0` | Default stop-loss percentage |
| `MAX_SECTOR_CONCENTRATION_PCT` | `25.0` | Max portfolio % in one sector |
| `ALLOW_SHORT_SELLING` | `False` | Allow sell signals for stocks not held |
| `CIRCUIT_BREAKER_LOSSES` | `3` | Consecutive losses to pause trading |
| `CIRCUIT_BREAKER_WINDOW_MIN` | `60` | Time window (minutes) for circuit breaker |
| `STALE_ORDER_MINUTES` | `1440` | Re-screen unfilled orders after N minutes (24h) |

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

# Dry-run mode (full pipeline, logs what it would trade, no actual orders)
python main.py --mode dry-run

# Backtest mode (no IBKR connection needed)
python main.py --mode backtest

# Backtest with specific tickers and date range
python main.py --mode backtest --backtest-tickers AAPL MSFT GOOGL --backtest-start 2025-01-01 --backtest-end 2025-06-30

# Backtest with custom initial capital
python main.py --mode backtest --capital 50000

# Force scan outside market hours (orders queue for next open as GTC)
python main.py --force

# Watchdog mode — IBC auto-starts gateway and reconnects after daily restarts
python main.py --watchdog

# Combine flags
python main.py --watchdog --force

# Live trading (requires explicit confirmation)
python main.py --mode live
```

### Command-Line Arguments

| Argument | Values | Default | Description |
|----------|--------|---------|-------------|
| `--mode` | `paper`, `live`, `backtest`, `dry-run` | `paper` | Trading mode |
| `--market` | `us`, `all` | `all` | Markets to trade |
| `--once` | flag | off | Run single scan then exit |
| `--force` | flag | off | Bypass market hours check (GTC orders queue for next open) |
| `--watchdog` | flag | off | Use IBC to auto-start gateway and reconnect on restarts |
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
| **Support/Resistance** | Price within 2% of support level (only if intraday low hasn't breached support) | Price within 2% of resistance level |

### Scoring

Each triggered indicator contributes to the candidate's score using configurable weights (`INDICATOR_WEIGHTS` in settings). The screener counts buy signals vs sell signals weighted by indicator importance, determines the dominant direction, and calculates a confidence score (0-100). Opposing signals actively reduce the score (net_score = direction - opposing), preventing conflicting indicators from producing falsely confident signals. All stocks scoring above the minimum threshold (default: 15) are passed to the AI analyst — there is no hard cap on the number of candidates.

Indicator weights can be tuned per-indicator (e.g., `{"RSI": 2.0, "MACD": 0.5}`) to emphasize indicators with higher predictive power. Weights default to 1.0 (equal weighting). Setting a weight to 0.0 effectively disables that indicator's contribution to the score.

Stop-loss and take-profit levels are calculated using ATR (Average True Range) for volatility-adjusted sizing.

### Extension Guard

Before scoring, the screener drops any ticker whose latest close sits more than `MAX_EXTENSION_OVER_MA20_PCT` (default 20%) above its 20-day simple moving average. This prevents parabolic breakouts (e.g. XNDU ripping $9 → $32 in a handful of sessions) from reaching the AI analyst, where confluent BUY indicators could otherwise override into a late entry. Set the config to `0` or negative to disable. The backtest exposes the same threshold as `BacktestConfig.max_extension_pct`; `scripts/sweep_extension_pct.py` runs a parameter sweep for tuning.

---

## AI Analyst

The AI analyst performs deep analysis on each screener candidate. It routes through **Gemini** (`gemini-2.5-flash-lite` by default) when `GEMINI_API_KEY` is set; on any Gemini transport failure (HTTP 5xx, network error, auth failure, or credits depleted) it transparently falls back to a local Ollama model (Qwen 2.5 7B by default) for this call. Permanent failures (invalid key, depleted prepayment credits) latch a process-lifetime flag so Gemini is skipped entirely until the process restarts — mirroring the Tavily→yfinance news-fallback pattern elsewhere in this codebase.

### Context Provided to the LLM

For each candidate, the analyst gathers:
- **5-day price action**: Recent OHLCV data showing price movement
- **Technical indicator values**: All computed indicator values from the screener
- **News headlines**: Top 5 recent news articles via Tavily API (primary), falling back to yfinance when Tavily errors, is unconfigured, or returns nothing. Candidates with zero headlines from both sources are dropped before the LLM call to avoid wasted compute on tickers with no external signal
- **Macro/political headlines**: Top 5 broad market political, regulatory, and macroeconomic headlines via Tavily (shared across all candidates, fetched once per scan cycle)
- **Sector context**: Current sector performance

### Structured Output

The LLM returns a structured JSON response:

```json
{
  "action": "buy",
  "confidence": 82,
  "entry_price": 150.25,
  "stop_loss": 145.75,
  "take_profit": 160.00,
  "reasoning": "Strong bullish MACD crossover with volume confirmation...",
  "trade_type": "swing"
}
```

Response validation checks all required fields including `trade_type` (must be "day" or "swing"). JSON parse errors (`KeyError`, `JSONDecodeError`) from either provider are caught with specific log messages. Content-level failures (malformed envelope, invalid JSON) trigger provider-internal retries up to 3 times; transport failures on Gemini fall straight through to Ollama without wasted retries.

### Filtering

Only signals with `confidence >= AI_CONFIDENCE_THRESHOLD` (default: 65) are forwarded to the risk manager. Lower-confidence signals are logged but not acted upon.

### Cost

Gemini Flash-Lite is cheap-to-free on light workloads (see Google's pricing page and free-tier limits). Ollama is free — it runs locally. Per-candidate Gemini latency is typically ~2-5 seconds; Ollama takes ~30-60 seconds on CPU hardware, 5-10x faster on a GPU. Use `get_daily_token_usage()` to inspect per-provider input/output token counters.

---

## Risk Management

Every trade must pass through **all 15 risk checks** before execution (10 core checks + 5 discipline checks for new entries only). If any check fails, the trade is rejected and the reasons are logged.

### Risk Checks

| Check | Rule | Default |
|-------|------|---------|
| **Short Selling Block** | Sell signals for stocks not currently held are blocked | Blocked (configurable) |
| **Position Size** | Single position cannot exceed X% of portfolio value | 50% |
| **Daily Loss Limit** | Halt all trading if daily P&L drops below -X% | 10% |
| **Max Open Positions** | Cannot exceed N concurrent open positions | 3 |
| **Stop-Loss Required** | Every trade must have a valid stop-loss order | Required |
| **Sector Concentration** | No sector can exceed X% of total portfolio | 50% |
| **No Duplicates** | Cannot open a second position in an already-held stock | Enforced |
| **Excluded Sector** | Block financial sector, defense/military stocks, and explicitly excluded tickers | Enforced |
| **Circuit Breaker** | Pause trading after N consecutive losses within a time window | 3 losses / 60 min |
| **Risk/Reward Ratio** | Take-profit/stop-loss ratio must exceed minimum (new entries only) | 1.5:1 |
| **Anti-Momentum** | Reject if price already moved >X% from signal entry (new entries only) | 8% |
| **Trend Confirmation** | Moving averages must align with trade direction (new entries only) | MA5 > MA10 > MA20 |
| **Analyst Consensus** | Block BUY when analysts rate sell/strong sell (new entries only) | Enabled |
| **Correlation Cap** | Reject a new entry whose daily-return correlation with any open position exceeds threshold (new entries only) | 0.7 |

> **Exit signals** (selling a held long, or buying back a held short) skip the discipline checks (risk/reward, anti-momentum, trend confirmation, analyst consensus, correlation cap) so positions can always be closed regardless of market conditions. This prevents the dangerous situation of being trapped in a losing position.

> **Note**: These defaults are tuned for a small account ($500). For larger accounts, see [docs/RISK-TUNING.md](docs/RISK-TUNING.md) for recommended values at different account sizes.

### Stale Order Re-evaluation

Unfilled limit orders are re-evaluated at the start of every scan cycle. If an order has been pending longer than `STALE_ORDER_MINUTES` (default: 24 hours), the system fetches fresh data and re-runs the technical screener on that stock. Orders that no longer pass screening are automatically cancelled (cancelling the parent entry order also cancels its attached stop-loss and take-profit children). Orders that still pass are kept alive. Telegram notifications are sent for each cancellation. In dry-run mode, the cancelled-order counter only increments for actual cancellations.

Order placement timestamps are persisted in the `pending_orders` database table to survive IBKR reconnections (the `ib_insync` trade log resets on every reconnect, which would otherwise make all orders appear brand new). Records are cleaned up automatically when orders fill or are cancelled.

### Position Sizing

Position size is calculated using the more conservative of two methods:
1. **Max position method**: `portfolio_value * MAX_POSITION_SIZE_PCT / entry_price`
2. **Risk-based method**: `(portfolio_value * RISK_PER_TRADE_PCT%) / (entry_price - stop_loss)` -- limits risk to 5% of portfolio per trade using stop-loss distance (default; was 1% for larger accounts)

When volatility scaling is enabled (`use_volatility_scaling` in backtest config, or passing `volatility` to `evaluate()`), position sizes are scaled inversely to realized volatility. High volatility → smaller positions, low volatility → base size (never increases beyond base to avoid leverage). In the backtest, volatility is computed per candidate from that ticker's own historical close series so each signal is sized by its own vol regime rather than a single market-wide proxy. The baseline annualized volatility is configurable via `VOLATILITY_BASELINE` (default: 20%).

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

IBKR and yfinance use different share-class symbol formats (IBKR: `BRK B` with a space; yfinance: `BRK-B` with a hyphen). The data layer translates IBKR symbols at the yfinance boundary so share-class tickers survive the backtest download and the analyst-consensus lookup instead of silently falling out.

### Backtest Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| Slippage | 0.1% | Simulated execution slippage |
| Commission | $1/trade | Per-trade commission cost |
| Initial capital | $100,000 | Starting portfolio value |
| Warmup period | 60 days | Days skipped for indicator stabilization |
| Indicator weights | Equal (1.0) | Per-indicator weight multipliers for scoring |
| Volatility scaling | Off | Scale position sizes inversely to realized volatility (per-candidate) |
| Max extension | 20% | Drop candidates whose close is more than this % above MA20 (0 disables) |

### Walk-Forward Validation

`walk_forward_backtest(config, train_ratio=0.6)` splits the date range into an **in-sample (IS)** training period and a non-overlapping **out-of-sample (OOS)** test period, runs a separate backtest on each with fresh capital, and reports per-period metrics plus a **degradation** dict (`OOS − IS` for each metric). Large negative degradation (e.g., IS win rate 70% but OOS win rate 45%) means the strategy memorized the IS window and will underperform in live trading. Returns a `WalkForwardResult` dataclass with both portfolios, both metric sets, the split date, and the degradation map.

### Split and Dividend Adjustment

YFinance downloads now pass `auto_adjust=True` explicitly, so the returned OHLC series is continuous through stock splits and dividends (a 4-for-1 split no longer appears as a -75% crash). Two helpers support post-hoc auditing when data comes from a non-adjusting source:
- `detect_unadjusted_splits(df, threshold=0.3)` — flags days with close-to-close changes large enough to look like an unadjusted split, returning `{date, ratio, type}` entries
- `adjust_for_splits(df, {date: ratio})` — retroactively scales pre-split OHLC and volume so the series becomes continuous

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

### AI Value-Add Comparison

The `compare_ai_value_add()` function compares screener-only vs screener+AI backtest results to measure whether the AI analyst adds or destroys value. It reports:

| Metric | Description |
|--------|-------------|
| **Return Alpha** | Return difference (AI - screener-only) |
| **Sharpe Alpha** | Risk-adjusted return difference |
| **P&L Alpha** | Absolute profit difference |
| **AI Filter Rate** | % of screener trades the AI filtered out |
| **AI Adds Value** | Boolean: whether the AI improved returns |

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
| **System Started** | Mode, portfolio value, cash balance |
| **Risk-Approved Signals** | Consolidated summary of all signals that passed risk checks: ticker, action, confidence, entry/SL/TP |
| **Scan Summary** | Per-cycle summary: candidates found, AI-approved signals, risk-approved trades, orders placed |
| **Trade Opened** | Ticker, action (BUY/SELL), quantity, price, stop-loss, take-profit, confidence score, AI reasoning |
| **Trade Closed** | Ticker, exit price, P&L percentage, profit/loss amount |
| **Daily Summary** | End-of-day report with portfolio value, daily P&L, number of trades, open positions |
| **Risk Warning** | Alerts when daily loss limit is approached, positions rejected by risk manager |
| **System Error** | Connection failures, API errors, unexpected exceptions |
| **System Stopped** | Notification when the trader shuts down |

### Setup

1. Create a Telegram bot via @BotFather (see [Prerequisites](#3-telegram-bot-for-notifications))
2. Add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` to your `.env` file
3. Notifications are fire-and-forget -- Telegram failures will not crash the trading system
4. **Interactive status**: Send "status" to the bot to get a detailed status update. The response makes a fresh API call to IBKR and includes: account summary, P&L breakdown (unrealized/realized/total), today's trade stats (W/L), open positions with live prices and current market value, open orders with status, and current phase

---

## Database Schema

The system uses SQLite (stored at `data/portfolio.db`) with WAL mode enabled for concurrent read/write performance.

### Tables

#### `positions` -- Open Positions
| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment ID |
| `ticker` | TEXT | Stock ticker symbol |
| `exchange` | TEXT | Exchange (SMART/NYSE/NASDAQ) |
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
| `exchange` | TEXT | Exchange (SMART/NYSE/NASDAQ) |
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
| `exchange` | TEXT | Exchange (SMART/NYSE/NASDAQ) |
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
| `conftest.py` | Shared fixtures: `make_signal()`, `make_position()` factories |
| `test_models.py` | Data class construction, properties, P&L calculations |
| `test_connection.py` | IBKR connection, reconnection, contract creation |
| `test_data.py` | Historical data fetching, caching (including mutation safety), news API, yfinance auto-adjust verification, split detection and retroactive adjustment |
| `test_portfolio.py` | Database CRUD operations, position lifecycle |
| `test_universe.py` | Universe building, financial sector filtering, caching |
| `test_screener.py` | Technical indicator calculations, scoring, signal generation |
| `test_analyst.py` | LLM integration, response validation (including `trade_type`), cost tracking |
| `test_risk.py` | All 15 risk checks, cumulative risk, sector concentration, circuit breaker, analyst consensus, correlation cap, volatility scaling, exit signal bypass, empty sector safety net |
| `test_executor.py` | Fill handling, exit handlers, bracket deduplication, day trade close with bracket cancellation, IBKR position import |
| `test_scheduler.py` | Streaming pipeline, fill handler ordering, exit tracking, nightly reconciliation (read-only drift detection with Telegram alerts) |
| `test_telegram.py` | Status commands, portfolio display, risk notifications |
| `test_backtest.py` | Backtest engine, MTM equity curve, realistic gap fills, short positions, look-ahead bias checks, anti-momentum current_price passthrough, simultaneous TP/SL resolution, walk-forward in-sample/out-of-sample splitting and degradation reporting |
| `test_stale_orders.py` | Stale order detection, re-screening, cancellation |
| `test_settings.py` | Configuration validation at startup |

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

4. **Daily loss limit** -- If the portfolio's daily P&L (realized + unrealized) drops below -2% (configurable), all trading is automatically halted for the remainder of the day.

5. **Position size limits** -- No single position can exceed 5% of portfolio value (configurable). Position sizing also accounts for stop-loss distance to limit risk to 1% of portfolio per trade.

6. **Sector concentration limits** -- No single sector can exceed 25% of the portfolio, preventing over-concentration.

7. **Duplicate position prevention** -- The system will not open a second position in a stock that is already held. This is enforced at both the risk manager level (check before approval) and the database level (guard in `add_position` prevents duplicate inserts from race conditions).

8. **Day trade auto-close** -- Day trade positions are automatically closed 15 minutes before market close to prevent unintended overnight exposure.

9. **Dry-run mode** -- Run the full pipeline and observe decisions without any orders being placed.

10. **Full audit trail** -- Every signal, trade, and risk decision is logged to SQLite and CSV for review.

11. **Financial sector exclusion** -- Banks, insurance companies, and lending institutions are permanently excluded from the trading universe.

11b. **Defense/military exclusion** -- Defense contractors, weapons manufacturers, and military equipment companies are permanently excluded from the trading universe.

12. **Non-equity ETF exclusion** -- Bond ETFs, leveraged/inverse ETFs, commodity ETFs, and volatility products are automatically filtered out. Equity index ETFs (SPY, QQQ, etc.) are kept.

13. **Startup position sync** -- On every startup (and reconnect), the bot fully syncs its database with IBKR: positions closed at IBKR while offline are recorded as trades at the stop-loss price (best estimate of actual fill), positions held at IBKR but missing from the DB are imported (with stop-loss/take-profit extracted from open bracket orders), and exit handlers are reattached for all existing orders.

13b. **Nightly reconciliation** -- Once per day, after all markets have closed, the scheduler runs a **read-only** reconciliation between the SQLite positions table and `ib.positions()`. Any orphaned positions (in DB but not at IBKR, or vice versa), quantity mismatches, or direction mismatches are reported via Telegram. Direction mismatches (DB long vs IBKR short) trigger a CRITICAL alert. Unlike startup reconciliation (which auto-closes orphans), the nightly check never modifies state — it purely detects silent drift during normal operation.

13. **Circuit breaker** -- If the system takes 3 consecutive losing trades within 60 minutes (both configurable), all new trading is paused and a Telegram alert is sent. This catches regime changes, stale data, or systematic issues before the daily loss limit is hit.

14. **4-tier sector fallback** -- Stock sector data is resolved through IBKR contract details, then yfinance, then Gemini, then Ollama. Gemini shares the same process-wide exhaustion flag as the AI analyst, so an auth failure or quota depletion in one path disables Gemini everywhere and the bot degrades cleanly to the Ollama-only path. Only stocks that fail all four are excluded.

14. **News resilience** -- Stock-specific news is fetched from Tavily first (richer results), falling back to yfinance when Tavily errors (e.g. rate limit), is unconfigured, or returns nothing. Each fetch logs which source succeeded so fallback behavior is visible in logs. Successful news is cached for 1 hour; failed fetches use a 60-second cache so retries happen sooner. When Tavily signals plan/rate-limit exhaustion, a process-lifetime flag short-circuits subsequent calls directly to the yfinance fallback until the process restarts — avoids burning dozens of Tavily requests per cycle once the quota is hit. Candidates with zero headlines from both sources are dropped before the AI analyst call so the LLM only sees tickers with at least one external news signal.

15. **Macro/political awareness** -- The AI analyst evaluates broad market political, regulatory, and macroeconomic conditions (elections, trade wars, sanctions, Fed policy) as part of its 7-point checklist. Macro headlines are fetched once per scan cycle via Tavily and shared across all candidates.

16. **Analyst consensus gate** -- Before buying, the bot fetches analyst consensus recommendations from yfinance. If the majority of analysts rate the stock as "sell" or "strong sell", the BUY signal is blocked. Stocks rated "buy", "strong buy", or "hold" pass through. If no analyst data is available (small-cap, newly listed), the buy is still allowed. Data is cached for 24 hours. Controlled by `CHECK_ANALYST_CONSENSUS` setting.

17. **PDT (Pattern Day Trader) protection** -- IBKR restricts accounts with Liquid Net Worth below `PDT_PROTECTION_THRESHOLD_USD` (default $5,000) to closing-orders-only for 30 days once 2 day trades occur within a rolling 5-business-day window. When the portfolio is at or above the threshold, this check is a pass-through. Below it, the bot counts same-calendar-day round-trip trades in the last 5 business days and blocks any new entry or same-day exit that would push the count to `PDT_MAX_DAY_TRADES_PER_5_DAYS` (default 1 — one less than IBKR's trigger of 2). Exits of positions opened on a prior day are never blocked so swing positions can always be closed. The scheduler queries the full 7-calendar-day window when fetching trades so the rolling count reflects day trades from earlier in the week, not just today.

18. **Parabolic breakout filter** -- Tickers whose latest close is more than `MAX_EXTENSION_OVER_MA20_PCT` (default 20%) above their 20-day moving average are dropped from the screener candidate pool before scoring. This prevents late entries into already-extended moves where confluent BUY indicators would otherwise convince the AI analyst to approve the trade.

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
- For unattended operation, use `--watchdog` mode which auto-restarts the gateway and reconnects

### IBKR Data Issues

**"Pacing violation" errors**
- IBKR limits historical data requests to 60 per 10 minutes
- The system batches and caches requests, and the universe builder inserts a 0.05-second delay between `reqContractDetails` calls to stay within IBKR rate limits
- Very large universes may still hit this limit -- reduce universe size or increase `SCAN_INTERVAL_MINUTES`

### AI Analyst Issues

**"Gemini auth failed (401/403) -- latching exhausted flag"**
- The `GEMINI_API_KEY` is invalid, revoked, or lacks permissions for the selected model
- Check the key at https://aistudio.google.com/apikey; generate a fresh one and update `.env`
- The flag clears on process restart; after fixing the key, restart the bot

**"Gemini plan exhausted -- short-circuiting for process lifetime"**
- Free-tier quota or prepaid credits are depleted on the Gemini project
- The bot will continue transparently on Ollama until restarted; top up billing at https://ai.studio/projects or switch `AI_PROVIDER=ollama` to silence the warning

**"Gemini transient HTTP 5xx" or "Gemini network error"**
- Transient — no action required. The bot falls back to Ollama for this call and retries Gemini on the next candidate

**"LLM call failed" or connection refused (Ollama)**
- Make sure Ollama is running: `ollama serve` (or it runs as a system service)
- Verify the model is downloaded: `ollama list` should show `qwen3:8b`
- Check `OLLAMA_HOST` in `.env` matches the Ollama address (default: `http://localhost:11434`)

**Slow analysis**
- Ollama takes ~30-60 seconds on CPU-only hardware -- normal for local inference
- With a GPU, Ollama responses are 5-10x faster; Gemini responses are typically 2-5 seconds regardless
- For faster Ollama at the cost of quality, try a smaller model (`qwen2.5:3b`)

### Backtest Issues

**"No data available" for tickers**
- YFinance may not have data for all tickers
- Check the ticker symbol is valid
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
| **US (NYSE/NASDAQ)** | 16:30 | 23:00 | Adjusted for TRT (Europe/Istanbul) |

The scheduler automatically detects if the market is open and only runs scans during active hours. Weekends are skipped.

---

## License

This project is for personal use. All rights reserved.

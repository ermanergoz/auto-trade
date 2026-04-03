# Auto Stock Trader - Implementation Plan

## Context

This plan covers building an automated stock trading system from scratch in Python. The system targets both US (NYSE/NASDAQ) and Turkish (BIST) equities through a single Interactive Brokers account. The motivation is to automate a two-stage screening pipeline: a fast technical screener filters thousands of stocks down to ~20 candidates, then an LLM (Claude or GPT) performs deep analysis on only those candidates, keeping AI costs under ~$1/day. A risk manager gates every trade before execution.

The user is a capable developer with some trading experience who wants to start with paper trading, validate the system over 1-2 weeks, and then move to small real-money positions. Financial sector stocks are explicitly excluded. The system runs locally on the user's machine with SQLite for persistence and Telegram for notifications.

The design spec defines 8 milestones. This plan orders the work with explicit dependencies, file-by-file creation guidance, integration points, and verification steps for each milestone.

---

## Project Initialization (Before Milestone 1)

Create the project root and virtual environment:

```bash
mkdir -p ~/auto-trader/{config,core,backtest,notifications,tests}
cd ~/auto-trader
python3 -m venv .venv
source .venv/bin/activate
```

Create `requirements.txt`:
```
ib_insync>=0.9.86
yfinance>=0.2.31
pandas>=2.1.0
pandas-ta>=0.3.14b
anthropic>=0.39.0
openai>=1.50.0
python-telegram-bot>=20.7
python-dotenv>=1.0.0
apscheduler>=3.10.4
rich>=13.7.0
tavily-python>=0.3.0
pytest>=7.4.0
pytest-asyncio>=0.23.0
```

Create `.gitignore` (include `.env`, `__pycache__`, `.venv`, `*.db`).

Create `.env.example` with placeholder keys for `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `TAVILY_API_KEY`.

---

## Milestone 1: Core Infrastructure

**Goal**: IBKR connection working, market data flowing, SQLite schema in place, data models defined.

**Dependencies**: None (first milestone).

### Files to Create

#### `config/settings.py`
All configurable parameters as module-level constants. Include every setting from the spec: broker connection (host, port, client_id), market schedules (BIST 10:00-18:00 TRT, US 16:30-23:00 TRT), screening thresholds, risk limits, notification config. Load `.env` values via `python-dotenv`. Provide a `is_paper_mode()` helper that checks the port (7497=paper, 7496=live).

#### `config/.env`
Actual API keys (gitignored). Copy from `.env.example`.

#### `core/models.py`
Python dataclasses for the entire system's data flow:
- `Signal` — action (buy/sell/hold), ticker, confidence, entry_price, stop_loss, take_profit, reasoning, timestamp, source (screener/ai)
- `Position` — ticker, exchange, quantity, entry_price, entry_time, stop_loss, take_profit, trade_type (day/swing), sector
- `Trade` — completed trade with entry/exit prices, P&L, reasoning, duration
- `DailySummary` — date, portfolio_value, daily_pnl, num_trades
- `StockInfo` — ticker, exchange, sector, market_cap, avg_volume

#### `core/portfolio.py`
SQLite-backed portfolio tracker. Create tables on first run: `positions`, `trades`, `daily_summary`, `signals`. Methods:
- `add_position(position)`, `close_position(ticker, exit_price)`, `get_open_positions()`
- `record_trade(trade)`, `get_daily_pnl()`, `get_portfolio_value()`
- `record_signal(signal)`, `get_daily_summary()`
- Use `contextmanager` for DB connections. All methods are synchronous (SQLite is local and fast).

#### `core/data.py`
Market data service — IBKR is the primary data source for everything:
- `get_historical_data(ib, contract, duration, bar_size)` — wraps `ib.reqHistoricalData()`. Works for both US and BIST contracts. Returns pandas DataFrame with OHLCV columns.
- `get_historical_data_yfinance(ticker, period, interval)` — fallback for backtest mode when IBKR isn't connected. For BIST tickers, auto-appends `.IS` suffix.
- `get_realtime_quote(ib, contract)` — requests snapshot from IBKR via `ib.reqMktData()`.
- `subscribe_realtime(ib, contract, callback)` — streaming real-time data via IBKR.
- `get_news(ticker, market)` — calls Tavily API for recent news headlines. Returns list of headline strings.
- Implements a simple in-memory cache with TTL to avoid redundant requests within a scan interval.

#### `core/connection.py`
**CRITICAL INTEGRATION POINT** — IBKR connection manager.
- `connect(host, port, client_id)` — returns `ib_insync.IB()` instance. Handles connection timeout (10s default).
- `disconnect(ib)` — clean shutdown.
- `ensure_connected(ib)` — reconnects if connection dropped. IBKR drops connections after inactivity or TWS restart.
- `create_contract(ticker, exchange)` — returns appropriate `ib_insync.Stock` contract. US stocks use "SMART" exchange; BIST stocks use "BIST" exchange with "TRY" currency.
- Important: TWS or IB Gateway must be running and configured to accept API connections on the correct port. This is a prerequisite the user must set up manually.

#### `main.py`
Entry point with `argparse`:
- `--mode paper|live|backtest|dry-run` (default: paper)
- `--market us|bist|all` (default: all)
- `--once` flag for single scan (useful for testing)
- Validates settings, establishes IBKR connection (unless backtest mode), initializes portfolio DB, hands off to scheduler.

### Verification Steps
1. `python main.py --mode paper --once` connects to TWS paper trading and prints account summary
2. `core/data.py` unit test: fetch 30 days of AAPL and THYAO.IS data, verify DataFrame shape
3. `core/portfolio.py` unit test: create DB, add/close position, verify trade record
4. `core/connection.py` unit test: connect/disconnect to paper TWS, create US and BIST contracts

### Critical Integration Notes
- **TWS must be running**: The system cannot start without TWS or IB Gateway running. Document this clearly in README.
- **API connection settings in TWS**: User must enable "Enable ActiveX and Socket Clients" in TWS API settings and set the socket port.
- **BIST contract qualification**: BIST stocks may need `ib.qualifyContracts()` to resolve ambiguous contracts. Test with a known BIST ticker like THYAO.

---

## Milestone 2: Technical Screener

**Goal**: Scan the full stock universe and produce ~10-20 candidates per market.

**Dependencies**: Milestone 1 (needs data service, models, IBKR connection for universe building).

### Files to Create

#### `core/universe.py`
Stock universe builder:
- `build_universe(ib, markets)` — queries IBKR scanner for all stocks on NYSE, NASDAQ, BIST. Filters out financial sector (GICS sector code). Applies `MIN_DAILY_VOLUME` and `MIN_MARKET_CAP` filters from settings.
- `cache_universe(stocks)` / `load_cached_universe()` — stores to/reads from a JSON file (`data/universe_{date}.json`). Only rebuilds once per day.
- For IBKR scanning: use `ib.reqScannerSubscription()` with scan codes for each exchange, or `ib.reqScannerParameters()` to discover available scan types.
- Fallback: if IBKR scanner is slow or limited, maintain a static ticker list seeded from YFinance's `Tickers` and filter via IBKR contract details for sector.

#### `core/screener.py`
Technical screener using `pandas-ta`:
- `screen_stocks(tickers, data_service)` — iterates all tickers, fetches 60 days of daily OHLCV data, computes indicators, checks for patterns.
- Indicator functions (all operate on a DataFrame):
  - `check_rsi(df)` — RSI(14), flag if < 30 or > 70
  - `check_macd(df)` — MACD crossover detection (signal line cross)
  - `check_ma_crossover(df)` — MA5 crossing MA20
  - `check_volume_spike(df)` — today's volume > 2x 20-day average
  - `check_bollinger(df)` — price outside Bollinger Bands (20, 2)
  - `check_support_resistance(df)` — price within 2% of recent pivot points
- `score_candidate(signals)` — count how many patterns triggered, rank candidates.
- Returns top 10-20 candidates per market as list of `Signal` objects with `source="screener"`.
- **Performance**: This scans potentially 2000+ tickers. Use IBKR `reqHistoricalData()` with batching (respect IBKR's pacing rules: max 60 historical data requests per 10 minutes). Consider `concurrent.futures.ThreadPoolExecutor` for parallelism. For initial universe scan, cache historical data locally to avoid re-fetching.

### Verification Steps
1. Build universe for US market, verify financial sector stocks are excluded (spot-check: no JPM, BAC, GS)
2. Run screener on a small subset (50 tickers), verify it produces candidates with valid indicator values
3. Timing test: full US universe screen should complete within 5 minutes
4. Verify BIST tickers resolve correctly (`.IS` suffix handling)

### Critical Integration Notes
- **IBKR pacing rules**: Max 60 historical data requests per 10 minutes. Batch and cache aggressively. If pacing-limited, queue requests with delays.
- **IBKR scanner limitations**: IBKR's scanner API has limited filtering capabilities for BIST. May need to maintain a manual BIST ticker list initially and filter by sector via contract details.
- **Data quality**: Some BIST tickers may have sparse historical data. Handle missing data gracefully (skip ticker, log warning).

---

## Milestone 3: AI Analyst

**Goal**: LLM analyzes screener candidates and produces structured trade signals.

**Dependencies**: Milestone 2 (needs screener candidates), Milestone 1 (needs data service for context gathering).

### Files to Create

#### `core/analyst.py`
LLM-powered analyst:
- `analyze_candidate(ticker, screener_signal, data_service, news_service)` — gathers context, calls LLM, parses response.
- Context gathering: recent price action (5 days OHLCV), all technical indicator values from screener, news headlines (last 3 days via Tavily), sector performance.
- Prompt construction: build a structured prompt that includes all context and asks for a JSON response with: `action` (buy/sell/hold), `confidence` (0-100), `entry_price`, `stop_loss`, `take_profit`, `reasoning`.
- LLM call: use `anthropic` SDK for Claude (default) or `openai` SDK for GPT. Model configurable via `AI_MODEL` setting. Use JSON mode / tool_use for structured output.
- Response parsing: validate JSON response, ensure all fields present, confidence is numeric, prices are reasonable relative to current price.
- Filtering: only return signals with `confidence >= AI_CONFIDENCE_THRESHOLD` (default 65).
- `analyze_batch(candidates)` — processes all candidates sequentially (to stay within rate limits). Returns list of `Signal` objects with `source="ai"`.

### Verification Steps
1. Feed 3 known tickers with clear technical signals to the analyst, verify structured responses parse correctly
2. Test with both Claude and GPT models, verify output format is consistent
3. Verify confidence threshold filtering works (mock a response with confidence=50, confirm it's filtered out)
4. Estimate cost: run on 20 candidates, check API usage — should be well under $1

### Critical Integration Notes
- **Structured output**: Use Claude's tool_use or GPT's function_calling / response_format to guarantee JSON structure. Do NOT rely on parsing free-text.
- **Error handling**: LLM calls can fail (rate limits, timeouts, malformed responses). Wrap in retry logic (3 attempts with backoff). On persistent failure, skip candidate and log.
- **Prompt engineering**: The prompt quality directly determines trade quality. Include explicit instructions about risk/reward ratio, market conditions, and the specific technical patterns that triggered the screener.
- **Cost control**: Log token usage per call. Alert if daily cost exceeds $2.

---

## Milestone 4: Risk Manager + Execution Engine

**Goal**: Risk rules gate every trade. Orders are placed and managed through IBKR.

**Dependencies**: Milestone 3 (needs AI signals), Milestone 1 (needs portfolio tracker, IBKR connection).

### Files to Create

#### `core/risk.py`
Risk manager — every trade must pass ALL checks:
- `check_position_size(signal, portfolio)` — proposed position <= `MAX_POSITION_SIZE_PCT` of portfolio value
- `check_daily_loss_limit(portfolio)` — today's P&L has not breached `-DAILY_LOSS_LIMIT_PCT`
- `check_max_positions(portfolio)` — fewer than `MAX_OPEN_POSITIONS` open
- `check_stop_loss(signal)` — signal has a stop_loss set (use default if missing)
- `check_sector_concentration(signal, portfolio)` — sector exposure won't exceed `MAX_SECTOR_CONCENTRATION_PCT`
- `check_no_duplicate(signal, portfolio)` — no existing position in this ticker
- `evaluate(signal, portfolio)` — runs all checks, returns `(approved: bool, reasons: list[str])`. Logs all rejections.
- `calculate_position_size(signal, portfolio)` — given approved signal, calculate number of shares based on max position size and stop-loss distance.

#### `core/executor.py`
**CRITICAL INTEGRATION POINT** — IBKR order execution:
- `place_order(ib, signal, quantity)` — creates bracket order via `ib_insync`:
  - Parent: limit or market order for entry
  - Stop-loss: stop order at `signal.stop_loss`
  - Take-profit: limit order at `signal.take_profit`
  - Uses `ib.placeOrder()` and returns order IDs
- `monitor_orders(ib, order_ids)` — checks order status (filled, partial, cancelled). Uses `ib.orderStatusEvent` callback.
- `cancel_order(ib, order_id)` — cancels unfilled orders.
- `close_position(ib, ticker, quantity)` — places market sell order to close.
- `close_all_day_trades(ib, portfolio)` — called before market close. Closes all positions marked as day trades.
- `handle_fill(trade, fill)` — callback when order fills. Updates portfolio tracker.
- Connection resilience: detect disconnection via `ib.disconnectedEvent`, attempt reconnection, re-subscribe to order updates.

#### `core/scheduler.py`
Main orchestration loop:
- Uses `APScheduler` to run the trading pipeline on `SCAN_INTERVAL_MINUTES`.
- `run_scan_cycle(ib, market)`:
  1. Check if market is currently open (time-based)
  2. Run screener on universe for this market
  3. Send candidates to AI analyst
  4. Pass approved signals through risk manager
  5. Execute approved trades
  6. Log everything
- Market hours logic: BIST 10:00-18:00 TRT, US 16:30-23:00 TRT. During overlap (16:30-18:00), scan both.
- End-of-day: 15 minutes before close, trigger `close_all_day_trades()`.
- Graceful shutdown: catch SIGINT/SIGTERM, cancel pending orders, disconnect cleanly.

### Verification Steps
1. **Risk manager unit tests**: Test each rule independently with mock portfolio states. Verify a signal that violates any rule is rejected.
2. **Paper trading order test**: Place a single bracket order on paper account. Verify parent fills, stop-loss and take-profit orders appear in TWS.
3. **Order monitoring**: Place an order, verify fill callback updates portfolio DB.
4. **Day trade close**: Open a day-trade position, verify it closes before market end.
5. **Connection drop test**: Disconnect TWS briefly, verify executor reconnects and resumes.

### Critical Integration Notes
- **Bracket orders in IBKR**: `ib_insync` bracket orders require setting `parentId` on child orders. Use `ib.bracketOrder()` helper. Test carefully — incorrect bracket setup can leave orphaned stop-loss orders.
- **Order types for BIST**: Verify that bracket orders work on BIST exchange. Some exchanges have restrictions on order types.
- **Fill timing**: IBKR fills are asynchronous. The executor must handle the case where a fill arrives while the system is processing other signals.
- **Day trade close timing**: Account for potential delays. Start closing 15 minutes before close, but if fills are slow, may need to send market orders.
- **Dry-run mode**: In dry-run, log the order that would be placed but do not call `ib.placeOrder()`. This is a simple flag check in `place_order()`.

---

## Milestone 5: Notifications + Logging

**Goal**: Telegram alerts, terminal dashboard, trade journal.

**Dependencies**: Milestone 4 (needs trades happening to report on).

### Files to Create

#### `notifications/telegram.py`
Telegram bot integration:
- `send_message(text)` — sends to configured chat ID via `python-telegram-bot`.
- `notify_trade(trade)` — formatted message: ticker, action, quantity, price, stop-loss, reasoning snippet.
- `notify_daily_summary(summary)` — end-of-day report: P&L, trades count, open positions.
- `notify_risk_warning(message)` — alerts for daily loss limit approached, connection issues, etc.
- `notify_error(error)` — system errors that need attention.
- Async: use `telegram.Bot.send_message()` directly (no need for full bot polling). Fire-and-forget with error handling (don't let Telegram failures crash the trader).

#### `core/logger.py`
Structured logging and terminal display:
- Configure Python `logging` with file handler (`logs/trader_{date}.log`) and console handler.
- Trade journal: append each trade to `logs/trades_{date}.csv` with all fields.
- Terminal dashboard using `rich`:
  - `display_positions(positions)` — table of open positions with current P&L
  - `display_recent_trades(trades)` — last 10 trades
  - `display_portfolio_summary(summary)` — total value, daily P&L, position count
  - `display_scan_results(candidates)` — screener output
- Dashboard updates on each scan cycle (not continuous — refreshes every `SCAN_INTERVAL_MINUTES`).

### Verification Steps
1. Send a test Telegram message, verify it arrives in the configured chat
2. Run a scan cycle with logging enabled, verify log file contains structured entries
3. Verify CSV export contains all trade fields and is valid (open in spreadsheet)
4. Verify rich dashboard renders correctly in terminal (test with mock data)

---

## Milestone 6: Backtesting Engine

**Goal**: Replay historical data through the same strategy code. No IBKR connection needed.

**Dependencies**: Milestone 2 (screener code), Milestone 3 (analyst code), Milestone 4 (risk manager).

### Files to Create

#### `backtest/engine.py`
Backtesting engine:
- `run_backtest(tickers, start_date, end_date, config)`:
  1. Download historical data for all tickers via YFinance
  2. Iterate day by day (or bar by bar for intraday)
  3. For each bar: run `screener.screen_stocks()` on data up to that point (no look-ahead)
  4. For screener candidates: either run AI analyst (expensive) or use cached signals from `signals` table
  5. Pass signals through risk manager with simulated portfolio state
  6. Simulate order execution with configurable slippage (default 0.1%) and commission (default $1/trade)
  7. Track simulated portfolio state throughout
- `SimulatedPortfolio` class: mirrors `portfolio.py` interface but operates in-memory without SQLite.
- Key constraint: **uses the exact same `screener.py` and `risk.py` code as live trading**. The backtester only replaces the data source (historical vs live) and execution (simulated vs IBKR).

#### `backtest/report.py`
Performance reporting:
- `calculate_metrics(trades, initial_capital)`:
  - Total return, annualized return
  - Sharpe ratio (assuming risk-free rate from settings)
  - Maximum drawdown (peak-to-trough)
  - Win rate (% of trades profitable)
  - Profit factor (gross profit / gross loss)
  - Average trade duration
  - Number of trades
- `compare_configs(results_list)` — side-by-side comparison table of multiple backtest runs.
- `plot_equity_curve(trades)` — optional matplotlib chart (save to file).
- Output as rich table to terminal and as JSON for programmatic consumption.

### Verification Steps
1. Backtest on 6 months of S&P 500 subset (50 tickers), verify no look-ahead bias (check that screener only sees data up to current bar)
2. Verify metrics calculations against a known set of trades (hand-calculate Sharpe, drawdown)
3. Verify that screener produces the same signals on historical data whether run via backtest or live code path
4. Run two configs side-by-side, verify comparison report renders correctly

### Critical Integration Notes
- **No look-ahead bias**: The backtester must slice the DataFrame to only include data up to the current simulation timestamp. This is the most common backtesting bug.
- **AI analyst in backtest**: Running the LLM on historical data is expensive and slow. Provide two modes: (a) use cached signals if available, (b) run AI analyst live (for initial signal generation). Cache all AI responses to the `signals` table for future reruns.
- **Slippage modeling**: Simple percentage slippage is a starting point. For more accuracy, consider volume-based slippage (harder to fill at desired price if volume is low).

---

## Milestone 7: Paper Trading Shakedown

**Goal**: Run the complete system on paper for 1-2 weeks. Tune parameters.

**Dependencies**: All previous milestones (1-6).

### No New Files — This Is a Tuning and Validation Phase

### Activities
1. **Run continuously during market hours** for at least 5 trading days on paper account.
2. **Monitor via Telegram and terminal dashboard**.
3. **Review every trade** in the trade journal. Check AI reasoning quality.
4. **Tune parameters based on observations**:
   - `SCAN_INTERVAL_MINUTES` — are 15-minute scans too frequent or too sparse?
   - `AI_CONFIDENCE_THRESHOLD` — are good trades being filtered at 65? Lower to 60? Or raise to 80?
   - `MAX_POSITION_SIZE_PCT` — is 5% appropriate for the account size?
   - `DEFAULT_STOP_LOSS_PCT` — is 3% too tight (getting stopped out on noise)?
   - Screener indicator thresholds (RSI boundaries, volume spike multiplier)
5. **Run backtest on the same period** — compare paper trading results with what the backtester would have predicted. Investigate any divergence (slippage, timing, data differences).
6. **Stress test**: What happens when TWS disconnects mid-day? When a ticker is halted? When the LLM API is down?
7. **Document all issues** found and fixes applied.

### Verification Steps
1. System runs for a full trading day without crashes
2. All trades have corresponding Telegram notifications
3. Daily summary is generated and accurate
4. Portfolio DB matches TWS paper account state (reconcile positions)
5. No orphaned stop-loss orders in TWS
6. System handles market open/close transitions cleanly

---

## Milestone 8: Options Support (Future)

**Goal**: Add options trading capability. Deferred until stock trading is stable.

**Dependencies**: Milestones 1-7 complete and validated.

### Files to Create/Modify (Sketch Only)
- `core/models.py` — add `OptionSignal` dataclass with strike, expiry, option_type
- `core/options_screener.py` — scan for options opportunities (unusual volume, IV rank)
- `core/options_executor.py` — IBKR options order placement
- `core/risk.py` — add options-specific risk rules (max Greeks exposure)
- `backtest/engine.py` — extend simulator for options P&L

This milestone is intentionally underspecified. Define detailed requirements after stock trading is proven.

---

## Dependency Graph

```
M1 (Infrastructure)
 ├── M2 (Screener)  ──depends on M1
 │    └── M3 (AI Analyst)  ──depends on M2
 │         └── M4 (Risk + Execution)  ──depends on M3, M1
 │              └── M5 (Notifications)  ──depends on M4
 │              └── M6 (Backtesting)  ──depends on M2, M3, M4
 │                   └── M7 (Paper Shakedown)  ──depends on all
 │                        └── M8 (Options)  ──deferred
```

Note: M5 and M6 can be worked on in parallel after M4 is complete.

---

## File Creation Order (Complete List)

| Order | File | Milestone | Purpose |
|-------|------|-----------|---------|
| 1 | `requirements.txt` | Setup | Dependencies |
| 2 | `.gitignore` | Setup | Git config |
| 3 | `.env.example` | Setup | API key template |
| 4 | `config/.env` | M1 | Actual API keys |
| 5 | `config/settings.py` | M1 | All configuration |
| 6 | `core/models.py` | M1 | Data classes |
| 7 | `core/connection.py` | M1 | IBKR connection manager |
| 8 | `core/portfolio.py` | M1 | SQLite portfolio tracker |
| 9 | `core/data.py` | M1 | Market data service |
| 10 | `main.py` | M1 | Entry point |
| 11 | `core/universe.py` | M2 | Stock universe builder |
| 12 | `core/screener.py` | M2 | Technical screener |
| 13 | `core/analyst.py` | M3 | LLM integration |
| 14 | `core/risk.py` | M4 | Risk manager |
| 15 | `core/executor.py` | M4 | IBKR order execution |
| 16 | `core/scheduler.py` | M4 | Main orchestration loop |
| 17 | `notifications/telegram.py` | M5 | Telegram alerts |
| 18 | `core/logger.py` | M5 | Logging + dashboard |
| 19 | `backtest/engine.py` | M6 | Backtesting engine |
| 20 | `backtest/report.py` | M6 | Performance metrics |

---

## Critical Integration Points Summary

1. **IBKR Connection** (`core/connection.py`): The single most fragile integration. TWS/Gateway must be running, API connections enabled, correct port configured. Connection drops are common and must be handled gracefully. Test with paper account first (port 7497).

2. **IBKR Bracket Orders** (`core/executor.py`): Bracket order setup with `ib_insync` requires correct parent/child order linking. Incorrect setup can result in orphaned orders. BIST exchange may have different order type support than US exchanges.

3. **IBKR Scanner/Universe** (`core/universe.py`): The scanner API has limitations, especially for BIST. May need a hybrid approach: IBKR scanner for US stocks, manual/alternative ticker list for BIST, with sector filtering via contract details.

4. **IBKR Historical Data Pacing** (`core/data.py`): IBKR limits to 60 historical data requests per 10 minutes. The screener must batch requests and cache results. For backtesting without IBKR, fall back to YFinance (BIST data via `.IS` suffix can be spotty).

5. **LLM Structured Output** (`core/analyst.py`): Must use tool_use/function_calling for reliable JSON parsing. Free-text parsing will break. Retry logic is essential.

6. **Backtester Code Reuse** (`backtest/engine.py`): The screener and risk manager must be written as pure functions that take data as input (not fetch it themselves) so the backtester can feed them historical data. This is an architectural constraint that must be enforced from Milestone 2 onward.

7. **Market Hours + Time Zones**: All time logic must use Turkey time (TRT / Europe/Istanbul). Python's `zoneinfo` module handles this. Be careful with daylight saving transitions.

---

## Machine Setup Guide

This project will be built on a separate machine. Below is everything needed to set up from scratch.

### Prerequisites

1. **Python 3.11+**
   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt install python3.11 python3.11-venv python3-pip
   # Or via pyenv
   curl https://pyenv.run | bash
   pyenv install 3.11.8 && pyenv global 3.11.8
   ```

2. **Interactive Brokers TWS or IB Gateway**
   - Download TWS from: https://www.interactivebrokers.com/en/trading/tws.php
   - OR download IB Gateway (lighter, no GUI): https://www.interactivebrokers.com/en/trading/ibgateway-stable.php
   - Create an IBKR paper trading account at: https://www.interactivebrokers.com/en/trading/free-trial.php
   - **TWS API settings** (must configure manually in TWS):
     - Edit > Global Configuration > API > Settings
     - Check "Enable ActiveX and Socket Clients"
     - Set Socket Port: `7497` (paper) or `7496` (live)
     - Check "Allow connections from localhost only"
     - Uncheck "Read-Only API" (we need to place orders)
   - TWS/Gateway must be running whenever the trading system runs

3. **Telegram Bot** (for notifications)
   - Message @BotFather on Telegram
   - Send `/newbot`, follow prompts, save the bot token
   - Send a message to your new bot, then visit `https://api.telegram.org/bot<TOKEN>/getUpdates` to find your chat_id

4. **API Keys**
   - **Claude API key**: https://console.anthropic.com/ (or OpenAI: https://platform.openai.com/)
   - **Tavily API key** (news): https://tavily.com/ (free tier: 1000 searches/month)

### Project Setup

```bash
# Clone or create project
mkdir -p ~/auto-trader && cd ~/auto-trader
git init

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your actual API keys
```

### Environment Variables (.env)

```bash
# IBKR (no key needed — connects via socket to running TWS/Gateway)
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1

# AI Model (choose one)
ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...

# Notifications
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_CHAT_ID=123456789

# News
TAVILY_API_KEY=tvly-...
```

### Running

```bash
# Activate venv
source .venv/bin/activate

# Paper trading (default) — single scan for testing
python main.py --once

# Paper trading — continuous during market hours
python main.py

# Backtest mode (no IBKR needed)
python main.py --mode backtest

# Dry-run (full pipeline, no actual orders)
python main.py --mode dry-run

# Live trading (requires explicit flag)
python main.py --mode live
```

### Verification Checklist

After setup, verify each component:

1. `python -c "from ib_insync import IB; ib = IB(); ib.connect('127.0.0.1', 7497, clientId=1); print(ib.accountSummary()); ib.disconnect()"` — IBKR connection works
2. `python main.py --mode paper --once` — full pipeline runs one cycle
3. Check Telegram for a test notification
4. `python main.py --mode backtest` — backtester runs without IBKR

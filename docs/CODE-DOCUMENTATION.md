# Auto-Trade: Complete Code Documentation

A plain-English guide to every part of this automated stock trading system — what it does, how it works, why it was built this way, and the challenges we hit along the way.

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [The Big Picture — How a Trade Happens](#2-the-big-picture--how-a-trade-happens)
3. [Project Structure](#3-project-structure)
4. [The Data Models (core/models.py)](#4-the-data-models--coremodelspy)
5. [Configuration (config/settings.py)](#5-configuration--configsettingspy)
6. [Connecting to the Broker (core/connection.py)](#6-connecting-to-the-broker--coreconnectionpy)
7. [Getting Market Data (core/data.py)](#7-getting-market-data--coredatapy)
8. [Building the Stock Universe (core/universe.py)](#8-building-the-stock-universe--coreuniversepy)
9. [The Technical Screener (core/screener.py)](#9-the-technical-screener--corescreenerpy)
10. [The AI Analyst (core/analyst.py)](#10-the-ai-analyst--coreanalystpy)
11. [The Risk Manager (core/risk.py)](#11-the-risk-manager--coreriskpy)
12. [Executing Trades (core/executor.py)](#12-executing-trades--coreexecutorpy)
13. [The Scheduler — Tying It All Together (core/scheduler.py)](#13-the-scheduler--tying-it-all-together--coreschedulerpy)
14. [Portfolio Tracking (core/portfolio.py)](#14-portfolio-tracking--coreportfoliopy)
15. [Logging & Dashboard (core/logger.py)](#15-logging--dashboard--coreloggerpy)
16. [Notifications (notifications/telegram.py)](#16-notifications--notificationstelegramy)
17. [Backtesting Engine (backtest/engine.py)](#17-backtesting-engine--backtestenginepy)
18. [Backtest Reporting (backtest/report.py)](#18-backtest-reporting--backtestreportpy)
19. [The Entry Point (main.py)](#19-the-entry-point--mainpy)
20. [Testing Strategy](#20-testing-strategy)
21. [Challenges We Faced & How We Solved Them](#21-challenges-we-faced--how-we-solved-them)
22. [Architecture Decisions & Why](#22-architecture-decisions--why)
23. [The Complete Data Flow](#23-the-complete-data-flow)

---

## 1. What Is This Project?

This is an **automated stock trading bot** that trades US stocks (NYSE and NASDAQ) through Interactive Brokers (IBKR). It runs on your own computer, makes its own decisions, and places real orders.

Think of it like a 3-person trading desk, automated:

1. **The Screener** — A fast number-cruncher that scans hundreds of stocks every 15 minutes, looking for interesting patterns in prices, volume, and technical indicators. It's like a junior analyst who flags "hey, these 15 stocks look interesting right now."

2. **The AI Analyst** — An LLM that takes those 15 candidates and does deep analysis. It routes through **Gemini** (`gemini-2.5-flash-lite` by default) as the primary provider, and falls back to a local **Ollama + Qwen 2.5 7B** when Gemini is unavailable (missing key, rate limit, credits depleted, network error). It looks at price trends, momentum, volume, news headlines, macro/political context, and a strict 7-point checklist. It's the senior analyst who says "of those 15, I'd actually buy these 3."

3. **The Risk Manager** — A paranoid rule-checker that gates every trade with 15 different safety checks (11 core + 4 discipline checks for new entries only). Position too big? Rejected. Already lost too much today? Rejected. Chasing a stock that already moved 5%? Rejected. Three losses in a row? Circuit breaker pauses everything. Intraday-margin deficit uncured? New entries are blocked to prevent a 90-day restriction. Exit signals skip discipline checks so positions can always be closed. It's the compliance officer who makes sure we never blow up.

Only after all three agree does an order actually get placed.

---

## 2. The Big Picture — How a Trade Happens

Here's the complete journey of a trade, step by step:

```
Every 15 minutes during market hours:

  1. CONNECT — Make sure we're connected to IBKR
                    ↓
  2. UNIVERSE — Get today's list of tradeable stocks (~100-350 stocks)
               (cached after first build — it doesn't change during the day)
                    ↓
  3. DATA — Download 60 days of price history for every stock in the universe
                    ↓
  4. SCREEN — Run 6 technical indicators on each stock
             Score them. Keep stocks scoring above the minimum threshold.
             Typical result: ~10-20 candidates out of hundreds
                    ↓
  5. ANALYZE — Send each candidate to the local AI model with:
              - Recent price action (last 5 days)
              - Indicator values (RSI, MACD, etc.)
              - News headlines
              The AI returns: BUY/SELL/HOLD + confidence + prices + reasoning
              Only signals with confidence >= 65 pass through
                    ↓
  6. RISK CHECK — For each AI-approved signal, run 11 safety checks:
                  Position size, daily loss limit, sector concentration,
                  duplicate positions, stop-loss validity, and more.
                  Calculate exact number of shares to buy.
                    ↓
  7. EXECUTE — Place a bracket order on IBKR:
              - Entry order (the buy/sell)
              - Take-profit limit order (auto-sell when price target hit)
              - Stop-loss order (auto-sell when max loss hit)
              These three are linked — when one TP/SL fills, the other cancels
                    ↓
  8. RECORD — Save position to SQLite database
             Send Telegram notification
             Log to CSV trade journal
             Update terminal dashboard
```

And 15 minutes before market close, the system auto-closes all day trades when `DAY_TRADE_ENABLED` is True (swing trades stay open overnight; this step is skipped entirely when the day-trade path is disabled).

---

## 3. Project Structure

```
auto-trade/
│
├── main.py                  # The front door. Parse arguments, pick mode, start.
├── requirements.txt         # Python packages we depend on
├── CLAUDE.md                # Instructions for AI assistants working on this code
├── README.md                # User-facing guide
│
├── config/
│   └── settings.py          # Every number the system uses: thresholds, ports,
│                             # intervals, indicator periods. One file, no magic.
│
├── core/                    # Where the actual trading logic lives
│   ├── models.py            # Data shapes: Signal, Position, Trade, etc.
│   ├── connection.py        # Talk to IBKR (connect, disconnect, reconnect)
│   ├── data.py              # Get price data and news
│   ├── universe.py          # Build the list of stocks we're allowed to trade
│   ├── screener.py          # Technical indicator checks (the fast filter)
│   ├── analyst.py           # AI-powered analysis (the deep filter)
│   ├── risk.py              # Safety checks and position sizing
│   ├── executor.py          # Actually place and manage orders on IBKR
│   ├── scheduler.py         # The main loop that orchestrates everything
│   ├── portfolio.py         # SQLite database for tracking positions and trades
│   ├── logger.py            # Pretty terminal output and CSV trade journals
│   └── state.py             # Shared mutable state (shutdown flag)
│
├── backtest/                # Historical testing
│   ├── engine.py            # Replay historical data day-by-day
│   └── report.py            # Calculate performance metrics (Sharpe, drawdown, etc.)
│
├── notifications/
│   └── telegram.py          # Send alerts to your phone via Telegram
│
├── tests/                   # Automated tests for every module
│   ├── conftest.py          # Shared fixtures (make_signal, make_position)
│   ├── test_screener.py
│   ├── test_risk.py
│   ├── test_analyst.py
│   ├── test_data.py
│   ├── test_connection.py
│   ├── test_portfolio.py
│   ├── test_models.py
│   ├── test_universe.py
│   ├── test_scheduler.py
│   ├── test_telegram.py
│   └── test_backtest.py
│
├── data/                    # Runtime data (gitignored)
│   ├── portfolio.db         # SQLite database
│   └── universe_us_*.json   # Cached daily stock lists
│
├── logs/                    # Log files (gitignored)
│   ├── trader_*.log         # Daily system logs
│   └── trades_*.csv         # Daily trade journals
│
└── docs/
    ├── DESIGN.md            # Architecture spec
    └── IMPLEMENTATION-PLAN.md  # Build roadmap
```

---

## 4. The Data Models — `core/models.py`

This file defines the **shapes of data** that flow through the system. Think of them like forms that every piece of information must fill out. They're Python dataclasses — just containers for data, no behavior.

### Signal — "Hey, I think we should trade this stock"

A Signal is the output of either the screener or the AI analyst. It says: "I think you should BUY (or SELL) this stock at this price, with this stop-loss and take-profit."

```
Signal:
  - ticker: "AAPL"              ← which stock
  - action: BUY / SELL / HOLD   ← what to do
  - confidence: 0-100           ← how sure are we (AI gives this)
  - entry_price: 175.50         ← what price to buy at
  - stop_loss: 170.00           ← bail out if it drops here (limits losses)
  - take_profit: 185.00         ← cash out if it reaches here (locks profit)
  - reasoning: "Strong uptrend..."  ← why this trade
  - source: "screener" or "ai"  ← who generated this signal
  - trade_type: DAY or SWING    ← close today, or hold overnight?
  - indicator_values: {...}     ← raw indicator numbers for the AI to see
```

### Position — "We currently own this stock"

```
Position:
  - ticker, exchange, quantity, entry_price, entry_time
  - stop_loss, take_profit, trade_type, sector
  - current_price (updated live)
  - unrealized_pnl → computed: (current_price - entry_price) * quantity
  - unrealized_pnl_pct → computed: percentage change (guards against zero entry_price)
```

### Trade — "We bought and sold this stock, here's how it went"

```
Trade:
  - Everything from Position, plus:
  - exit_price, exit_time
  - pnl → computed: (exit_price - entry_price) * quantity
  - pnl_pct → computed: percentage gain/loss (guards against zero entry_price)
  - duration → computed: how long we held it (handles mixed tz-aware/naive datetimes from DB rows or backtest deserializers)
```

### Other models

- **DailySummary** — End-of-day snapshot: portfolio value, P&L, win/loss counts
- **StockInfo** — Basic info about a tradeable stock: ticker, sector, market cap, volume, country
- **Action** (enum) — BUY, SELL, HOLD
- **TradeType** (enum) — DAY (close same day), SWING (hold days/weeks)

---

## 5. Configuration — `config/settings.py`

This is the **single source of truth** for every number in the system. No magic numbers hidden in random files. If you want to change how the system behaves, this is where you go.

### Broker Settings
```python
IBKR_HOST = "127.0.0.1"    # IBKR runs on your local machine
IBKR_PORT = 7497            # 7497 = paper trading, 7496 = real money
IBKR_CLIENT_ID = 1          # Identifies our connection to IBKR
```

### What We Trade
```python
MARKETS = ["US"]                    # Only US stocks
EXCLUDED_SECTORS = ["Financials"]   # No banks, insurance, lending
FINANCIAL_KEYWORDS = [...]          # Shared keyword list used by both universe
                                    # builder and risk manager to detect financials
MIN_DAILY_VOLUME = 100_000          # Stock must trade 100K shares/day minimum
MIN_MARKET_CAP = 50_000_000         # $50M minimum market cap
```

Why exclude financials? Banks and insurance companies behave very differently from regular companies — their prices are driven by interest rates, regulations, and credit cycles rather than normal business performance. Our technical indicators don't work well on them.

### Strategy Settings
```python
SCAN_INTERVAL_MINUTES = 15         # Run the full pipeline every 15 min
AI_CONFIDENCE_THRESHOLD = 65       # AI must be 65%+ confident
AI_MAX_CANDIDATES = 0              # Max stocks sent to AI per cycle (0 = no limit)
AI_PROVIDER = "gemini"             # Primary LLM: "gemini" (auto-falls back to Ollama) or "ollama"
GEMINI_API_KEYS = []               # Comma-separated list in env (preferred). Bot round-robins
                                   #   per call; 1, 2, or 3 keys all work. Free tier is 1,000 RPD
                                   #   per key, so 3 keys ≈ 3,000 RPD/day, comfortably above the
                                   #   bot's ~2,688 calls/day at 15-min cadence.
GEMINI_API_KEY = ""                # Legacy single-key form, used only when GEMINI_API_KEYS is empty
GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_HOST = "https://generativelanguage.googleapis.com"
AI_MODEL = "qwen3:8b"             # Ollama fallback model
OLLAMA_HOST = "http://localhost:11434"  # Where Ollama runs
```

### Risk Limits — The Safety Net
```python
MAX_POSITION_SIZE_PCT = 50.0        # Max 50% of portfolio in one stock (tuned for $500 account)
DAILY_LOSS_LIMIT_PCT = 10.0         # Stop trading if down 10% today
MAX_OPEN_POSITIONS = 3              # Max 3 stocks at once
DEFAULT_STOP_LOSS_PCT = 3.0         # Default: bail if stock drops 3%
DEFAULT_TAKE_PROFIT_PCT = 6.0       # Default: cash out at 6% profit
MAX_SECTOR_CONCENTRATION_PCT = 50.0 # Max 50% in one sector
ANTI_MOMENTUM_PCT = 8.0             # Don't buy if already moved 8%
MIN_RISK_REWARD_RATIO = 1.5         # Potential profit must be 1.5x potential loss
ALLOW_SHORT_SELLING = False         # Block sells for stocks not held (no shorting)
CIRCUIT_BREAKER_LOSSES = 3          # Pause after 3 consecutive losses
CIRCUIT_BREAKER_WINDOW_MIN = 60     # Within this many minutes
DEFAULT_TRADE_TYPE = "swing"         # "swing" (default) or "day"; Signal.trade_type defaults to SWING
DAY_TRADE_ENABLED = False            # gate day-trading path; when False AI "day" signals are downgraded to "swing"
REG_T_MIN_EQUITY_USD = 2000.0        # Reg-T minimum equity to trade on margin ($2,000)
INTRADAY_MAINTENANCE_MARGIN_PCT = 25.0  # Intraday maintenance margin floor (25%)
MARGIN_REGIME = "both"               # Active margin model: "intraday" | "legacy_pdt" | "both"
LEGACY_PDT_THRESHOLD_USD = 25000.0   # Legacy PDT threshold (used only under legacy_pdt/both regimes)
```

These values are tuned for a small account ($500). For larger accounts, tighten them back — see [RISK-TUNING.md](RISK-TUNING.md) for a full comparison table and scaling guide.

The risk manager also includes a **cumulative risk check** that ensures total open risk (all positions' max loss via stop-loss) stays within the daily loss limit. It uses the actual calculated position size (which reflects volatility scaling) rather than re-estimating from config, ensuring accurate risk assessment when position sizes are adjusted down. This prevents a scenario where 3 positions each sized at 5% risk = 15% total risk, exceeding the 10% daily limit.

A **`validate_settings()`** function validates all configuration at startup and rejects invalid values (e.g., port not 7496/7497, negative ratios, zero positions).

### Technical Indicator Settings
```python
RSI_PERIOD = 14                # Standard RSI lookback window
RSI_OVERSOLD = 30              # Below 30 = oversold (potential buy)
RSI_OVERBOUGHT = 70            # Above 70 = overbought (potential sell)
MACD_FAST = 12, SLOW = 26     # MACD moving average periods
MA_FAST = 5, MA_SLOW = 20     # Short and long moving averages
VOLUME_SPIKE_MULTIPLIER = 2.0  # Volume must be 2x normal to count
```

---

## 6. Connecting to the Broker — `core/connection.py`

This module handles all communication with Interactive Brokers (IBKR). IBKR provides a desktop app called TWS (Trader Workstation) or a lighter version called IB Gateway. Our bot connects to either of these over a socket on your local machine.

### Key Functions

**`connect(host, port, client_id, timeout)`** — Establishes the connection. Think of it like logging into your brokerage account, but programmatically. If TWS isn't running or the port is wrong, it raises a `ConnectionError`.

**`ensure_connected(ib, ...)`** — IBKR connections drop frequently (network hiccups, TWS restarts). This function checks if we're still connected and reconnects if not. It's called at the start of every scan cycle.

**`create_contract(ticker, exchange)`** — Creates an IBKR "contract" object. IBKR needs to know exactly which financial instrument you're talking about. For US stocks, we use the "SMART" exchange (IBKR's smart order routing that finds the best price across exchanges).

**`get_account_summary(ib)`** — Asks IBKR: "What's my account worth? How much cash do I have? What's my P&L today?" Returns a dictionary with values like NetLiquidation (total account value), AvailableFunds, and RealizedPnL.

### Why It's Separate

Having connection logic in its own file means the rest of the code never needs to worry about connection details. The screener doesn't know about sockets. The risk manager doesn't know about ports. They just get data handed to them.

---

## 7. Getting Market Data — `core/data.py`

This module fetches prices and news. It has two data sources and caching to avoid being wasteful.

### Primary Source: IBKR

**`get_historical_data(ib, contract, duration, bar_size)`** — Gets OHLCV bars (Open, High, Low, Close, Volume) from IBKR. Default: 60 days of daily bars. This is what the screener analyzes.

**`get_realtime_quote(ib, contract)`** — Gets a current price snapshot. Used to check if a stock has already moved too much before buying.

### Fallback Source: YFinance

**`get_historical_data_yfinance(ticker, period, interval, auto_adjust=True)`** — Same data but from Yahoo Finance. Used only by the backtester (since you don't need an IBKR connection to backtest) and as a fallback if IBKR data fails. `auto_adjust=True` is the default and asks yfinance to return split- and dividend-adjusted OHLC so the series is continuous through corporate actions. Without this, a 4-for-1 split would appear as a -75% crash and poison every backtest.

### Split & Dividend Adjustment Helpers

**`detect_unadjusted_splits(df, threshold=0.3)`** — Scans a DataFrame for close-to-close changes large enough to look like an unadjusted split. Returns a list of `{date, ratio, type}` entries, where `type` is `"forward"` (price drop, ratio > 1 like 4.0 for 4-for-1) or `"reverse"` (price jump, ratio < 1 like 0.1 for 1-for-10). Intended for validating that a third-party data source actually delivered adjusted bars — if this returns anything, the series is suspect.

**`adjust_for_splits(df, {date: ratio})`** — Retroactively fixes a split-poisoned series. For a forward split (ratio > 1), pre-split prices are divided by the ratio and volume is multiplied by it; for a reverse split (ratio < 1), the same arithmetic applies (which multiplies price and divides volume). On-and-after-split rows are left untouched. Multiple splits compose cumulatively, so earlier bars are scaled by the product of every ratio applied after them. Returns a new DataFrame (the input is never mutated).

### News

**`get_news(ticker, market, max_results)`** — Fetches recent news headlines for AI context. Tries Tavily first (richer results); if Tavily is unconfigured, errors (e.g. rate limit), or returns nothing, falls back to yfinance. Logs whichever source won, so you can see fallback behavior in the trader logs. When Tavily signals plan/rate-limit exhaustion, a module-level `_tavily_exhausted` flag short-circuits subsequent calls for the rest of the process (flag resets on restart). Returns `[]` when both sources fail — the scheduler uses that to **skip the candidate entirely** before the LLM call, since no news = no external signal for the analyst to reason over.

**`get_macro_news(max_results)`** — Fetches broad market political/macro headlines via Tavily (no yfinance equivalent). Called once per scan cycle and shared across all candidates.

**`get_analyst_recommendation(ticker)`** — Fetches analyst consensus recommendation via yfinance's `recommendations_summary`. Returns a dict with `consensus` ("strong_buy", "buy", "hold", "sell", "strong_sell") and `details` (raw analyst counts). Cached for 24 hours. Returns None if no data available or on error. First of the two analyst-consensus sources used by the risk manager.

**`get_analyst_recommendation_ibkr(ib, ticker, exchange)`** — Second analyst-consensus source. Calls `ib.reqFundamentalData(Stock(ticker, exchange, 'USD'), 'RESC')` to fetch the Reuters/Refinitiv I/B/E/S report and parses the `<ConsRecom>` element — a 1.0–5.0 mean rating across all covering analysts. Mapped to the same vocabulary as yfinance via `_ibkr_cons_recom_to_label`: `<1.5` → strong_buy, `<2.5` → buy, `<3.5` → hold, `<4.5` → sell, else strong_sell. Cached 24 h. Returns None on subscription absence, network errors, missing field, or any parser exception — failures are swallowed because a missing IBKR read should block the BUY (handled at the risk layer), not crash the scan cycle. Requires the IBKR Reuters Fundamentals subscription on the connected account; without it, BUYs will be blocked because two-source agreement cannot be confirmed.

### Realtime Subscriptions

**`subscribe_realtime(ib, contract, callback)`** — Subscribes to live price updates for a contract. The callback is stored per IB instance and properly removed on unsubscribe to prevent callback leaks across reconnections.

**`unsubscribe_realtime(ib, contract)`** — Cancels a realtime subscription and removes the stored callback.

**`clear_realtime_subscriptions()`** — Resets all tracked subscriptions. Called by the disconnect handler after IBKR drops connections, so stale callbacks from the old connection don't accumulate.

**Snapshot quote cleanup** — After reading a snapshot quote via `get_realtime_quote()`, `cancelMktData` is called to free the IBKR market data slot. Without this, each snapshot permanently consumed one of the limited IBKR data lines.

### Caching

Every data fetch is cached with a time-to-live (TTL):
- Historical bars: cached for 5 minutes (they don't change that fast)
- Quotes: cached for 30 seconds
- News (stock-specific + macro): cached for 1 hour on success, 60 seconds on failure (so retries happen sooner when APIs are down)

This prevents us from hammering the APIs with identical requests within the same scan cycle.

### Error Handling

Tavily API failures are logged at WARNING level (not debug) so they surface in the logs when news fetching degrades.

### Column Normalization

Different data sources return columns with different names (IBKR uses "Close", YFinance might use "close"). This module normalizes everything to lowercase: `open, high, low, close, volume`. This way, downstream code never needs to worry about capitalization.

---

## 8. Building the Stock Universe — `core/universe.py`

Before we can screen stocks, we need a list of stocks TO screen. That's what the universe builder does. It answers: "Which stocks should we even look at today?"

### The Process

```
Step 1: Check cache — did we already build today's universe?
   YES → re-apply filters to cached list (in case filter rules changed)
         and use it (the raw stock list doesn't change intraday)
   NO  → continue to Step 2

Step 2: Ask IBKR to scan the market using 10 different scanner types:
   - MOST_ACTIVE         (highest volume today)
   - TOP_PERC_GAIN       (biggest gainers)
   - TOP_PERC_LOSE       (biggest losers — could be short candidates)
   - HOT_BY_VOLUME       (unusual volume spikes)
   - TOP_OPEN_PERC_GAIN  (gapped up at open)
   - TOP_OPEN_PERC_LOSE  (gapped down at open)
   - HIGH_VS_13W_HL      (near 13-week high)
   - LOW_VS_13W_HL       (near 13-week low)
   - TOP_TRADE_COUNT     (most individual trades)
   - TOP_TRADE_RATE      (fastest rate of trades)

   Each scanner call is bounded by a 15-second timeout (SCAN_TIMEOUT_SECONDS).
   If IBKR hangs on one scanner (observed to block indefinitely in production),
   that scan is skipped and the remaining scanners continue. This prevents a
   single stuck scanner from freezing the entire universe build for hours.

Step 3: Combine all results and remove duplicates.
        A stock that appears in 3 different scanners still only appears once.
        Typical result: ~200-350 unique stocks

Step 3.5: Enrich with contract details.
          The IBKR scanner doesn't return sector or company name data.
          So we call reqContractDetails for each stock to fill in those fields.
          A 0.05-second delay is inserted between each request to avoid
          IBKR pacing violations.
          This is what makes the financial sector filter actually work.

Step 3.6: 4-tier sector fallback for stocks still missing sector data:
          1. yfinance — looks up sector and country via yf.Ticker().info
             For ETFs, checks the "category" field to classify as:
               - "Equity ETF" (SPY, QQQ, IWM — kept in universe)
               - "Bond ETF" (HYG, SGOV — excluded)
               - "Leveraged ETF" (TQQQ, SQQQ, SOXL — excluded)
               - "Non-Stock ETF" (BITO, USO, UVIX — excluded)
          2. Gemini — asks gemini-2.5-flash-lite to classify by sector
             and country (fast ~1s, ~128 tokens). Shares the process-wide
             _gemini_exhausted flag with core.analyst, so an auth failure
             or quota depletion here turns Gemini off everywhere.
          3. Ollama LLM — local fallback for when Gemini is unconfigured,
             exhausted, or returned no sector (slow ~30-60s on CPU,
             ~128 tokens per query)
          4. Exclude — if all four sources fail, the stock is excluded
             since we can't safely filter financials without knowing the sector

Step 4: Filter out:
   - Financial sector stocks (banks, insurance, etc.)
   - Defense/military stocks (weapons, ammunition, combat systems, etc.)
   - Non-equity ETFs (bond, leveraged, inverse, commodity, volatility)
   - Stocks with volume below 100K shares/day
   - Stocks with market cap below $50M
   - Stocks from excluded countries
   - Explicitly excluded tickers

Step 5: Cache the result as a JSON file for the rest of the day
```

### The Fallback Chains

**Universe source fallback** — what if IBKR scanners aren't available?
1. **Try cache** — maybe we already built it today
2. **Try IBKR scanners** — the main approach
3. **Static fallback** — a hardcoded list of ~100 well-known US stocks

**Sector classification fallback** — what if IBKR doesn't know the sector?
1. **IBKR contract details** — `reqContractDetails` returns `category` for most stocks
2. **yfinance** — `yf.Ticker().info["sector"]` for stocks, `.info["category"]` for ETFs
3. **Gemini** — `gemini-2.5-flash-lite` classifies the ticker by sector and country (~1s, ~128 tokens). Skipped without an HTTP call when `GEMINI_API_KEY` is unset, `AI_PROVIDER != "gemini"`, or the shared `_gemini_exhausted` flag is latched.
4. **Ollama LLM** — asks the local AI model to classify by sector and country when Gemini is unavailable or returned nothing
5. **Exclude** — unclassifiable stocks are dropped (can't safely filter financials)

**Stock news priority** — Tavily-first for richer results, yfinance as free fallback:
1. **Tavily API** — primary source; richer headlines from a broader crawl
2. **yfinance** — fallback when Tavily errors (e.g. rate limit), is unconfigured, or returns nothing. `yf.Ticker().news` is free and has no rate limits.

Each call logs which source succeeded (`Tavily OK`, `yfinance OK`, or both-failed), so offline-review can confirm fallback worked.

**Macro/political news** — Tavily only (no free alternative for broad market headlines):
1. **Tavily API** — fetched once per scan cycle, shared across all candidates

This means the system ALWAYS has stocks to trade and context for AI analysis, even if external APIs are flaky.

---

## 9. The Technical Screener — `core/screener.py`

This is the **fast, free first filter**. It runs 6 technical indicators on every stock in the universe and scores them. It takes maybe a second to screen 300 stocks. No AI, no API calls — just math on price data.

### The Architecture Rule

The screener is written as **pure functions**. This is a critical design decision.

What does "pure function" mean? It means the screener:
- Takes data IN (a DataFrame of prices)
- Returns results OUT (a list of Signals)
- Never reaches out to fetch data itself
- Never writes to a database
- Has no side effects

Why? Because the **backtester uses the exact same screener code**. During live trading, the scheduler feeds it today's data. During backtesting, the engine feeds it historical data. Same code, different inputs. Zero duplication.

### The 6 Indicators

Each indicator check looks at the stock's price history and returns either a signal (BUY or SELL) or nothing.

#### 1. RSI (Relative Strength Index) — `check_rsi(df)`

RSI measures how "overbought" or "oversold" a stock is on a scale of 0-100.

- **Below 30** → "This stock has been beaten down. It might bounce back." → BUY signal
- **Above 70** → "This stock has been on a tear. It might pull back." → SELL signal
- **30-70** → Normal territory, no signal

The strength of the signal scales with how extreme the RSI is. RSI of 15 is a stronger buy signal than RSI of 28.

#### 2. MACD (Moving Average Convergence Divergence) — `check_macd(df)`

MACD tracks momentum by comparing two moving averages.

- When the fast MACD line **crosses above** the signal line → momentum is turning bullish → BUY
- When it **crosses below** → momentum is turning bearish → SELL

We specifically check that the crossover happened in the **last 2 bars** (recent crossovers matter, old ones don't).

#### 3. Moving Average Crossover — `check_ma_crossover(df)`

Compares a fast moving average (5-day) against a slow one (20-day).

- **Fast crosses above slow** → "Golden cross" → BUY (short-term trend is now above long-term)
- **Fast crosses below slow** → "Death cross" → SELL

Again, only recent crossovers count (within last 2 bars).

#### 4. Volume Spike — `check_volume_spike(df)`

Looks for days where volume is **2x or more** the 20-day average.

Why does this matter? Big volume means big interest. If a stock suddenly trades 3x its normal volume, something is happening — earnings, news, institutional buying. Volume confirms that a price move is "real" and not just noise.

This check doesn't generate BUY or SELL by itself — it confirms other signals.

MACD requires a minimum data length of `MACD_SLOW + MACD_SIGNAL - 1` bars to produce valid output. Shorter DataFrames are skipped.

#### 5. Bollinger Bands — `check_bollinger(df)`

Bollinger Bands create an envelope around the price:
- Middle band = 20-day moving average
- Upper band = middle + 2 standard deviations
- Lower band = middle - 2 standard deviations

Statistically, price stays within the bands ~95% of the time.

- **Price drops below lower band** → "Unusually cheap" → BUY
- **Price rises above upper band** → "Unusually expensive" → SELL

#### 6. Support & Resistance — `check_support_resistance(df)`

Looks at the stock's 20-day high and low.

- **Price within 2% of the 20-day low** → Near support → BUY (might bounce), but only if today's intraday low hasn't breached the support level (a broken support is bearish, not a buying opportunity)
- **Price within 2% of the 20-day high** → Near resistance → SELL (might reverse)

### Scoring

Each stock gets a score from 0 to 100 based on:

1. **Weighted indicator count** — More signals = higher score, but each indicator's contribution is multiplied by its weight from `INDICATOR_WEIGHTS`. If RSI (weight 2.0), MACD (weight 1.0), and volume (weight 1.0) ALL say "buy", RSI contributes twice as much to the score.
2. **Weighted signal strength** — Each indicator returns a strength between 0 and 1, multiplied by its weight. A deeply oversold RSI of 15 with weight 2.0 scores much higher than a barely-oversold RSI of 29 with weight 0.5.
3. **Consensus** — Are all signals pointing the same direction? Opposing signals actively reduce the score: `net_score = direction_signals - opposing_signals`. This prevents conflicting indicators from producing a falsely confident signal.

Indicator weights are configurable in `config/settings.py` via `INDICATOR_WEIGHTS` (dict mapping indicator name → float weight). Default is 1.0 for all indicators (equal weighting). Set a weight to 0.0 to disable an indicator. The backtester also accepts custom weights via `BacktestConfig.indicator_weights` for A/B testing different weight profiles.

Stocks scoring above the minimum threshold (default: 15.0) become candidates and get passed to the AI analyst.

### ATR-Based Stop Losses

Instead of using a fixed 3% stop-loss for every stock, the screener uses **ATR (Average True Range)** — a measure of how much a stock typically moves in a day.

A volatile stock like Tesla might move 5% in a normal day, so a 3% stop-loss would get triggered by normal noise. ATR says: "This stock normally moves $X per day, so set the stop-loss at 1.5x that distance below entry."

This adapts the stop-loss to each stock's personality.

---

## 10. The AI Analyst — `core/analyst.py`

The AI analyst is the "smart filter." It takes candidates from the screener and does deep qualitative analysis using a local Large Language Model.

### Why Gemini + Ollama?

The analyst routes through **Gemini** (primary) with **Ollama + Qwen 2.5 7B** as an automatic fallback:

- **Gemini (primary)** — cloud, fast (~2-5s per analysis), capable. Gemini Flash-Lite is cheap-to-free on light workloads. Works when `GEMINI_API_KEY` is set.
- **Ollama (fallback)** — local, slower (~30-60s on CPU, 5-10x faster on GPU), free, works offline. Kicks in automatically when Gemini is unreachable, rate-limited, has depleted credits, or is not configured.

This was a pivot from the original cloud-only Claude/GPT design (see Challenges section). Keeping Ollama as the fallback preserves offline-capable, zero-cost resilience, while Gemini as the primary gives us significantly lower latency per scan cycle and higher-quality reasoning without the old cost-per-analysis anxiety.

#### How the fallback routes

`core/analyst._call_llm()` is the router:

1. If `AI_PROVIDER == "gemini"`, at least one Gemini key is configured (`GEMINI_API_KEYS` or `GEMINI_API_KEY`), and the process-lifetime exhaustion flag `_gemini_exhausted` is not latched — try Gemini up to `max_retries` times for content-level failures (malformed JSON, invalid response), falling straight through to Ollama on any transport failure (HTTP 5xx, network error, all-keys-exhausted).
2. Otherwise (or after Gemini gives up) call Ollama up to `max_retries` times.

Two failure modes drive routing:

- **Transport failures** raise `_GeminiTransportError` — router breaks out of the Gemini leg immediately, no retry loop on stateless server errors.
- **Content failures** return `None` — router retries Gemini up to `max_retries` because a re-prompt may produce a parseable response.

#### Multi-key rotation

`_call_gemini` rotates round-robin across every key in `GEMINI_API_KEYS` so the free-tier RPD ceiling (1,000/day per key) scales linearly with the number of configured keys — three keys ≈ 3,000 RPD/day pool, comfortably above the bot's ~2,688 calls/day at 15-min cadence. Each call advances a thread-safe cursor by one position; per-call failure handling iterates from that starting index through every active key:

- **RPD/auth (permanent) on a key** — `_call_gemini_with_key` raises `_GeminiKeyExhausted`; the rotation latches THAT key's per-key flag and tries the next active key. The global `_gemini_exhausted` is set only when EVERY key is exhausted, which is the trigger for the Ollama fallback path.
- **RPM (transient) on a key** — `_call_gemini_with_key` raises `_GeminiKeyTransient`; the rotation tries the next active key without latching anything. If every active key returns RPM 429 in the first pass *and* the deployment has more than one key configured, sleep `_GEMINI_RPM_RETRY_SLEEP` (30 s) once and try a second pass. Single-key deployments skip the sleep entirely — there is no sibling to recover.

Single-key deployments fall back to a one-element rotation, so the legacy `GEMINI_API_KEY=...` form keeps working unchanged.

#### Exhaustion markers (permanent vs transient)

`_GEMINI_EXHAUSTION_MARKERS` is a tuple of substring matchers (case-insensitive) that classify a 429 body as PERMANENT (RPD/quota/credits depleted) vs TRANSIENT (RPM rate limit). Permanent matches latch the per-key flag; transient ones do not. The list is intentionally tighter than Tavily's because Gemini emits "Quota exceeded per minute" for transient per-minute rate limits, so a bare `"quota"` marker would false-positive and lock out a key on a recoverable condition. The 2026-04-27 production incident — `"You exceeded your current quota"` body — was missed by every existing marker, so the flag never latched and every candidate burned a wasted Gemini round-trip; the fix is to add `"exceeded your current quota"` and `"requests per day"` to the marker list while keeping `"per minute"` *out* of it.

In addition to RPD-marker 429s, HTTP 401/403 (invalid key / missing permissions) also raises `_GeminiKeyExhausted` — that key is dead for the rest of the process, but sibling keys keep working.

#### Structured output (`_gemini_response_schema`)

Gemini 2.5 Flash-Lite was observed dropping `trade_type` on a majority of responses, triggering validator retries that wasted time on re-prompts. The fix: every `_call_gemini` payload now includes a `responseSchema` in `generationConfig` that lists `trade_type` (and every other required field) as mandatory. The `action` enum adapts to `ALLOW_SHORT_SELLING` — when shorts are disabled the schema only offers `["buy", "hold"]`, closing the same loophole the prompt already closed so the model can't emit a `sell` even if it wanted to.

#### Per-provider token tracking

`get_daily_token_usage()` now returns a per-provider breakdown:

```python
{
  "gemini": {"input": N, "output": N},
  "ollama": {"input": N, "output": N},
  "date": "YYYY-MM-DD",
}
```

Counters reset at local midnight (`_reset_daily_usage_if_needed`). This is a shape change from the prior flat `{input, output, date}`; no in-repo callers read the old shape, so the switch is safe. External consumers (if any) will need to update.

### How Analysis Works

For each candidate stock, the analyst:

#### Step 1: Build the Prompt

The prompt is a structured document that gives the AI everything it needs:

```
"You are a disciplined stock trader making real money decisions..."

Stock: AAPL (US)

Recent Price Action (last 5 days):
  2026-03-27: O=174.50 H=176.20 L=173.80 C=175.90 V=45.2M
  2026-03-28: O=175.90 H=178.30 L=175.10 C=177.80 V=52.1M
  ...

Technical Indicators:
  - RSI(14) = 35.2 → BUY signal (oversold)
  - MACD crossover: bullish (1 bar ago)
  - Volume: 2.3x average (spike confirmed)

News Headlines:
  - "Apple announces record services revenue"
  - "iPhone 17 leaks suggest major camera upgrade"

Macro/Political Headlines:
  - "Fed holds rates steady amid inflation concerns"
  - "US-China trade talks resume after tariff escalation"

Decision Checklist — evaluate each:
  1. TREND: Is the stock in a clear trend?
  2. MOMENTUM: Is momentum confirming?
  3. VOLUME: Is there volume confirmation?
  4. RISK/REWARD: Is reward >= 1.5x risk?
  5. NEWS: Any catalysts or red flags?
  6. ANTI-CHASE: Has it already moved >5%?
  7. MACRO/POLITICAL: Do macro conditions create risk or opportunity?
```

#### Step 2: Call the AI

Routes through the provider router: Gemini's `generateContent` endpoint first (when configured and not exhausted), with an automatic fallback to the local Ollama `/api/generate` endpoint. Either way the response is structured JSON:

```json
{
  "action": "buy",
  "confidence": 78,
  "entry_price": 177.50,
  "stop_loss": 173.00,
  "take_profit": 185.00,
  "reasoning": "Strong oversold bounce with volume confirmation...",
  "trade_type": "swing"
}
```

#### Step 3: Validate the Response

The system checks:
- Is `action` one of buy/sell/hold?
- Is `confidence` a number between 0 and 100?
- Are prices provided (for buy/sell)?
- Is `stop_loss` below `entry_price` (for buys)?
- Is `take_profit` above `entry_price` (for buys)?
- Is `trade_type` one of "day" or "swing"?

Each provider's JSON response is parsed inside a try/except that catches both `KeyError` (missing fields) and `JSONDecodeError` (malformed output), logging specific error messages. Gemini's response is doubly nested — the outer envelope has `candidates[0].content.parts[0].text`, which is itself a JSON string (because we set `responseMimeType: application/json` in the request).

Invalid responses are rejected and retried up to 3 times. Gemini transport failures (HTTP 5xx, network error, auth, credits depleted) skip the retry loop and fall straight through to Ollama instead — retrying stateless server errors rarely helps and wastes scan-cycle latency.

#### Step 4: Filter by Confidence

Only signals with confidence >= 65 pass through. The AI is encouraged to be honest — if it's uncertain, it should say confidence 40, and we'll skip it.

### The Discipline Rules

These are embedded in the prompt to prevent common trading mistakes:

1. **Require 5/7 checklist items favorable** — Don't buy just because RSI is oversold. Need multiple confirmations.
2. **Anti-chase** — If a stock already moved 5%+ in the direction of the signal, reject it. You missed the move.
3. **Conservative confidence** — Only give 65+ when trend + momentum + volume align. Be honest about uncertainty.
4. **No FOMO** — It's okay to say HOLD. Missing a trade is better than taking a bad one.

### Batch Processing

The `analyze_batch()` function processes multiple candidates sequentially, collecting all AI-approved signals before passing them to the risk manager.

---

## 11. The Risk Manager — `core/risk.py`

The risk manager is the **last line of defense** before real money moves. Even if the screener loves a stock and the AI says "buy with 85% confidence," the risk manager can still say "no" if any safety rule is violated.

Like the screener, it's **pure functions** — takes portfolio state as input, returns approval/rejection. No data fetching, no side effects. Same code works in live trading and backtesting.

### The 15 Safety Checks

Every signal must pass ALL of these. Fail one, the trade is rejected.

#### 1. Short Selling Block — `check_short_selling()`
"Is this a sell signal for a stock we don't own?"

Rule: If `ALLOW_SHORT_SELLING` is False (the default), reject any SELL signal where we don't already hold the stock.

Why: Short selling (selling borrowed shares hoping the price drops) carries unlimited downside risk — a stock can rise infinitely. With a local AI model, the risk of a bad short call is too high. This check is configurable via `ALLOW_SHORT_SELLING` in settings for advanced users who understand the risks.

Defense-in-depth: the analyst (`core/analyst.py`) also gates on `ALLOW_SHORT_SELLING`. When `False`, the LLM prompt's Response Format offers only `"buy" or "hold"` (dropping `"sell"`) and the Discipline Rules include an explicit "do not short" line; `_validate_response` rejects any SELL that slips through without retry. This prevents the LLM from wasting 200+s of compute on shorts the risk manager would reject anyway.

#### 2. Position Size Check — `check_position_size()`
"Is this trade too big relative to our portfolio?"

Rule: No single position can be more than 5% of total portfolio value.

Why: If one stock crashes, you lose at most 5% of your portfolio, not 50%.

#### 3. Daily Loss Limit — `check_daily_loss_limit()`
"Have we already lost too much today?"

Rule: If today's losses exceed 2% of portfolio value, STOP TRADING. No more new positions.

Why: Bad days happen. This prevents emotional revenge-trading. If you're down 2%, the system shuts off and you live to trade another day. This is the single most important safety feature.

#### 4. Max Open Positions — `check_max_positions()`
"Do we have too many open positions?"

Rule: Maximum 10 open positions at once. Exit signals (SELL on existing long, BUY on existing short) are always allowed through — they reduce positions, not add new ones.

Why: More positions = more to monitor = more risk of something slipping through. Also keeps the portfolio manageable. The exit exemption prevents a situation where a stop-loss or take-profit signal is blocked when the portfolio is at max capacity.

#### 5. Stop-Loss Validation — `check_stop_loss()`
"Does this signal have a valid stop-loss?"

Rule: Every trade MUST have a stop-loss. For buys, stop-loss must be below entry price. For sells, above.

Why: A trade without a stop-loss has unlimited downside. Never acceptable.

#### 6. Sector Concentration — `check_sector_concentration()`
"Are we too heavy in one sector?"

Rule: No more than 25% of portfolio in a single sector (like Technology or Healthcare). The check includes the proposed new position's estimated value (worst-case max position size) to prevent the first position in a sector from bypassing the limit.

Why: If all your money is in tech stocks and tech crashes, everything drops together. Diversification protects you.

#### 7. No Duplicates — `check_no_duplicate()`
"Do we already have a position in the same direction?"

Rule: Can't open a new long in AAPL if we already hold AAPL long. But a SELL signal on an existing long position is allowed (it's closing the position, not opening a new one).

Why: Prevents doubling down on a losing position (a common emotional mistake) while still allowing position exits through the normal signal pipeline.

#### 8. Excluded Sector — `check_excluded_sector()`
"Is this stock in a sector we don't trade?"

Rule: No financial sector stocks (banks, insurance, lending) and no defense/military stocks (weapons, ammunition, combat systems). Also checks the `EXCLUDED_TICKERS` list — tickers that are explicitly blocked are rejected here too, not just in the universe builder.

Why: Safety net. Even if IBKR's sector data is wrong and a bank or defense contractor slips through the universe filter, this catches it at the risk level. The explicit ticker check provides a third layer of defense for known problematic symbols.

#### 9. Anti-Momentum — `check_anti_momentum()`
"Has this stock already moved too much?"

Rule: Reject if the current price has already moved more than 5% from the signal's entry price. Also rejects signals with zero or invalid prices — these indicate data problems and must not bypass risk checks.

Why: Chasing. If the screener flagged TSLA at $200 but by the time we get to risk check it's at $212, we missed the move. Buying now means we're chasing and likely buying at a local top.

#### 10. Trend Confirmation — `check_trend_confirmation()`
"Are the moving averages aligned in our favor?"

Rule: For buys, need MA5 > MA10 > MA20 (short-term above long-term = uptrend). For sells, reversed.

Why: Trading against the trend is fighting the market. This check ensures we're swimming WITH the current.

#### 11. Risk/Reward Ratio — `check_risk_reward()`
"Is the potential profit worth the potential loss?"

Rule: (take_profit - entry) / (entry - stop_loss) must be >= 1.5. Signals with zero or invalid prices are rejected rather than skipping the check.

Why: If you risk $1 to make $1, you need to be right >50% of the time to profit. At 1.5:1, you only need to be right ~40% of the time. The math works in your favor.

#### 12. Circuit Breaker — `check_circuit_breaker()`
"Have we been losing too many trades in a row?"

Rule: If the last 3 consecutive closed trades (within a 60-minute window) are all losses, pause all new trading and send a Telegram alert. A win or breakeven trade resets the streak. Both the loss count and time window are configurable via `CIRCUIT_BREAKER_LOSSES` and `CIRCUIT_BREAKER_WINDOW_MIN`. Trades missing an `exit_time` attribute are safely skipped to prevent crashes when the trade list contains incomplete records.

Why: The daily loss limit is reactive — it triggers after you've already bled money. The circuit breaker is proactive: 3 rapid losses in a row usually signals something systemic (market regime change, stale data feed, broken model). Better to pause and review than keep firing. Think of it as a smoke detector vs. a fire extinguisher.

#### 13. Analyst Consensus — `check_analyst_consensus()`
"Do Wall Street analysts agree this is a buy — across two independent sources?"

Rule: Before buying, fetch the analyst consensus from **two** sources: yfinance's `recommendations_summary` (Yahoo's republished sell-side ratings) and IBKR's Reuters/Refinitiv RESC report (`reqFundamentalData(contract, 'RESC')`, the `ConsRecom` 1.0–5.0 mean rating mapped to the same `strong_buy`/`buy`/`hold`/`sell`/`strong_sell` vocabulary). The BUY is allowed **only when both sources agree on `buy` or `strong_buy`**. Anything else — `hold` or worse from either side, *or* missing data on either side — blocks the BUY. SELL and HOLD signals always pass (we don't want to gate exits on analyst data). Controlled by `CHECK_ANALYST_CONSENSUS` setting (default: True).

Why: The previous rule allowed `hold` and "no data" to pass, which let too many marginal trades through. Tightening to "buy/strong_buy from both sources" trades trade-frequency for win-rate: if Yahoo and Refinitiv both lean bullish, the entry has a much higher chance of working. Treating missing data as a block is deliberate — when only one source has an opinion, we can't confirm two-source agreement, and small-cap/newly-listed coverage is exactly where the worst losing entries live. Both sources are cached for 24 hours since analyst ratings change infrequently.

#### 14. Correlation Cap — `check_correlation()`
"Is this new position really independent, or is it just another copy of something we already hold?"

Rule: Compute Pearson correlation between the candidate's recent daily returns and each open position's daily returns. If the **maximum** correlation exceeds `CORRELATION_CAP_THRESHOLD` (default 0.7), reject the new entry. Callers pass a `returns_lookup: dict[str, pd.Series]`; if no data is provided for the candidate or for all existing positions, the check is skipped. Negative correlation (hedges) always passes. Setting the threshold to 1.0 or above disables the check.

Why: If you already own NVDA and buy AMD, both semiconductor stocks move together. Tomorrow's chip-sector news hits both positions simultaneously — so two "independent" positions are really one big bet. Institutional risk systems measure and constrain correlation between holdings; this check brings a simple version of that discipline into the bot. Uses a default `min_periods=20` to avoid unreliable correlations on short series, and gracefully handles NaN values (produced by halted or missing bars) by skipping them.

#### 15. Intraday-Margin Guard — `check_intraday_margin()`
"Would this entry create or leave an uncured intraday-margin deficit?"

The FINRA PDT rule was eliminated 2026-06-04. This check replaces it with intraday-margin awareness to guard against the 90-day restriction that IBKR imposes for uncured intraday-margin deficits.

Rule: Two margin thresholds are enforced on new entries (exits always pass):

1. **Reg-T minimum equity (`REG_T_MIN_EQUITY_USD`, default $2,000).** If portfolio equity is below $2,000, new entries are blocked — the account cannot legally trade on margin under Regulation T. Exits are always allowed.
2. **Intraday maintenance margin floor (`INTRADAY_MAINTENANCE_MARGIN_PCT`, default 25%).** The check computes whether adding the proposed position would push the account's margin ratio below the 25% intraday maintenance floor. Entries that would create an uncured deficit are rejected; exits always pass.

`MARGIN_REGIME` (default `"both"`) controls which checks run:
- `"intraday"` — only the two guards above.
- `"legacy_pdt"` — only the legacy `LEGACY_PDT_THRESHOLD_USD` ($25,000) gate.
- `"both"` — all checks run simultaneously (recommended during IBKR's phase-in period through 2027-10-20).

`LEGACY_PDT_THRESHOLD_USD` is set to $25,000 — the correct legacy Pattern Day Trader threshold. Under `"legacy_pdt"` or `"both"`, accounts below this value face additional day-trade restrictions as defined by the legacy gate logic.

Exits of positions opened on a prior day are **always** allowed — blocking exits would trap the trader in a losing position.

Why: Uncured intraday-margin deficits trigger a 90-day closing-orders-only restriction at IBKR regardless of account size. With PDT eliminated, the margin floor ($2,000 Reg-T equity + 25% intraday maintenance) is the real gating condition that must be respected. `MARGIN_REGIME = "both"` provides a safety net during the broker phase-in so the legacy guard remains active for accounts that haven't fully migrated to the new regime.

### Exit Signal Bypass

The `evaluate()` function detects whether a signal is an **exit** (closing an existing position) rather than a new entry. When a SELL signal targets a held long position, or a BUY signal targets a held short position, the following checks are **skipped**:

- Position Size (`check_position_size`) — we're not opening, just closing
- Cumulative Risk (`check_cumulative_risk`) — exits REDUCE total open risk, not add to it
- Sector Concentration (`check_sector_concentration`) — exits REDUCE concentration, not add to it
- Excluded Sector / Ticker (`check_excluded_sector`) — legacy positions must remain exitable even after the ticker/sector is added to an exclusion list
- Anti-Momentum (`check_anti_momentum`)
- Trend Confirmation (`check_trend_confirmation`)
- Risk/Reward Ratio (`check_risk_reward`)
- Analyst Consensus (`check_analyst_consensus`)
- Correlation Cap (`check_correlation`)

Why: Most of these checks gate on market conditions and only make sense for deciding whether to **enter** a new position. Applying them to exits creates a dangerous trap — if the market moves against you, the very conditions that make you want to exit (price dropped, trend reversed, sector now crowded) are the same conditions these checks reject. Without this bypass, a losing position could never be closed because the anti-momentum check would say "price already dropped too far", the cumulative-risk check would see the existing position PLUS phantom new risk, and the excluded-sector check would keep the trader trapped in legacy financial-sector holdings.

Core safety checks (daily loss limit, stop-loss coherence, max positions, no-duplicate, circuit breaker, intraday-margin) still apply to all signals.

For exits, `evaluate()` also sizes the order using the **existing position's absolute quantity** rather than a freshly-calculated new-entry size. If we re-calculated, the "risk-budget" sizing could produce a quantity LARGER than the holding and flip the position to a net short — a position inversion that violates `ALLOW_SHORT_SELLING=False` and is never the user's intent.

### Position Sizing

If all checks pass, the risk manager calculates HOW MANY shares to buy:

```
Method 1: Max position = 5% of portfolio / entry price
Method 2: Risk-based = 1% of portfolio / distance to stop-loss

Use the SMALLER of the two.
```

Method 2 is the clever one. It says: "If I'm wrong and my stop-loss gets hit, I want to lose at most 1% of my portfolio." So if the stop-loss is very tight (close to entry), you can buy more shares. If it's wide, you buy fewer. This is professional-grade position sizing.

#### Volatility Regime Scaling

When market volatility is provided (via `volatility` parameter to `evaluate()` or `calculate_position_size()`), position sizes are scaled inversely to realized volatility:

```
vol_scale = min(VOLATILITY_BASELINE / current_volatility, 1.0)
adjusted_quantity = base_quantity * vol_scale
```

Key properties:
- **High volatility** (e.g., 40% annualized) → positions shrink (scale = 0.5 with 20% baseline)
- **Low volatility** (e.g., 10% annualized) → positions stay at base size (capped at 1.0, no leverage)
- **No volatility data** (`None`) → original sizing (backward compatible)

The baseline annualized volatility is configurable via `VOLATILITY_BASELINE` (default: 20%). `calculate_realized_volatility(closes, window=20)` computes this from a close-price series using 20-day rolling log returns, annualized by √252.

### The RiskResult

The output is a simple object:
```python
RiskResult:
  approved: True/False
  reasons: ["Daily loss limit breached", ...]  # empty if approved
  position_size: 45  # shares (existing quantity for exits, calculated for entries)
  is_exit: True/False  # True when signal closes an existing position
```

The `is_exit` flag lets the executor route exits to a plain close order rather than a bracket. A bracket's SL/TP children would stay live at IBKR after the parent closes our position — those orders could later re-enter the ticker when price crossed the child's trigger.

---

## 12. Executing Trades — `core/executor.py`

Once the risk manager approves a trade, the executor places actual orders on IBKR.

### Bracket Orders

The primary order type is a **bracket order** — three linked orders placed together:

```
1. PARENT ORDER: Buy 100 shares of AAPL at $175.50 (limit order)
2. TAKE-PROFIT: Sell 100 shares of AAPL at $185.00 (limit order)
3. STOP-LOSS: Sell 100 shares of AAPL at $170.00 (stop order)

These are linked:
- Parent fills first
- Then TP and SL become active
- When TP fills → SL automatically cancels (and vice versa)
```

This is **atomic** — all three orders are placed as a unit. You never end up with a position that has no stop-loss, even for a millisecond.

### Key Functions

**`place_order(ib, signal, quantity, dry_run)`** — The main function. Creates a bracket order from the signal's entry, stop-loss, and take-profit prices. All legs use `tif='GTC'` (Good Till Cancelled) so orders placed outside market hours persist and execute at market open. The parent and take-profit orders are placed with `transmit=False`, and only the stop-loss (last order) has `transmit=True` — this ensures all three legs transmit atomically to IBKR, preventing the parent from filling before child orders are registered. In dry-run mode, it just logs what WOULD happen.

**`close_position_market(ib, position, dry_run)`** — Immediately close a position with a market order. Derives the closing action from the position's quantity sign (SELL for long, BUY for short). Used when day-trade positions need to be closed before market close.

**Exit signals bypass `place_order` entirely.** When `RiskResult.is_exit` is true, the scheduler calls `place_market_order()` (no bracket) and attaches a `setup_exit_close_handler()`. Reason: `place_order`'s bracket attaches TP/SL children with `transmit=True` on the stop-loss leg, and IBKR keeps those children alive after the parent fills. If the parent is a SELL-to-close-long, the TP/SL are BUY orders; when price later crosses either child's trigger, IBKR re-enters the ticker as a fresh long. Routing exits through a single market order eliminates the tail.

**`setup_exit_close_handler(ib, signal, close_trade, on_exit)`** — Attaches a fill handler for the AI-driven market close. Matches on the close order's `permId` (or `orderId` + ticker as a fallback), not `parentId > 0`, because the market close is the parent itself with no children. On fill, calls `db_close_position` to record the exit at the actual fill price, fires the optional `on_exit` callback, and deregisters. Includes a fast-fill race guard: if the close is already "Filled" at handler attachment time (rare but possible on very liquid names during IBKR's permId poll), the handler fires synchronously with `_fired` flag protection to prevent double-processing.

**`close_all_day_trades(ib, positions, dry_run)`** — Called 15 minutes before market close. Finds all positions marked as DAY trades, cancels ALL open orders for those tickers at IBKR (unfilled parents AND orphaned TP/SL children from already-filled brackets) to prevent race conditions, then closes each position with a market order. After fill verification via `monitor_orders()` (30-second timeout), records each close in the portfolio DB via `db_close_position()` at the actual fill price. Logs at ERROR level for any unfilled orders — critical because unfilled orders mean positions stay open overnight.

**`handle_fill(signal, quantity, fill_price)`** — Records a filled order in the SQLite database. Returns `None` if the fill quantity is zero or negative, preventing phantom positions from being recorded. IBKR always reports filled quantity as a positive integer regardless of trade direction. For short entries (SELL side), the stored quantity is negated so that P&L math `(exit − entry) × qty`, position reconciliation, and close-side selection all work correctly without special-casing shorts.

**`setup_fill_handler(ib, signal, quantity, on_fill)`** — Attaches an async callback to handle entry order fills. Only processes parent entry orders (`parentId == 0`) matching the signal's ticker — child orders (take-profit, stop-loss) are handled by `setup_exit_handler` instead. Uses a `fired` flag to ensure each handler instance fires at most once, preventing duplicate position recording when multiple signals for the same ticker are active. When the parent order status changes to "Filled", it records the position in the database, calls the optional `on_fill` callback, and deregisters itself from `ib.orderStatusEvent` to prevent handler accumulation across scan cycles. The post-fill logic (handle_fill, remove_pending_order, callback, unsubscribe) is wrapped in try/finally to ensure the event handler is always cleaned up even if a callback raises an exception. Also logs warnings for partial fills (when filled_qty < requested quantity). The scheduler attaches fill and exit handlers BEFORE the rejection check sleep (not after) to close a race window where fast fills could fire before handlers are registered. Handlers only act on "Filled" status, so they are harmless no-ops for rejected orders.

**`setup_exit_handler(ib, signal, on_exit)`** — Attaches a callback to handle exit order fills (take-profit and stop-loss). When a child order fills, it closes the position in the database via `portfolio.close_position()`, which also writes to the CSV trade journal. Logs a warning if the exit fill cannot be matched to a database position (e.g., position already closed or missing). Warns when `parent_order_id` is missing on the exit trade, as this risks cross-bracket interference (matching the wrong bracket's exit to a position). Calls the optional `on_exit` callback for Telegram notifications. Uses try/finally to ensure the event handler is always deregistered even if callback errors occur.

**`import_ibkr_positions(ib, orphaned_tickers)`** — Imports IBKR positions that exist at the broker but not in the DB. Reads `avgCost` and quantity from `ib.positions()`, extracts stop-loss (STP/STP LMT child orders) and take-profit (LMT child orders) from `ib.openTrades()`. Positions without bracket orders get SL/TP defaulted to 0. Trade type defaults to SWING since the original intent is unknown.

**`reattach_exit_handlers(ib)`** — Called at startup to re-register exit handlers for existing bracket orders. In-memory event handlers don't survive bot restarts, so this function scans `ib.openTrades()` for child orders (stop-loss/take-profit) that match open DB positions and attaches fresh exit handlers with Telegram cache refresh callbacks. Deduplicates by parent order ID so each bracket gets exactly one handler (not one per child), preventing duplicate `db_close_position` calls when a bracket has both TP and SL children. Returns the number of handlers attached. Uses `make_on_exit()` factory to avoid closure variable capture issues in the loop.

**`setup_disconnect_handler(ib, reconnect=True)`** — Attaches a callback for connection drops. Important because IBKR connections are notoriously unstable. Uses a re-entrancy guard (`_reconnecting` flag) to prevent cascading reconnect loops where a reconnect triggers another disconnect event. Calls `clear_realtime_subscriptions()` to reset subscription tracking after IBKR drops connections, preventing stale callbacks from the old connection from accumulating. Reads the shared `shutting_down` flag from `core.state` to skip reconnection during intentional shutdown (avoiding the circular import that would result from importing directly from `scheduler.py`). When `reconnect=False`, manual reconnect is skipped entirely — used in Watchdog mode where IBC manages the gateway restart; a competing manual reconnect fails with "clientId already in use".

### Stale Order Re-evaluation

**`get_stale_orders(ib, stale_minutes)`** — Queries IBKR via `ib.openTrades()` for all unfilled parent limit orders (where `parentId == 0` and `orderType == "LMT"` with status `Submitted` or `PreSubmitted`). Looks up order age from the `pending_orders` database table first (persisted at placement time), falling back to `trade.log[0].time` for orders placed before this tracking was added. The DB lookup is critical because `ib_insync` resets the trade log on every reconnection, which would otherwise make all orders appear brand new. Returns a list of dicts with the Trade object, ticker, exchange, and age in minutes for orders older than the threshold.

**`cancel_bracket_order(ib, trade)`** — Cancels a parent entry order. IBKR automatically cancels the attached child orders (take-profit and stop-loss) when the parent is cancelled. Also removes the `pending_orders` DB record for the cancelled order. Follows the same try/except pattern as `cancel_order()`.

The scheduler's `check_stale_orders()` orchestrates the full flow: for each stale order, it fetches fresh historical data, re-runs the screener, and cancels orders where the stock no longer passes technical screening. This runs at the beginning of every scan cycle to free up capital and position slots before new candidates are evaluated. In dry-run mode, the cancelled-order counter only increments when orders are actually cancelled, not when they merely fail screening.

### Cash-Reserve Accounting and Eviction

**`get_pending_buy_reserve(ib)`** — Returns the total cash committed to unfilled parent BUY limit orders (`parentId == 0`, `orderType == "LMT"`, status `Submitted`/`PreSubmitted`). Sums `totalQuantity * lmtPrice` per qualifying order. Needed because IBKR's `TotalCashValue` does not reflect bracket reserves until a fill lands; without this subtraction the risk manager can greenlight a second BUY that IBKR then rejects with Error 201 ("Cash needed for this order and other pending orders").

**`get_pending_buy_orders(ib)`** — Like `get_pending_buy_reserve` but returns a list of dicts (`trade`, `perm_id`, `ticker`, `confidence`, `reserved_cash`) — the DB join exposes the AI confidence that approved each order. Pre-migration orders with no DB row come back with `confidence=None`, treated as zero for ranking.

**`evict_weakest_pending(ib, new_confidence, needed_cash, margin=5)`** — Cancels the lowest-confidence pending BUY only when the new candidate beats it by at least `margin` (default 5 points) AND cancelling it frees enough cash to cover the new order. v1 cancels at most one order — if a single eviction wouldn't free enough cash, it declines rather than chain-cancelling. Returns `True` when an order was cancelled. The scheduler calls this from `_on_signal` immediately after detecting a cash shortfall and retries the cash check before placement; if eviction declines, the candidate is logged and skipped. Short entries and exit signals bypass the entire gate because they don't consume settled cash.

### Dry-Run Mode

When the system runs in `dry-run` mode, the executor does everything EXCEPT actually calling `ib.placeOrder()`. It logs the exact order it would have placed. This is invaluable for testing the full pipeline without risking money.

---

## 13. The Scheduler — Tying It All Together — `core/scheduler.py`

The scheduler is the **conductor** of the orchestra. It doesn't play any instrument itself — it tells each module when to play and passes data between them.

### Market Hours

The scheduler knows when markets are open:

```python
US Market: 16:30 - 23:00 Turkey time (9:30 AM - 4:00 PM Eastern)
```

(The timezone is Turkey because that's where the developer is located.)

It only runs scan cycles during market hours. On weekends, it sleeps. With the `--force` flag, it bypasses both the market-open check and the end-of-day close check, allowing the full pipeline to run outside market hours (orders queue as GTC for the next open).

### The Scan Cycle — `run_scan_cycle()`

This is the heart of the system. Called every 15 minutes:

```python
def run_scan_cycle(ib, markets, mode="paper", force=False):
    # 1. Make sure we're connected
    ensure_connected(ib, ...)

    # 1.5 Re-evaluate stale unfilled orders (cancel if they no longer pass screening)
    check_stale_orders(ib, mode)

    # 2. Get account info
    account = get_account_summary(ib)
    portfolio_value = account["NetLiquidation"]
    daily_pnl = account["RealizedPnL"] + account["UnrealizedPnL"]

    # 3. Build today's universe (cached)
    universe = build_universe(ib, markets)

    # 4. For each active market (or all markets if force=True):
    for market in active_markets:

        # 4.5 Close day trades near market close (skipped in force mode)
        if not force and minutes_to_close(market) <= 15:
            close_all_day_trades(ib, open_positions)
            continue

        # 5. Fetch 60 days of data for every stock
        stock_data = {}
        for stock in universe[market]:
            df = get_historical_data(ib, contract, "60 D", "1 day")
            stock_data[stock.ticker] = (stock.exchange, df)
        sector_lookup = {s.ticker: s.sector for s in universe[market]}

        # 6. Run the screener (then inject sector from universe into candidates)
        candidates = screen_stocks(stock_data)
        for sig in candidates:
            sig.indicator_values["sector"] = sector_lookup.get(sig.ticker, "")

        # 7. AI analysis on candidates (progress tracked via on_progress callback)
        #    The _on_ai_progress callback binds `market` via default argument
        #    to avoid loop variable capture bugs
        ai_signals = analyze_batch(candidates, on_progress=update_ai_progress)

        # 8. Risk check and execute
        for signal in ai_signals:
            result = evaluate(signal, open_positions, portfolio_value, daily_pnl)
            if result.approved:
                place_order(ib, signal, result.position_size, dry_run=...)
                setup_fill_handler(ib, signal, ...)  # async: records + notifies on actual fill
                notify_trade(signal, ..., action_type="SUBMITTED")
```

### The Main Loop — `start_scheduler()`

```python
def start_scheduler(ib, markets, mode="paper"):
    # Handle Ctrl+C gracefully
    setup_signal_handlers()

    last_reconcile_date = None
    while not shutting_down:
        if any market is open:
            run_scan_cycle(ib, markets, mode)

        # Nightly reconciliation — once per day, after all markets close
        today = now().date()
        if last_reconcile_date != today and no markets open:
            run_nightly_reconciliation(ib)
            last_reconcile_date = today

        # Sleep using ib_insync's event loop (not time.sleep!)
        ib.sleep(SCAN_INTERVAL_MINUTES * 60)
```

### Nightly Reconciliation — `run_nightly_reconciliation()`

The startup/reconnect-time sync in `main.py` auto-closes positions that IBKR no longer holds, because the bot was offline and something happened (stop-loss hit, manual intervention). But during normal, continuous operation, drift can still creep in — a missed fill callback, a double insert, a manual trade someone made in TWS while the bot was running. Without a periodic check, that drift compounds silently until something breaks.

`run_nightly_reconciliation(ib, db_path)` runs once per day after all markets close and performs a **read-only** check: it fetches `ib.positions()` and `ib.fills()`, calls `reconcile_positions` with `auto_fix=False`, and sends a Telegram alert via `notify_reconciliation_mismatch` on any discrepancy. It never modifies the database — because during normal operation, unexpected divergence is usually a bug or a human action, and the safe response is to tell the operator, not to silently "correct" it.

The Telegram alert lists every orphaned ticker (DB-only and IBKR-only) and every quantity mismatch. **Sign mismatches** — where the DB says we're long and IBKR says we're short (or vice versa) — are escalated to a CRITICAL alert because they indicate the bot's understanding of its position is fundamentally wrong, and continuing to trade blind is dangerous.

The scheduler uses a `last_reconcile_date` guard to ensure the job runs exactly once per calendar day (in the configured timezone), regardless of how many times the main loop iterates after markets close.

### Why `ib.sleep()` Instead of `time.sleep()`?

This was one of our challenges (see Section 21). `ib_insync` needs its event loop running to process callbacks (order fills, connection events). Regular `time.sleep()` blocks the event loop. `ib.sleep()` sleeps while keeping the event loop alive. We originally used APScheduler but had to replace it because of this.

### Graceful Shutdown

When you press Ctrl+C:
1. Sets `state.shutting_down = True` (in `core/state.py`) to prevent reconnection attempts
2. Closes all open day-trade positions (so you don't accidentally hold them overnight)
3. Disconnects from IBKR cleanly

---

## 14. Portfolio Tracking — `core/portfolio.py`

This module manages all persistent state using SQLite — a simple file-based database that doesn't require a separate server.

### The Database Tables

```
positions (currently held stocks):
  - id, ticker, exchange, quantity, entry_price, entry_time
  - stop_loss, take_profit, trade_type, sector

trades (completed, closed trades):
  - id, ticker, exchange, quantity
  - entry_price, exit_price, entry_time, exit_time
  - trade_type, sector, reasoning

daily_summary (end-of-day snapshots):
  - date, portfolio_value, daily_pnl, daily_pnl_pct
  - num_trades, winning_trades, losing_trades

signals (every signal ever generated — audit trail):
  - ticker, action, confidence, prices, reasoning
  - source (screener/ai), timestamp

pending_orders (tracks unfilled order placement times):
  - perm_id (IBKR permanent order ID, primary key)
  - ticker, placed_at
  - confidence (AI confidence 0-100, nullable for legacy/manual orders)
  - Cleaned up on fill or cancellation
  - Legacy DBs missing the confidence column are migrated in place by
    init_db() via ALTER TABLE guarded by PRAGMA table_info.
```

### Key Operations

**`init_db(db_path)`** — Creates all tables and indexes if they don't exist. Uses individual `conn.execute()` calls (not `executescript()`) so connection-level PRAGMAs such as `foreign_keys=ON` are never silently reset by an implicit commit.

**`verify_db(db_path)`** — Called immediately after `init_db()` at startup. Queries `sqlite_master` for every required table and raises `RuntimeError` if any are missing, giving a clear error message rather than a cryptic "no such table" failure later during trading. Protects against silent `init_db` failures (e.g., disk full, permission error swallowed by a bug).

**`add_position(position)`** — When a trade fills, save it to the positions table. Checks for an existing position with the same ticker before inserting to prevent duplicate positions (e.g., from a race condition where two fills arrive back-to-back for the same stock).

**`close_position(ticker, exit_price)`** — When closing a trade:
1. Read the position from the positions table
2. Create a Trade record with entry AND exit info
3. Insert into trades table
4. Delete from positions table
5. Log the trade to the daily CSV journal (`core/logger.log_trade_to_csv`)
6. All in one transaction (either everything succeeds or nothing does)

**`get_daily_pnl(day, db_path, unrealized_pnl)`** — Calculate today's total P&L (realized + unrealized). Realized P&L sums all trades closed on the given day. The optional `unrealized_pnl` parameter (from IBKR account summary) adds mark-to-market losses on open positions, ensuring the daily loss limit catches unrealized drawdowns too.

**`get_trades(start_date, end_date, ticker, db_path)`** — Return completed trades, optionally filtered by date range and/or ticker. Used by the exit handler to notify the correct trade (filtered by ticker) and by the circuit breaker (filtered by date).

**`record_signal(signal)`** — Save every signal for audit trail. Even rejected ones. Useful for backtesting analysis: "How often did we reject signals that would have been profitable?"

**`save_pending_order(perm_id, ticker, db_path=DB_PATH, *, confidence=None)`** — Records when a parent order was placed. Called from `place_order()` after the bracket is submitted, with the AI confidence that approved the signal. Uses `INSERT OR IGNORE` so duplicate calls are safe. `confidence` is keyword-only to preserve the long-standing `(perm_id, ticker, db_path)` positional call sites.

**`get_pending_order_time(perm_id)`** — Returns the original placement datetime for a pending order, or `None` if not found. Used by `get_stale_orders()` to calculate accurate order age across reconnections.

**`get_pending_order_confidence(perm_id)`** — Returns the AI confidence (0-100) recorded at placement time, or `None` for pre-migration rows and non-AI orders. Used by the executor's eviction logic to rank the weakest pending BUY.

**`remove_pending_order(perm_id)`** — Deletes a pending order record. Called from `cancel_bracket_order()`, `cancel_order()`, and `setup_fill_handler()` when orders are cancelled or filled.

**`reconcile_positions(ibkr_positions, auto_fix=False)`** — Compares DB positions against IBKR's live positions. Returns a reconciliation report with orphaned positions (in DB only or IBKR only) and quantity mismatches. When `auto_fix=True`, automatically closes orphaned DB positions by recording them as trades at entry price (actual fill price is unknown). Called at startup with `auto_fix=True` to sync state after restarts where stop-loss/take-profit orders may have filled while the bot was offline.

### Transaction Safety

All database operations use Python's context manager pattern:

```python
with _db_connection(db_path) as conn:
    conn.execute("INSERT INTO positions ...")
    # If an error happens here, everything rolls back automatically
```

This prevents half-written data if the system crashes mid-operation.

---

## 15. Logging & Dashboard — `core/logger.py`

### Log Files

Every day creates a new log file: `logs/trader_2026-04-01.log`

Logs include timestamps, log levels, and which module generated the message:
```
2026-04-01 16:45:32 INFO  [scheduler] Scan cycle started - US market
2026-04-01 16:45:33 INFO  [screener] 12 candidates from 287 stocks
2026-04-01 16:45:58 INFO  [analyst] AAPL: BUY confidence=78
2026-04-01 16:45:59 WARNING [risk] TSLA rejected: daily loss limit
```

### Trade Journal

Every completed trade gets written to a CSV: `logs/trades_2026-04-01.csv`

This creates a paper trail you can open in Excel or Google Sheets to review your trading performance.

### Rich Terminal Dashboard

When the system runs, it displays a styled dashboard in your terminal using the Rich library:

```
┌─ Portfolio Summary ─────────────────────────┐
│ Value: $105,234.50  Daily P&L: +$1,234.50   │
│ Open Positions: 4    Win Rate: 65%           │
└─────────────────────────────────────────────┘

┌─ Open Positions ────────────────────────────┐
│ AAPL  100 shares  +2.3%  $+405.00           │
│ MSFT   50 shares  -0.8%  $-162.00           │
│ NVDA   30 shares  +4.1%  $+891.00           │
└─────────────────────────────────────────────┘

┌─ Scan Results ──────────────────────────────┐
│ GOOGL  BUY   Score: 72  RSI oversold + MACD │
│ AMZN   SELL  Score: 65  Bollinger breach     │
└─────────────────────────────────────────────┘
```

---

## 16. Notifications — `notifications/telegram.py`

Sends alerts to your phone via Telegram. You create a Telegram bot (using @BotFather), get a token, and the system sends messages through it.

### Notification Types

- **System started** — Mode, portfolio value, cash balance. Sent once at startup.
- **Scan summary** — After each 15-min scan cycle: candidates found, AI approved, risk approved, orders placed.
- **Trade opened** — "BUY 100 AAPL @ $175.50 | SL: $170.00 | TP: $185.00" with AI reasoning.
- **Trade closed** — "SOLD 100 AAPL @ $183.20 | P&L: +$770.00 (+4.4%)"
- **Daily summary** — End-of-day report with total P&L, trades, win rate
- **Risk warning** — "Daily loss limit reached. Trading halted."
- **Error** — "IBKR connection lost. Attempting reconnect."
- **System stopped** — Sent when the trader shuts down (Ctrl+C or signal).

### Interactive Status

The system runs a background listener thread that polls for incoming Telegram messages. When you send **"status"** (or "/status") to the bot, it replies with:

- Current phase (fetching data, AI analyzing with progress e.g. "3/90", waiting, etc.)
- Mode (paper/live — derived from IBKR port, not CLI argument)
- Account summary (portfolio value, cash available, invested amount)
- P&L breakdown: unrealized, realized (today), and total daily P&L
- Trade stats for the day (total trades, winners, losers)
- Open positions with live IBKR prices, current market value, and P&L percentage
- Open orders from IBKR (action, quantity, order type, status)
- Last scan summary

This runs on a daemon thread so it doesn't interfere with trading. The polling uses Telegram's long-polling with a 10-second timeout, so responses come within seconds.

The status data is served from an in-memory cache (`_system_status`) updated at scan cycle start and on position changes. When a user sends `/status`, `refresh_positions_cache()` makes a fresh asynchronous call to IBKR via the stored IB connection reference (`_ib_ref`, set at startup by `set_ib_instance()`) to fetch live account values (NetLiquidation, cash, unrealized P&L), then re-reads positions from the DB. This ensures the response always reflects the current broker state, not stale scan-cycle data. If the IBKR connection is unavailable, it falls back to the last cached account values gracefully. The trading mode shown in status is derived from the IBKR port (`is_paper_mode()`) rather than the CLI argument, so it always reflects the actual connection type.

### Design

Notifications are fire-and-forget. If Telegram is down or the token is missing, the error is logged but the system keeps trading. Notifications are nice-to-have, not critical path.

### HTML Sanitization

All messages are sent through `_send_sync()` with `parse_mode="HTML"`. Before each send, `_sanitize_html()` runs `html.escape()` on the body and then restores a small whitelist of tags that the `notify_*` helpers actually emit (`<b>`, `<i>`, `<code>`, `<pre>`, `<u>`, `<s>`). Anything else — notably the `<br>` fragments IBKR embeds in rejection reasons — is escaped rather than interpreted, which prevents the "Can't parse entities" failure that previously dropped order-rejection alerts.

---

## 17. Backtesting Engine — `backtest/engine.py`

The backtester answers: "If this strategy had been running for the last 6 months, how would it have performed?"

### The Key Insight

The backtester uses the **exact same screener and risk manager code** as live trading. It doesn't have a separate "backtest version" of the screener. This means what you backtest IS what you'll run live. No surprises.

The only things that differ:
- **Data source**: YFinance instead of IBKR (no connection needed)
- **Execution**: Simulated fills instead of real orders
- **Slippage**: Adds 0.1% slippage to simulate real-world execution
- **Commission**: Adds $1 per trade

Additional backtest-only features:
- **Indicator weights**: Override per-indicator scoring weights (e.g., weight RSI higher than Bollinger) via `BacktestConfig.indicator_weights`
- **Volatility scaling**: Enable `use_volatility_scaling=True` to scale position sizes inversely to realized market volatility — the same `calculate_realized_volatility()` function used in live trading

### How It Works

```
Day-by-day replay:

For each trading day in the date range:
  1. CHECK EXITS — Do any open positions hit their stop-loss or take-profit?
     Look at today's open, high, and low:
       - If BOTH SL and TP could trigger on the same bar (wide-range day),
         use the open price to resolve which triggered first:
         * Open gaps past SL → SL triggered first (fill at open)
         * Open gaps past TP → TP triggered first (fill at TP)
         * Open between SL and TP → assume SL (conservative)
       - If only SL hit:
         * Gap scenario (open already past stop) → fill at open price
         * Intraday scenario (open above stop, low dips below) → fill at stop price
       - If only TP hit → close at take_profit

  2. BUILD DATA — Gather all price history UP TO today (not including future!)
     This is critical: NO LOOK-AHEAD BIAS.
     On March 15, the screener only sees data through March 15.

  3. SCREEN — Run the same screener with same settings

  4. RISK CHECK — Run the same risk manager with today's open price as
     current_price for anti-momentum checks (prevents bypass where
     entry_price == current_price would always show 0% move)

  5. SIMULATE FILL — "Buy" at today's open price + slippage
     (realistic: signal from yesterday's close, execution at today's open)

  6. RECORD EQUITY — Write down portfolio value at end of day (mark-to-market
     using current closing prices, not entry prices)

After all days:
  Close any remaining open positions at last day's close
  Calculate performance metrics
```

### No Look-Ahead Bias

This is the #1 sin of backtesting: accidentally using future data to make past decisions. Our protection:

```python
# Only use data BEFORE current date (strict <, not <=)
historical = full_data[full_data.index.date < current_date]
candidates = screen_stocks(historical)  # screener only sees past
```

The screener genuinely doesn't know what happens tomorrow.

### SimulatedPortfolio

An in-memory portfolio that tracks:
- Cash remaining
- Open positions (long positions have positive quantity, short positions have negative quantity)
- Closed trades
- Equity curve (portfolio value at end of each day)

Short positions are handled correctly: opening a short credits cash (sale proceeds), closing a short debits cash (buying back), and P&L is calculated via `(exit_price - entry_price) * quantity` where negative quantity inverts the sign as expected.

No SQLite needed — it's all in memory since it's just a simulation.

### Walk-Forward Validation — `walk_forward_backtest()`

A single backtest over one contiguous window tells you how a strategy performed in *that specific past*. It doesn't tell you whether the strategy actually works or whether you just got lucky (or — worse — whether you tuned parameters until the numbers looked good, a form of **overfitting**). Walk-forward validation is the standard way to pressure-test this.

**How it works:** `walk_forward_backtest(config, train_ratio=0.6)` takes a standard `BacktestConfig` with bounded `start_date` and `end_date`, splits the range into an **in-sample (IS)** training period and a non-overlapping **out-of-sample (OOS)** test period (60/40 by default), and runs a fresh backtest on each. Both runs start with the same initial capital — OOS does **not** inherit IS positions or cash. All other config (tickers, slippage, commission, indicator weights, volatility scaling) is preserved for both runs.

**What you get back:** a `WalkForwardResult` dataclass containing both `SimulatedPortfolio` objects, both full metric dicts, the split dates, and a **degradation** dict computed as `OOS_value − IS_value` for every numeric metric. If a robust strategy produces similar metrics on both halves, degradation will be near zero. If the strategy is overfit, degradation will be sharply negative (win rate and total return both drop). In practice this is the single cheapest check to run before trusting a backtest result.

**Guards:** `train_ratio` must be strictly between 0 and 1, both dates must be set, and `end_date` must be after `start_date`. Violating any of those raises `ValueError` rather than silently running a useless backtest.

---

## 18. Backtest Reporting — `backtest/report.py`

Takes the results from the backtester and calculates professional performance metrics.

### Metrics Calculated

| Metric | What It Means |
|--------|---------------|
| **Total Return** | If you started with $100K, how much did you end with? |
| **Annualized Return** | Total return normalized to a yearly rate |
| **Sharpe Ratio** | Return per unit of risk. >1 is good, >2 is great, <0 is losing money |
| **Max Drawdown** | Worst peak-to-trough decline. "At worst, I was down X% from my best" |
| **Win Rate** | What % of trades were profitable |
| **Profit Factor** | Total profits / total losses. >1 means profitable overall |
| **Avg Trade P&L** | Average profit/loss per trade |
| **Best/Worst Trade** | Your biggest win and biggest loss |
| **Avg Duration** | How long trades are held on average |

### Comparison Feature

You can run multiple backtests with different settings and compare them side by side:

```bash
python main.py --mode backtest --tickers AAPL MSFT GOOGL --capital 100000
```

### AI Value-Add Comparison

The `compare_ai_value_add(screener_metrics, ai_metrics)` function answers the key question: "Is the AI analyst actually helping?" It takes metrics from two backtest runs (screener-only vs screener+AI) and computes:

| Alpha Metric | What It Measures |
|-------------|-----------------|
| **Return Alpha** | Return difference: AI minus screener-only |
| **Sharpe Alpha** | Risk-adjusted return difference |
| **P&L Alpha** | Absolute dollar profit difference |
| **AI Filter Rate** | % of screener trades the AI filtered out (rejected as hold/low-confidence) |
| **AI Adds Value** | Boolean flag: True if AI improved returns |

Run two backtests — one with `use_ai=False` (default) and one with `use_ai=True` — then compare. Negative return alpha means the AI is destroying value by filtering out good trades or adding noise.

---

## 19. The Entry Point — `main.py`

This is where everything starts. It parses command-line arguments and routes to the right mode.

### Command Line Usage

```bash
# Paper trading (default — safe, uses fake money)
python main.py

# Single scan cycle (run once and exit)
python main.py --once

# Force scan outside market hours (GTC orders queue for next open)
python main.py --force

# Watchdog mode — IBC manages gateway lifecycle, auto-reconnects after restarts
python main.py --watchdog

# Dry run (full pipeline, but only LOG orders, don't place them)
python main.py --mode dry-run

# Live trading (REAL MONEY — requires confirmation)
python main.py --mode live

# Backtesting
python main.py --mode backtest --backtest-tickers AAPL MSFT GOOGL

# Backtest with date range
python main.py --mode backtest --backtest-tickers AAPL --backtest-start 2025-01-01 --backtest-end 2025-12-31
```

### Watchdog Mode

When you pass `--watchdog`, the system uses IBC (IB Controller) to manage the full IB Gateway lifecycle:

1. IBC starts IB Gateway and logs in automatically (using credentials from `~/ibc/config.ini`)
2. The `Watchdog` class from `ib_insync` monitors the connection
3. When the gateway auto-restarts (daily restart at the configured time), the Watchdog detects the disconnect, waits for the gateway to come back, and reconnects automatically
4. The scheduler continues running on the reconnected IB instance

This is the recommended mode for unattended operation. It can also be run as a systemd service (see `~/.config/systemd/user/auto-trader.service`).

### Live Mode Safety

When you run `--mode live`, the system:
1. Prints a big warning
2. Shows your account details
3. Asks you to type "CONFIRM LIVE" to proceed
4. Connects to port 7496 (live) instead of 7497 (paper)

This prevents accidentally trading with real money.

### Startup Sequence

```
1. Parse arguments
2. Setup logging (file + console)
3. Load environment variables (.env)
4. Initialize SQLite database (create tables if first run)
5. If backtest mode → run backtester → display results → exit
6. If --watchdog → start IBC Watchdog (starts gateway, connects, monitors) → start scheduler on connect
7. Otherwise → connect to IBKR directly (gateway must already be running)
8. Display account summary
9. Reconcile positions (auto-fix: close orphaned DB positions not held at IBKR)
10. Reattach exit handlers for existing bracket orders (survives restarts)
11. Start Telegram listener + update portfolio cache
12. If --once → run single scan cycle → exit
13. Otherwise → start scheduler loop (runs until Ctrl+C)
```

### Position Sync on Startup

The bot's internal state (SQLite) can drift from IBKR's actual state when the bot is offline while stop-loss or take-profit orders fill. Three mechanisms keep them in sync:

1. **Auto-fix reconciliation** (`reconcile_positions(auto_fix=True, ibkr_fills=...)`): Compares DB positions against IBKR's live positions. Any DB position not found at IBKR is automatically closed. When `ibkr_fills` is provided (extracted from `ib.fills()` at connection sync), the most recent SLD fill after the position's entry_time is used as the exit price — giving accurate P&L for positions that filled while the bot was offline (including take-profit exits that would otherwise be mis-recorded as stop-outs). When no matching fill exists, falls back to the position's stop-loss price. This ensures daily P&L, loss limits, and circuit breaker checks reflect reality.

2. **Import orphaned IBKR positions** (`import_ibkr_positions(ib, orphaned_tickers)`): Positions held at IBKR but missing from the DB are automatically imported. Entry price comes from IBKR's `avgCost`, and stop-loss/take-profit are extracted from open bracket child orders. This handles cases where the DB was cleared, positions were opened manually, or the bot missed a fill event.

3. **Exit handler reattachment** (`reattach_exit_handlers(ib)`): After reconciliation, scans `ib.openTrades()` for child orders (stop-loss/take-profit) belonging to open DB positions and attaches fresh exit handlers. This ensures fills that occur after restart are caught in real-time instead of waiting for the next scan cycle.

4. **Immediate cache refresh** (`refresh_positions_cache()`): When an exit handler fires (stop-loss or take-profit fills), the Telegram status cache is updated immediately from the DB — not just at the next scan cycle. Additionally, every `/status` command from Telegram triggers a fresh DB read before building the response, so the user always sees the latest positions and P&L. All writes and reads to the status cache are protected by a threading lock to prevent torn reads across the scheduler, executor, and Telegram listener threads.

---

## 20. Testing Strategy

### Test Philosophy

Every module has its own test file. Tests use **synthetic data** — hand-built DataFrames and mock objects — so they don't need an IBKR connection or internet access. Shared test fixtures (`make_signal()`, `make_position()`) live in `tests/conftest.py` to avoid duplication across test files.

### What's Tested

| Test File | What It Verifies |
|-----------|-----------------|
| `conftest.py` | Shared fixtures: `make_signal()`, `make_position()` factories |
| `test_screener.py` | Each indicator triggers correctly on crafted price data |
| `test_risk.py` | Each safety check accepts/rejects correctly, circuit breaker streak logic |
| `test_analyst.py` | Prompt building, response parsing, validation (including `trade_type` field) |
| `test_data.py` | Data fetching, caching, column normalization |
| `test_connection.py` | Contract creation, connection handling |
| `test_portfolio.py` | DB operations, position lifecycle |
| `test_models.py` | Dataclass construction, computed properties (P&L, duration) |
| `test_universe.py` | Universe building, filtering, caching |
| `test_scheduler.py` | Streaming signal pipeline, callback-based risk check + execution |
| `test_executor.py` | Fill direction, exit handler reattachment on startup, IBKR position import |
| `test_stale_orders.py` | Persistent order timestamps, stale detection, DB cleanup on fill/cancel |
| `test_telegram.py` | Status commands, portfolio display, risk notifications, cache refresh |
| `test_backtest.py` | Full backtest loop, exit checking, no look-ahead |

### Running Tests

```bash
# Run all tests
pytest tests/

# Run one file with verbose output
pytest tests/test_screener.py -v

# Run one specific test
pytest tests/test_risk.py::test_daily_loss_limit -v
```

---

## 21. Challenges We Faced & How We Solved Them

### Challenge 1: Cloud AI Was Too Expensive

**The problem**: The original design used cloud AI APIs (Claude, GPT) for stock analysis. Running 10-20 analyses every 15 minutes, all day long, would cost real money. And what if the API goes down during market hours?

**The solution**: Switched to **Ollama** running **Qwen 2.5 7B** locally. Zero cost, zero internet dependency. The model runs on your own GPU/CPU. We had to rewrite the analyst module from cloud API calls to local HTTP calls, but the prompt structure stayed the same.

**Tradeoff**: A 7B model isn't as smart as GPT-4 or Claude. But for stock analysis with structured data, it's good enough — and it's free.

### Challenge 2: APScheduler vs ib_insync Event Loop

**The problem**: We initially used APScheduler to run the scan cycle every 15 minutes. But `ib_insync` (the IBKR library) has its own asyncio event loop, and it requires all IBKR calls to happen on the main thread's event loop. APScheduler was running our scan function in a way that conflicted with this.

**The symptoms**: Random connection drops, callbacks not firing, "not connected" errors in the middle of operations.

**The solution**: Ripped out APScheduler entirely. Replaced it with a simple `while` loop using `ib.sleep()` — which sleeps while keeping ib_insync's event loop alive. Much simpler, much more reliable.

**Lesson**: Sometimes a simple `while True` loop is better than a fancy scheduler library.

### Challenge 3: AI Rejecting Valid "Hold" Responses

**The problem**: When the AI analyst decided a stock wasn't worth trading (HOLD), it would return `action: "hold"` with no prices (no entry, stop-loss, or take-profit — because there's no trade to make). But our validation code required prices for ALL responses and was rejecting these as "invalid."

**The symptoms**: Every HOLD response was logged as a validation failure. The AI was working correctly, but we were throwing away its valid "no trade" decisions.

**The solution**: Updated validation to skip price checks when action is HOLD. Null prices are perfectly fine for holds — there IS no trade. Also promoted these validation failures from DEBUG to WARNING level so we'd actually notice them in the logs.

### Challenge 4: Multi-Market Complexity (BIST)

**The problem**: The original design supported both US and Turkish (BIST) markets. But BIST had different trading hours, different currency (TRY), different data sources, different contract types, and different sector classifications. Every module needed `if market == "BIST"` branches.

**The symptoms**: Complexity in 10+ files. Edge cases everywhere. BIST data was less reliable. The effort to maintain two markets was slowing down development of core features.

**The solution**: Made the strategic decision to **remove BIST entirely** and focus exclusively on US stocks. Deleted 54 lines of BIST-specific code across 10 files. The codebase became dramatically simpler.

**Lesson**: It's better to do one market really well than two markets poorly.

### Challenge 5: Small Stock Universe

**The problem**: Initially, the universe builder ran just 1-2 IBKR scanner types, which returned maybe 50 stocks. That's a tiny pool to find trading opportunities in.

**The solution**: Expanded to **10 different scanner types** (most active, top gainers, top losers, hot by volume, gap ups/downs, 13-week highs/lows, top trade count/rate). Each returns different stocks, so after deduplication, we get 200-350 unique stocks. Much better coverage.

**Additionally**: Added a static fallback list of ~100 well-known stocks. Even if all IBKR scanners fail, the system has stocks to screen.

### Challenge 6: Candidate Cap Was Limiting Opportunities

**The problem**: The screener originally capped candidates at 20 (the top 20 scores). This meant stocks scoring #21 never got analyzed by the AI, even if they were great opportunities.

**The solution**: Removed the cap entirely. ALL stocks above the minimum score get passed to the AI analyst. The AI is smart enough to reject bad ones (that's its job). Why have a dumb numeric cutoff do the AI's job?

**Tradeoff**: More AI analyses per cycle = slightly longer scan time. But with a local model, there's no cost penalty, just time (~2-3 seconds per analysis).

### Challenge 7: IBKR Connection Instability

**The problem**: IBKR's TWS/Gateway connections drop randomly. Network hiccup? Connection lost. TWS auto-restarts? Connection lost. IBKR does daily server resets around midnight? Connection lost.

**The solution**: The `ensure_connected()` function at the top of every scan cycle. Before doing anything, check the connection. If it's dead, reconnect. Plus: the graceful shutdown handler sets a `shutting_down` flag (in the shared `core/state.py` module) to prevent the reconnection logic from fighting a deliberate disconnect (this was a bug where Ctrl+C would try to reconnect instead of shutting down). The flag lives in a shared module to avoid a circular import between `scheduler.py` and `executor.py`.

### Challenge 8: Financial Sector Stocks Slipping Through

**The problem**: IBKR's sector data isn't always reliable. Sometimes a bank stock would have sector = "N/A" or a wrong sector, passing through the universe filter.

**The solution**: **Double filtering**. The universe builder filters by sector at build time. Then the risk manager has its OWN `check_excluded_sector()` that catches any stragglers. Belt AND suspenders. Both use the same `FINANCIAL_KEYWORDS` and `DEFENSE_KEYWORDS` lists from `config/settings.py` to stay in sync. Also added an explicit exclusion list of ~40 specific tickers known to be problematic.

**Follow-up (code audit round 3)**: `_is_excluded_sector("")` previously returned `False`, so a cached stock with a blank sector string (e.g. a cache entry written before enrichment fully populated the field) would bypass the filter entirely. Flipped the default to **fail-closed** — unknown sector = excluded. Safety defaults that fail to "reject" are safer than defaults that fail to "accept" for this kind of filter.

### Challenge 9: DST and the Istanbul/ET Mismatch

**The problem**: Market hours were stored as Istanbul wall-clock times (`16:30–23:00`). Istanbul (TRT) is a fixed UTC+3; NYSE is ET which observes DST. In EDT (summer), 09:30 ET = 16:30 TRT, so the config happened to line up. In EST (winter), 09:30 ET = 17:30 TRT — the bot would scan for an hour before the market actually opened and stop scanning an hour before the market actually closed.

**The solution**: Store market hours in the exchange's native timezone (`America/New_York`) and compute `datetime.now(market_tz)` inside `is_market_open`. ZoneInfo handles the DST transitions so the bot tracks the actual session boundaries.

**Related**: `check_pdt_restriction` used `datetime.now(timezone.utc).date()` for its "is this a day trade" comparison. IBKR's PDT rule tracks the US Eastern calendar day, not UTC. A trade opened at 23:00 ET Monday (03:00 UTC Tuesday) and exited at 10:00 ET Tuesday would be seen as same-UTC-day (day trade) when ET correctly classifies it as a swing trade. Fixed by routing both entry and exit through `astimezone(America/New_York).date()`.

### Challenge 10: The Fast-Fill Race

**The problem**: `place_order` polls IBKR for `permId` for up to 3 seconds after submitting the bracket. During that window, a market order on a liquid stock can already fill. Meanwhile, the fill handler is registered by the scheduler *after* `place_order` returns. ib_insync does not replay past events, so any fill that landed during the polling window was silently dropped — the position existed at IBKR but never got written to the DB, visible only at next reconciliation.

**The solution**: `setup_fill_handler` now accepts the parent `Trade` object. After registering the event listener, it inspects `parent_trade.orderStatus.status`; if already `"Filled"`, it invokes the fill path synchronously. A `threading.Event` guards against double-firing if a late event also arrives.

### Challenge 11: The Cancel/Close Race

**The problem**: `close_all_day_trades` issued `ib.cancelOrder` for bracket children, then immediately placed market close orders. IBKR's cancel is asynchronous — the request returns instantly but the order can still fill before the broker processes the cancel. A filling SL child + our market order = doubled close (net flat → net short).

**The solution**: After issuing cancels, poll `ib.openTrades()` in a short loop until the cancelled orders clear (or a 3-second timeout expires). Only then place the market closes.

### Challenge 12: Reconciliation Biased Losses

**The problem**: When the bot restarted and a DB position no longer existed at IBKR, auto-reconcile fell back to `stop_loss` as the exit price. This unconditionally recorded a loss regardless of whether the position was actually stopped-out or took-profit while the bot was offline — systematically biasing the trade journal and circuit-breaker history.

**The solution**: Use the **midpoint** of SL and TP as the unbiased expected-value estimator when no fill data is available, and flag the reasoning field as "estimate" so downstream consumers (daily P&L, circuit breaker) know the exit price is inferred. Over many reconciliations the biases cancel; in any single trade the error is half of what it was.

### Challenge 13: Backtest Gap Entries

**The problem**: The screener's `signal.stop_loss` and `signal.take_profit` are absolute price levels computed from the signal's `entry_price` (yesterday's close). The backtest fills at the next bar's open (realistic execution). If today's open gaps, the stored SL/TP no longer match the intended risk/reward percentages — in the worst case a long's stop_loss sits above the fill price, triggering an immediate non-sensical exit.

**The solution**: In `open_position`, preserve the original SL/TP percentages relative to `signal.entry_price` and recompute new absolute levels from the actual slippage-adjusted fill price. The R-multiple the risk check reasoned about is now what the position actually risks.

### Challenge 14: TOCTOU on Position Insertion

**The problem**: `add_position` did a SELECT for existing rows, then INSERTed if missing. Across two SQLite connections (fill handler thread + scheduler thread), both SELECTs could return empty before either INSERT committed, resulting in duplicate rows for the same ticker.

**The solution**: Added a `UNIQUE` index on `positions(ticker)` and switched the INSERT to `INSERT OR IGNORE`. The DB engine now enforces uniqueness atomically. The loser of the race sees `cursor.rowcount == 0`, looks up the existing row, and returns its ID — no duplicates possible no matter how the threads interleave.

### Challenge 15: yfinance MultiIndex Ambiguity

**The problem**: `yf.download` returns columns as a `MultiIndex`. Across yfinance versions, the level order varies — some return `(field, ticker)`, others return `(ticker, field)`. Flattening with `get_level_values(0)` unconditionally worked only on one ordering; on the other it collapsed to repeated ticker names and the subsequent column selection raised `KeyError`.

**The solution**: Detect which level contains the string `"Close"` and flatten using that level. Version-independent.

### Challenge 16: Parabolic Breakouts Reaching the AI Analyst

**The problem**: On stocks that had just ripped vertically (e.g. XNDU from ~$9 to $32 in a handful of sessions), every BUY-side screener indicator fired at once — RSI overbought flipped into a "momentum confirmation" reading, volume spike, Bollinger breakout, MA crossover. The AI analyst, seeing a wall of confluent signals and positive news headlines, could override an otherwise-SELL ranked candidate into a BUY and hand the order to the executor. The stop-loss inevitably fired on the first pullback.

**The solution**: Added an **extension guard** at the screener level. A ticker whose latest close sits more than `MAX_EXTENSION_OVER_MA20_PCT` (default **15%**) above its 20-day SMA is dropped from the candidate pool *before* scoring and *before* the AI ever sees it. Filtering at the source is the only robust fix — any later layer (AI, risk manager) can be overridden by the same confluent-signal pattern that produced the problem in the first place. The original 6-month sweep landed on 20%, but the bot kept buying near the local peak in the 16–20% band; the 2026-04-28 sweep (`data/sweep_extension_pct_2026-04-28.csv`) lifted win rate to 66.7% (vs 50% at 20%) and total return to +8.32% (vs ~breakeven) by tightening to 15%. Trades that fall in the 16–20% band are systematically losers (~+$8 avg per trade at 20% vs +$1500/trade at 15%), so 15% is the empirical sweet spot — defensive enough to cut the loss-clustered tail, but loose enough to keep ~75% of the trade frequency.

### Challenge 17: Legacy Day-Trade Rolling Count Silently Truncated to Today Only

**The problem**: Under the legacy PDT regime (LEGACY_PDT_THRESHOLD_USD, accounts below $25K), the day-trade count was gated on a rolling 5-business-day window. But the scheduler was calling `get_trades(start_date=datetime.now(utc).date())`, which only returns trades with `exit_time` on today's UTC date. The 7-calendar-day filter inside the legacy check then iterated a list that was already narrowed to today, making the protection effectively a single-day counter.

**The solution**: Scheduler now queries `get_trades(start_date=(now - timedelta(days=7)).date())` so the full rolling window is visible to the filter. Day trades from earlier in the week are no longer invisible to the count. This fix applies to the legacy day-trade counter used under `MARGIN_REGIME = "legacy_pdt"` or `"both"`.

### Challenge 18: Share-Class Tickers Silently Dropped from yfinance

**The problem**: IBKR's symbol format for dual-class stocks uses a space (`BRK B`, `BF B`, `QRED U`). yfinance's canonical form is hyphenated (`BRK-B`). `yf.download("BRK B")`, `yf.Ticker("BRK B").news`, and `yf.Ticker("BRK B").recommendations_summary` all returned empty/None silently. Backtests quietly dropped these tickers from `all_data`; the analyst-consensus filter passed through as if no rating data existed (disabling the filter for the affected ticker); news fetches came back blank.

**The solution**: Added a `_to_yfinance_ticker(ticker)` translation helper in `core/data.py` that replaces spaces with hyphens. Applied at all three yfinance entry points. The IBKR-side symbol remains unchanged — translation happens only at the yfinance boundary.

### Challenge 19: Backtest Volatility Proxy Used One Arbitrary Ticker for All Signals

**The problem**: When `use_volatility_scaling=True`, the backtest engine computed `market_volatility` once per bar by iterating `all_data.items()` and taking the first ticker's realized volatility. Dict iteration order follows insertion order, which follows `config.tickers` order — usually alphabetical. A high-vol first ticker (e.g. AMC at ~120% annualized) produced `vol_scale = 0.20 / 1.20 = 0.167`, and every signal in every bar was sized at 17% of base regardless of the candidate's own vol regime. The strategy looked dramatically weaker than a correctly-sized run.

**The solution**: Compute volatility from the signal's own historical close series (`stock_data[signal.ticker]`, which already uses the look-ahead-safe `date < current_date` slice). Each signal now receives its own vol scaling factor. The screener and risk manager already support per-candidate volatility — the backtest was the only place passing a single global proxy.

---

### Challenge 20: Daily-Loss Limit Denominator Drifted with MTM

**The problem**: `check_daily_loss_limit` computed its dollar cap as `portfolio_value * limit_pct`. That denominator was the current MTM equity — which shrinks as losses accumulate within the session, making the cap itself shrink. After a 5% drawdown the 2% limit effectively enforces ~1.9%. Worse, the cap becomes path-dependent — trading during a choppy day hits a different threshold than trading during a straight-down day with identical terminal P&L.

**The solution**: Snapshot start-of-day NetLiquidation on each ET-date rollover in `core/state.py`, pass `start_of_day_equity` to `check_daily_loss_limit`, and compute the cap off that stable baseline. In the backtest, yesterday's end-of-day equity from `equity_curve[-1][1]` serves the same purpose.

We also rewired how `daily_pnl` itself is computed. The old code summed today's realized P&L plus IBKR's `UnrealizedPnL`, but `UnrealizedPnL` is cumulative since entry — a swing position opened a week ago would inflate today's reported P&L by the whole week's gain. With a start-of-day snapshot in hand, today's P&L is just `current NetLiquidation − start_of_day_equity`, which is self-consistent with the cap denominator.

---

### Challenge 21: Stale Open-Positions View Between Rapid AI Approvals

**The problem**: The streaming AI pipeline calls `_on_signal` for each approved candidate. The callback placed the order, slept 0.5 s, and refreshed `open_positions` from the DB. But the bracket entry is a limit order — it almost never fills within 500 ms, so the DB still showed the PRIOR set of positions when the NEXT AI approval arrived. Two rapid signals therefore evaluated `max_positions`, `check_sector_concentration`, and `check_cumulative_risk` against the same stale snapshot, and both could pass a gate that should have tripped on the second.

**The solution**: `run_scan_cycle` now maintains a `pending_this_cycle: list[Position]` of virtual positions for brackets placed this cycle but not yet filled. Each subsequent `_on_signal` invocation merges DB positions with these virtuals (de-duplicated by ticker) and passes the merged list to `evaluate`. `entry_time` on the virtuals is UTC, matching every other code path.

---

### Challenge 22: Concurrent `close_position` Duplicated Trade Rows

**The problem**: `close_position` does SELECT → INSERT INTO trades → DELETE from positions. With default SQLite isolation, two threads (the exit-fill event handler and `close_all_day_trades` on the scheduler) could both SELECT the same row, both INSERT a Trade, then both attempt DELETE (second is a no-op). Result: one position, two Trade rows, two "position closed" Telegram notifications.

**The solution**: `_db_connection` gained an `immediate=True` mode that issues `BEGIN IMMEDIATE` at transaction start, acquiring the SQLite write lock upfront. `close_position` uses it, so the critical section is serialized. `busy_timeout=30000` lets the second writer wait for the first to commit rather than failing with "database is locked." The UNIQUE index on `positions.ticker` already handled the dual-INSERT case for opens; this closes the matching gap on closes.

---

### Challenge 23: Exit Handler Matching Children By Ticker Alone

**The problem**: `setup_exit_handler` fell back to ticker-only matching when `parent_order` was missing (the reattach path could end up calling it this way). If a day-trade closed INTC in the morning and the bot re-entered INTC later that same session, the old exit handler was never deregistered from `ib.orderStatusEvent`. When the new bracket's stop-loss fired, the old handler matched on ticker and fired `db_close_position` on the fresh position — corrupting the P&L record and emitting a second spurious Telegram notification.

**The solution**: Refuse to register an exit handler without a non-zero `parent_order.orderId`. The handler now requires precise `parentId` matching, so cross-bracket fires are structurally impossible. Reattach logs a clear warning when it can't resolve the parent rather than silently attaching a brittle handler.

---

### Challenge 24: Market-Close Boundary Crossed Mid-Scan

**The problem**: `run_scan_cycle` checked `minutes_to_close` once per market when the scan started. AI analysis can take several minutes per candidate. A scan that started with 20 minutes to close could reach `place_order` for the last signal at 10 minutes to close — inside the `CLOSE_MINUTES_BEFORE=15` window. Then the `finally` block's `close_all_day_trades` immediately flattens the fresh position at uncertain market-on-close pricing, churning a round-trip for nothing.

**The solution**: `_on_signal` re-checks `minutes_to_close(market)` right before `place_order`. If wall-clock time crossed the threshold, the callback logs and skips. The initial `run_scan_cycle` check remains as the coarse gate for starting the scan; this second check is the fine gate for placing each individual order.

---

### Challenge 25: Liquidity Filter Was a No-Op on the IBKR Scanner Path

**The problem**: `universe._filter_universe` only applied `MIN_DAILY_VOLUME` and `MIN_MARKET_CAP` when the corresponding field was positive — and the IBKR scanner path stored `avg_volume=0.0, market_cap=0.0` because the scanner doesn't return them. So every ticker the scanner surfaced bypassed the liquidity gate entirely. Illiquid names (low-volume SPAC units, niche ETFs) could reach the AI analyst and waste a ~3-minute LLM call per candidate.

**The solution**: Added a screener-level liquidity gate. `screen_stocks` computes the trailing 20-day mean of `volume` from the OHLCV DataFrame and skips the ticker when it's below `MIN_DAILY_VOLUME`. The DataFrame is the authoritative source — we already have it, and it reflects actual trading activity rather than scanner metadata.

---

### Challenge 26: Flat Per-Trade Backtest Commission Hid Friction

**The problem**: `BACKTEST_COMMISSION = 1.0` charged a flat $1 regardless of share count. For a 1000-share position IBKR actually charges ~$5 per leg under tiered pricing. Backtests of sized-up strategies (50 % of a $100 k account) undercounted round-trip friction by 5–10 %, inflating reported Sharpe and profit factor.

**The solution**: Added `BACKTEST_COMMISSION_PER_SHARE = 0.005` to settings. `SimulatedPortfolio` now computes `max(min_commission, per_share * qty)` per leg — matching IBKR's tiered structure. Tests that set `commission=0` to isolate other mechanics also zero `commission_per_share` in `__post_init__` so they remain friction-free.

---

### Challenge 27: Shutdown and Tavily Flags Were Bare Booleans

**The problem**: `state.shutting_down` and `data._tavily_exhausted` were plain module-level bools. CPython's GIL makes assignment atomic on a single attribute, but there is no documented visibility guarantee across threads — and CPython is not the only possible runtime. More practically, the reads and writes happened across the ib_insync asyncio thread, the signal-handler thread, the scheduler main thread, and the Telegram listener thread. Bare-bool patterns like `if _tavily_exhausted: ...` imply a memory model we don't actually have.

**The solution**: Both flags are now `threading.Event` instances. `.set()` / `.clear()` / `.is_set()` have explicit cross-thread atomicity and visibility semantics. Callers were updated in `core/scheduler.py`, `core/executor.py`, and both places where the Tavily flag is reset in the test suite.

---

## 22. Architecture Decisions & Why

### "Pure Functions" for Screener and Risk Manager

**Decision**: The screener and risk manager never fetch data. They receive data as function arguments.

**Why**: The backtester needs to run the same logic on historical data. If the screener fetched live data internally, the backtester would need a separate "backtest screener" with duplicate logic. Pure functions mean ONE screener for both live and backtest. Change a rule, and it changes everywhere.

### Local AI Instead of Cloud

**Decision**: Use Ollama + Qwen 2.5 7B running locally instead of Claude/GPT APIs.

**Why**: Cost and reliability. 10-20 analyses every 15 minutes × 6.5 hours per trading day × 252 trading days = ~100,000+ analyses per year. Even at $0.01 each, that's $1,000/year. Local is $0. Plus no outage risk during market hours.

### SQLite Instead of PostgreSQL

**Decision**: SQLite file-based database instead of a proper database server.

**Why**: This is a single-user system running on one machine. There's no concurrent access, no need for network database, no complex queries. SQLite is zero-config, zero-maintenance, and the database is just a file you can back up by copying it.

### Bracket Orders

**Decision**: Every trade is placed as a bracket order (entry + take-profit + stop-loss).

**Why**: Atomicity. With separate orders, there's a window where you have a position but no stop-loss (if the system crashes between placing the entry and the stop-loss). With bracket orders, IBKR handles all three as a unit. Your stop-loss exists from the very first moment. The parent and take-profit orders use `transmit=False` and only the stop-loss (last leg) uses `transmit=True`, ensuring all three transmit to IBKR as a single atomic unit.

### Paper Mode as Default

**Decision**: `--mode paper` is the default. Live requires explicit `--mode live` AND typing "CONFIRM LIVE".

**Why**: The worst bug in a trading system is accidentally trading with real money. Two barriers (flag + confirmation) make this extremely unlikely.

### Turkey Timezone

**Decision**: All times are in `Europe/Istanbul` timezone.

**Why**: The developer is in Turkey. US market hours are 16:30-23:00 TRT. Rather than constantly converting, everything uses local time.

---

## 23. The Complete Data Flow

Here's the entire system in one diagram, showing exactly what data flows where:

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py                                 │
│                    (parse args, pick mode)                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
            ┌─────────────┼────────────────┐
            │             │                │
         BACKTEST      PAPER/LIVE       DRY-RUN
            │             │                │
            ▼             ▼                ▼
     ┌──────────┐  ┌──────────────────────────────────┐
     │ YFinance │  │     scheduler.py (every 15 min)   │
     │  (data)  │  │                                    │
     └────┬─────┘  │  ┌─────────────────────────────┐  │
          │        │  │ 1. ensure_connected()        │  │
          │        │  │ 2. get_account_summary()     │  │
          │        │  │ 3. build_universe()  ────────┼──┼──→ IBKR Scanners
          │        │  │ 4. get_historical_data() ────┼──┼──→ IBKR Data
          │        │  │ 5. screen_stocks()           │  │
          │        │  │    ├─ check_rsi()            │  │
          │        │  │    ├─ check_macd()           │  │
          │        │  │    ├─ check_ma_crossover()   │  │
          │        │  │    ├─ check_volume_spike()   │  │
          │        │  │    ├─ check_bollinger()      │  │
          │        │  │    └─ check_support_resist() │  │
          │        │  │ 6. analyze_batch()  ─────────┼──┼──→ Ollama (local AI)
          │        │  │    └─ per candidate:         │  │
          │        │  │       ├─ build prompt        │  │
          │        │  │       ├─ call LLM            │  │
          │        │  │       └─ validate response   │  │
          │        │  │ 7. evaluate() (risk mgr)     │  │
          │        │  │    ├─ position size           │  │
          │        │  │    ├─ daily loss limit        │  │
          │        │  │    ├─ max positions           │  │
          │        │  │    ├─ stop-loss valid         │  │
          │        │  │    ├─ sector concentration    │  │
          │        │  │    ├─ no duplicate            │  │
          │        │  │    ├─ excluded sector         │  │
          │        │  │    ├─ anti-momentum           │  │
          │        │  │    ├─ trend confirmation      │  │
          │        │  │    └─ risk/reward ratio       │  │
          │        │  │ 8. place_order() ────────────┼──┼──→ IBKR (bracket order)
          │        │  │ 9. add_position() ───────────┼──┼──→ SQLite
          │        │  │ 10. notify_trade() ──────────┼──┼──→ Telegram
          │        │  │ 11. log + dashboard ─────────┼──┼──→ Terminal + log file
          │        │  └─────────────────────────────┘  │
          │        └───────────────────────────────────┘
          │
          ▼
   ┌──────────────┐
   │  backtest/    │
   │  engine.py    │
   │              │
   │  Same code:  │        ┌───────────────┐
   │  screen_stocks() ────→│ backtest/      │
   │  evaluate()      ────→│ report.py      │
   │  (simulated fills)    │                │
   │                       │ Sharpe ratio   │
   │  Day-by-day replay    │ Max drawdown   │
   │  No look-ahead!       │ Win rate       │
   └──────────────┘        │ Profit factor  │
                           └───────────────┘
```

---

## Final Notes

This system was built iteratively over 8 milestones, with each one building on the previous. The code evolved through real-world challenges — switching from cloud to local AI, removing multi-market complexity, fixing event loop conflicts, and tightening discipline rules.

The core philosophy throughout has been:
- **Safety first** — Paper mode default, daily loss limits, bracket orders, multiple confirmation layers
- **Same code everywhere** — Pure functions mean the backtester uses identical logic to live trading
- **Simplicity over cleverness** — SQLite over Postgres, while loop over scheduler framework, local AI over cloud
- **Graceful degradation** — Always have a fallback (cache, static list, retry)

The system is currently fully operational in paper trading mode (Milestones 1-7 complete). Options support (Milestone 8) is planned for the future.

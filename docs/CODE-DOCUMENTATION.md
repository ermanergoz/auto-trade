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

2. **The AI Analyst** — A local AI model (Qwen 2.5 7B running on Ollama) that takes those 15 candidates and does deep analysis. It looks at price trends, momentum, volume, news headlines, macro/political context, and a strict 7-point checklist. It's the senior analyst who says "of those 15, I'd actually buy these 3."

3. **The Risk Manager** — A paranoid rule-checker that gates every trade with 12 different safety checks. Position too big? Rejected. Already lost too much today? Rejected. Chasing a stock that already moved 5%? Rejected. Three losses in a row? Circuit breaker pauses everything. It's the compliance officer who makes sure we never blow up.

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

And 15 minutes before market close, the system auto-closes all day trades (swing trades stay open overnight).

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
  - duration → computed: how long we held it
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
AI_MODEL = "qwen2.5:7b"           # The local AI model
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
```

These values are tuned for a small account ($500). For larger accounts, tighten them back — see [RISK-TUNING.md](RISK-TUNING.md) for a full comparison table and scaling guide.

The risk manager also includes a **cumulative risk check** that ensures total open risk (all positions' max loss via stop-loss) stays within the daily loss limit. This prevents a scenario where 3 positions each sized at 5% risk = 15% total risk, exceeding the 10% daily limit.

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

**`get_historical_data_yfinance(ticker, period, interval)`** — Same data but from Yahoo Finance. Used only by the backtester (since you don't need an IBKR connection to backtest) and as a fallback if IBKR data fails.

### News

**`get_news(ticker, market, max_results)`** — Fetches recent news headlines for AI context. Tries yfinance first (free, no rate limits); if yfinance returns nothing, falls back to Tavily API. This conserves Tavily quota for macro/political headlines.

**`get_macro_news(max_results)`** — Fetches broad market political/macro headlines via Tavily (no yfinance equivalent). Called once per scan cycle and shared across all candidates.

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

Step 3: Combine all results and remove duplicates.
        A stock that appears in 3 different scanners still only appears once.
        Typical result: ~200-350 unique stocks

Step 3.5: Enrich with contract details.
          The IBKR scanner doesn't return sector or company name data.
          So we call reqContractDetails for each stock to fill in those fields.
          A 0.05-second delay is inserted between each request to avoid
          IBKR pacing violations.
          This is what makes the financial sector filter actually work.

Step 3.6: 3-tier sector fallback for stocks still missing sector data:
          1. yfinance — looks up sector and country via yf.Ticker().info
             For ETFs, checks the "category" field to classify as:
               - "Equity ETF" (SPY, QQQ, IWM — kept in universe)
               - "Bond ETF" (HYG, SGOV — excluded)
               - "Leveraged ETF" (TQQQ, SQQQ, SOXL — excluded)
               - "Non-Stock ETF" (BITO, USO, UVIX — excluded)
          2. Ollama LLM — asks the local AI model to classify the stock
             by sector and country (fast, ~128 tokens per query)
          3. Exclude — if all three sources fail, the stock is excluded
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
3. **Ollama LLM** — asks the local AI model to classify by sector and country
4. **Exclude** — unclassifiable stocks are dropped (can't safely filter financials)

**Stock news priority** — yfinance-first to conserve Tavily quota:
1. **yfinance** — `yf.Ticker().news` returns recent headlines for free (no rate limits)
2. **Tavily API** — fallback when yfinance returns nothing

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

### Why Local AI?

We use **Ollama** running **Qwen 2.5 7B** locally. This was a deliberate choice (and a pivot from the original design — see Challenges section). Benefits:

- **Zero cost** — No API fees. Cloud AI (Claude, GPT) would cost money per analysis, and we run 10-20 analyses every 15 minutes.
- **No internet dependency** — Works offline. No API outages.
- **Privacy** — Your trading data never leaves your machine.
- **No rate limits** — Run as many analyses as you want.

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

Makes an HTTP request to the local Ollama server. The AI processes the prompt and returns structured JSON:

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

The Ollama JSON response is parsed inside a try/except that catches both `KeyError` (missing fields) and `JSONDecodeError` (malformed output), logging specific error messages for each case.

Invalid responses are rejected and retried (up to 3 times with exponential backoff).

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

### The 12 Safety Checks

Every signal must pass ALL of these. Fail one, the trade is rejected.

#### 1. Short Selling Block — `check_short_selling()`
"Is this a sell signal for a stock we don't own?"

Rule: If `ALLOW_SHORT_SELLING` is False (the default), reject any SELL signal where we don't already hold the stock.

Why: Short selling (selling borrowed shares hoping the price drops) carries unlimited downside risk — a stock can rise infinitely. With a 7B local AI model, the risk of a bad short call is too high. This check is configurable via `ALLOW_SHORT_SELLING` in settings for advanced users who understand the risks.

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
  position_size: 45  # shares to buy
```

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

**`close_all_day_trades(ib, positions, dry_run)`** — Called 15 minutes before market close. Finds all positions marked as DAY trades and closes them with market orders. After placing all close orders, uses `monitor_orders()` with a 30-second timeout to wait for fills, and logs at ERROR level for any that haven't filled yet — critical because unfilled orders mean positions stay open overnight.

**`handle_fill(signal, quantity, fill_price)`** — Records a filled order in the SQLite database. Returns `None` if the fill quantity is zero or negative, preventing phantom positions from being recorded.

**`setup_fill_handler(ib, signal, quantity, on_fill)`** — Attaches an async callback to handle entry order fills. Only processes parent entry orders (`parentId == 0`) matching the signal's ticker — child orders (take-profit, stop-loss) are handled by `setup_exit_handler` instead. Uses a `fired` flag to ensure each handler instance fires at most once, preventing duplicate position recording when multiple signals for the same ticker are active. When the parent order status changes to "Filled", it records the position in the database, calls the optional `on_fill` callback, and deregisters itself from `ib.orderStatusEvent` to prevent handler accumulation across scan cycles. The post-fill logic (handle_fill, remove_pending_order, callback, unsubscribe) is wrapped in try/finally to ensure the event handler is always cleaned up even if a callback raises an exception. Also logs warnings for partial fills (when filled_qty < requested quantity). The scheduler attaches this handler BEFORE placing the order to avoid a race condition where fast fills fire before the handler is registered.

**`setup_exit_handler(ib, signal, on_exit)`** — Attaches a callback to handle exit order fills (take-profit and stop-loss). When a child order fills, it closes the position in the database via `portfolio.close_position()`, which also writes to the CSV trade journal. Logs a warning if the exit fill cannot be matched to a database position (e.g., position already closed or missing). Warns when `parent_order_id` is missing on the exit trade, as this risks cross-bracket interference (matching the wrong bracket's exit to a position). Calls the optional `on_exit` callback for Telegram notifications. Uses try/finally to ensure the event handler is always deregistered even if callback errors occur.

**`setup_disconnect_handler(ib)`** — Attaches a callback for connection drops. Important because IBKR connections are notoriously unstable. Uses a re-entrancy guard (`_reconnecting` flag) to prevent cascading reconnect loops where a reconnect triggers another disconnect event. Calls `clear_realtime_subscriptions()` to reset subscription tracking after IBKR drops connections, preventing stale callbacks from the old connection from accumulating. Reads the shared `shutting_down` flag from `core.state` to skip reconnection during intentional shutdown (avoiding the circular import that would result from importing directly from `scheduler.py`).

### Stale Order Re-evaluation

**`get_stale_orders(ib, stale_minutes)`** — Queries IBKR via `ib.openTrades()` for all unfilled parent limit orders (where `parentId == 0` and `orderType == "LMT"` with status `Submitted` or `PreSubmitted`). Looks up order age from the `pending_orders` database table first (persisted at placement time), falling back to `trade.log[0].time` for orders placed before this tracking was added. The DB lookup is critical because `ib_insync` resets the trade log on every reconnection, which would otherwise make all orders appear brand new. Returns a list of dicts with the Trade object, ticker, exchange, and age in minutes for orders older than the threshold.

**`cancel_bracket_order(ib, trade)`** — Cancels a parent entry order. IBKR automatically cancels the attached child orders (take-profit and stop-loss) when the parent is cancelled. Also removes the `pending_orders` DB record for the cancelled order. Follows the same try/except pattern as `cancel_order()`.

The scheduler's `check_stale_orders()` orchestrates the full flow: for each stale order, it fetches fresh historical data, re-runs the screener, and cancels orders where the stock no longer passes technical screening. This runs at the beginning of every scan cycle to free up capital and position slots before new candidates are evaluated. In dry-run mode, the cancelled-order counter only increments when orders are actually cancelled, not when they merely fail screening.

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

    while not shutting_down:
        if any market is open:
            run_scan_cycle(ib, markets, mode)

        # Sleep using ib_insync's event loop (not time.sleep!)
        ib.sleep(SCAN_INTERVAL_MINUTES * 60)
```

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
  - Cleaned up on fill or cancellation
```

### Key Operations

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

**`save_pending_order(perm_id, ticker)`** — Records when a parent order was placed. Called from `place_order()` after the bracket is submitted. Uses `INSERT OR IGNORE` so duplicate calls are safe.

**`get_pending_order_time(perm_id)`** — Returns the original placement datetime for a pending order, or `None` if not found. Used by `get_stale_orders()` to calculate accurate order age across reconnections.

**`remove_pending_order(perm_id)`** — Deletes a pending order record. Called from `cancel_bracket_order()`, `cancel_order()`, and `setup_fill_handler()` when orders are cancelled or filled.

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
- Mode (paper/live)
- Account summary (portfolio value, cash available, invested amount, unrealized P&L)
- Daily realized P&L
- Open positions (ticker, quantity, entry price)
- Last scan summary

This runs on a daemon thread so it doesn't interfere with trading. The polling uses Telegram's long-polling with a 10-second timeout, so responses come within seconds.

### Design

Notifications are fire-and-forget. If Telegram is down or the token is missing, the error is logged but the system keeps trading. Notifications are nice-to-have, not critical path.

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
     Look at today's high and low:
       - If high >= take_profit → close at take_profit (win!)
       - If low <= stop_loss → close at stop_loss (loss)

  2. BUILD DATA — Gather all price history UP TO today (not including future!)
     This is critical: NO LOOK-AHEAD BIAS.
     On March 15, the screener only sees data through March 15.

  3. SCREEN — Run the same screener with same settings

  4. RISK CHECK — Run the same risk manager

  5. SIMULATE FILL — "Buy" at today's close price + slippage

  6. RECORD EQUITY — Write down portfolio value at end of day

After all days:
  Close any remaining open positions at last day's close
  Calculate performance metrics
```

### No Look-Ahead Bias

This is the #1 sin of backtesting: accidentally using future data to make past decisions. Our protection:

```python
# Only use data up to current date
historical = full_data[full_data.index <= current_date]
candidates = screen_stocks(historical)  # screener only sees past
```

The screener genuinely doesn't know what happens tomorrow.

### SimulatedPortfolio

An in-memory portfolio that tracks:
- Cash remaining
- Open positions
- Closed trades
- Equity curve (portfolio value at end of each day)

No SQLite needed — it's all in memory since it's just a simulation.

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
9. If --once → run single scan cycle → exit
10. Otherwise → start scheduler loop (runs until Ctrl+C)
```

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
| `test_stale_orders.py` | Persistent order timestamps, stale detection, DB cleanup on fill/cancel |
| `test_telegram.py` | Status commands, portfolio display, risk notifications |
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

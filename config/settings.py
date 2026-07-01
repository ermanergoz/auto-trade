"""All configurable parameters for the auto-trader system."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root or config directory
_project_root = Path(__file__).resolve().parent.parent
_env_paths = [_project_root / ".env", _project_root / "config" / ".env"]
for _env_path in _env_paths:
    if _env_path.exists():
        load_dotenv(_env_path)
        break

# ---------------------------------------------------------------------------
# Broker (IBKR)
# ---------------------------------------------------------------------------
IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", "7497"))
IBKR_CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", "1"))
IBKR_TIMEOUT = 10  # seconds

# IBC (IB Controller) — manages gateway lifecycle and auto-login
IBC_PATH = os.getenv("IBC_PATH", str(Path.home() / "ibc"))
IBC_INI = os.getenv("IBC_INI", str(Path.home() / "ibc" / "config.ini"))
TWS_PATH = os.getenv("TWS_PATH", str(Path.home() / "Jts"))
TWS_VERSION = int(os.getenv("TWS_VERSION", "1037"))
IBC_USERID = os.getenv("IBC_USERID", "")
IBC_PASSWORD = os.getenv("IBC_PASSWORD", "")

# ---------------------------------------------------------------------------
# Markets
# ---------------------------------------------------------------------------
MARKETS = ["US"]
EXCLUDED_SECTORS = ["Financials"]
FINANCIAL_KEYWORDS = [
    "bank", "insurance", "lending", "mortgage", "loan", "credit",
    "capital markets", "consumer finance", "financial",
    "diversified finan", "investment companies", "private equity",
    "savings & loans", "closed-end funds", "sovereign",
    "microfinance", "payday", "debt", "usury",
]
DEFENSE_KEYWORDS = [
    "defense", "defence", "military", "weapon", "arms", "ammunition",
    "aerospace & defense", "munition", "missile", "combat",
    "ordnance", "warship", "armament",
]
EXCLUDED_COUNTRIES = {"Israel"}

# Specific tickers to always exclude from universe
EXCLUDED_TICKERS = {
    "CHKP", "MNDY", "CYBR", "TEVA", "WIX", "NICE", "INMD", "GILT",
    "CEVA", "SILC", "RDWR", "MGIC", "DSNY", "SEDG", "FVRR", "GLBE",
    "RSKD", "GLMD", "ELBM", "AURA", "CRNT", "ORMP", "MRVL",
    "ARQT", "CPRI", "ELBT", "KRNT", "OPAL", "PERI", "RVSN",
    "SGHT", "SMWB", "TOVX", "MNDO", "BSQR", "PRGO",
}
MIN_DAILY_VOLUME = 100_000
MIN_MARKET_CAP = 50_000_000  # $50M

# Market hours expressed in the market's native timezone — must be the
# exchange's local time so DST transitions are handled correctly. NYSE is in
# America/New_York which observes DST; Istanbul (TRT) is a fixed UTC+3 offset
# and would drift one hour off NYSE every winter if hours were stored there.
MARKET_HOURS = {
    "US": {"open": "09:30", "close": "16:00", "tz": "America/New_York"},
}

# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
SCAN_INTERVAL_MINUTES = 15
AI_CONFIDENCE_THRESHOLD = 65
AI_MAX_CANDIDATES = 0               # Max candidates sent to AI per cycle (0 = unlimited)

# Trade cadence. Swing is the default on cost/edge grounds for a sub-$25k retail
# account (day-trading loses to commissions/slippage/noise). The day-trade code
# path stays in the codebase but is OFF unless DAY_TRADE_ENABLED is true — the
# Phase-2 edge-validation harness decides day-vs-swing out-of-sample, not assumption.
DEFAULT_TRADE_TYPE = os.getenv("DEFAULT_TRADE_TYPE", "swing").lower()
DAY_TRADE_ENABLED = os.getenv("DAY_TRADE_ENABLED", "false").lower() == "true"

# LLM provider — "gemini" (primary; auto-falls back to Ollama on error/exhaustion)
# or "ollama" (legacy local-only path). Lower-cased so env vars like "Gemini" work.
AI_PROVIDER = os.getenv("AI_PROVIDER", "gemini").lower()

# Gemini (primary when AI_PROVIDER=gemini). Missing key silently falls back to
# Ollama — matching the TAVILY_API_KEY "optional API" precedent.
#
# Two ways to configure keys, both checked at startup:
#   GEMINI_API_KEYS=key1,key2,key3   ← preferred: comma-separated rotation list
#   GEMINI_API_KEY=key1              ← legacy: single key (used only when KEYS unset)
# Rotation triples free-tier RPD headroom — see docs/CODE-DOCUMENTATION.md.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


def _parse_gemini_keys(api_keys_env: str, single_key_env: str) -> list[str]:
    """Resolve the active Gemini key list.

    Precedence:
        1. GEMINI_API_KEYS (comma-separated, whitespace trimmed, empty
           segments dropped). Wins whenever it parses to ≥1 keys.
        2. GEMINI_API_KEY (single key) — backward-compat fallback.
        3. Empty list — no keys configured; analyst will raise and the
           router falls through to Ollama.
    """
    if api_keys_env:
        keys = [k.strip() for k in api_keys_env.split(",") if k.strip()]
        if keys:
            return keys
    if single_key_env:
        return [single_key_env]
    return []


GEMINI_API_KEYS = _parse_gemini_keys(
    os.getenv("GEMINI_API_KEYS", ""),
    GEMINI_API_KEY,
)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_HOST = os.getenv("GEMINI_HOST", "https://generativelanguage.googleapis.com")

# Ollama (fallback) — local, no key required.
AI_MODEL = os.getenv("AI_MODEL", "qwen2.5:7b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Risk Management
# ---------------------------------------------------------------------------
MAX_POSITION_SIZE_PCT = 15.0    # Hard ceiling per position; RISK_PER_TRADE_PCT is the binding sizing constraint
DAILY_LOSS_LIMIT_PCT = 10.0
# Caps total open at-risk capital (sum of entry→stop risk across open positions)
# as a % of equity. Intentionally tighter than DAILY_LOSS_LIMIT_PCT — a tighter
# concentration sibling of the daily-loss-tied cumulative-risk check.
MAX_PORTFOLIO_HEAT_PCT = 6.0
MAX_OPEN_POSITIONS = 3
DEFAULT_STOP_LOSS_PCT = 3.0
DEFAULT_TAKE_PROFIT_PCT = 6.0
MAX_SECTOR_CONCENTRATION_PCT = 50.0

# Discipline rules
ANTI_MOMENTUM_PCT = 8.0         # Reject if price moved >8% from signal entry
MAX_EXTENSION_OVER_MA20_PCT = 15.0  # Drop screener candidates whose close is more than this % above MA20 (anti-chase at the source). Tightened from 20% after the bot bought stocks at the local peak again. 6-month sweep on 2026-04-28 (data/sweep_extension_pct_2026-04-28.csv) shows 15% is the sweet spot: 66.7% win rate vs 50% at 20%, +8.32% return vs -0.06% at 20%, and still keeps 6 trades vs 8 at 20%. Trades in the 16–20% band were systematically losers (avg ~+$8/trade at 20%, +$1500/trade at 15%).
TREND_CONFIRMATION = True       # Require MA5 > MA10 > MA20 alignment for buys
MIN_RISK_REWARD_RATIO = 1.5     # Minimum reward/risk ratio
RISK_PER_TRADE_PCT = 5.0        # Risk per trade as % of portfolio (used in sizing + cumulative risk)
ALLOW_SHORT_SELLING = False     # Block sell signals for stocks not currently held
VOLATILITY_BASELINE = 0.20      # Baseline annualized volatility (20%) for position scaling
CHECK_ANALYST_CONSENSUS = True  # Block BUY unless BOTH yfinance and IBKR analyst consensus are buy/strong_buy
CORRELATION_CAP_THRESHOLD = 0.7 # Reject candidate if return-correlation with any open position exceeds this (1.0 disables)

# Circuit breaker — pause trading after consecutive losses
CIRCUIT_BREAKER_LOSSES = 3      # Number of consecutive losses to trip
CIRCUIT_BREAKER_WINDOW_MIN = 60 # Time window in minutes to look back

# ---------------------------------------------------------------------------
# Intraday Margin (post-2026-06-04 framework)
# ---------------------------------------------------------------------------
# FINRA's Pattern Day Trader rule was eliminated 2026-06-04 (Notice 26-10) and
# replaced by a real-time intraday-margin framework under Rule 4210. The old
# $5,000 day-trade gate is dead code AND a known bug (an $8k–$24,999 account
# got zero protection from it). The real constraints now are the Reg-T
# account-equity minimum, the 25% intraday maintenance margin, and avoiding
# repeated uncured intraday-margin deficits (which can trigger a 90-day
# restriction). See .planning/codebase/CONCERNS.md and PROJECT.md Constraints.
REG_T_MIN_EQUITY_USD = 2000.0           # Reg-T minimum account equity (USD)
INTRADAY_MAINTENANCE_MARGIN_PCT = 25.0  # Intraday maintenance margin requirement (%)

# Operator-confirmable margin regime. Allowed values:
#   "intraday"   — only the new intraday-margin guard runs
#   "legacy_pdt" — only the legacy day-trade counter runs
#   "both"       — run every applicable guard (conservative default)
# The default "both" keeps every guard active until the operator confirms their
# IBKR account's regime during the broker phase-in (through 2027-10-20).
# See MGN-03 / STATE.md Phase-1 blocker.
MARGIN_REGIME = os.getenv("MARGIN_REGIME", "both").lower()

# Legacy PDT day-trade counter — retained only as a safety net behind
# MARGIN_REGIME ("legacy_pdt"/"both"). Uses the CORRECT $25k legacy PDT equity
# threshold, never the eliminated $5k value.
LEGACY_PDT_THRESHOLD_USD = 25000.0
PDT_MAX_DAY_TRADES_PER_5_DAYS = 1   # Block the trade that would take us to this count (IBKR trips at 2)

# Stale order re-evaluation
STALE_ORDER_MINUTES = 1440      # Re-screen unfilled orders after 24 hours

# ---------------------------------------------------------------------------
# Day Trading
# ---------------------------------------------------------------------------
CLOSE_DAY_TRADES_BEFORE_MARKET_CLOSE = True
CLOSE_MINUTES_BEFORE = 15

# ---------------------------------------------------------------------------
# Swing Hold Horizon (LLM veto gate — D-06)
# ---------------------------------------------------------------------------
# Maximum expected swing-trade hold horizon, in TRADING days. Used by the LLM
# veto gate (core.gate.gate_signal) as the fallback earnings-blackout window:
# a CONFIRMED earnings date landing within the per-trade hold horizon vetoes a
# fresh entry (a swing position must not ride a binary earnings gap). Grounded in
# a ~2-week swing hold; ATR-scaled per-trade horizons are a future refinement.
MAX_SWING_HOLD_DAYS = 10

# ---------------------------------------------------------------------------
# Screening Thresholds
# ---------------------------------------------------------------------------
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MA_FAST = 5
MA_SLOW = 20
VOLUME_SPIKE_MULTIPLIER = 2.0
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0
SUPPORT_RESISTANCE_PCT = 2.0

# Indicator weights for scoring — higher weight = more influence on score.
# Default 1.0 = equal weighting. Set to 0.0 to disable an indicator's contribution.
INDICATOR_WEIGHTS = {
    "RSI": 1.0,
    "MACD": 1.0,
    "MA_CROSSOVER": 1.0,
    "VOLUME_SPIKE": 1.0,
    "BOLLINGER": 1.0,
    "SUPPORT": 1.0,
    "RESISTANCE": 1.0,
}

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = _project_root
DB_PATH = _project_root / "data" / "portfolio.db"
LOG_DIR = _project_root / "logs"
DATA_DIR = _project_root / "data"

# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------
BACKTEST_SLIPPAGE_PCT = 0.1
# IBKR tiered pricing: $0.005/share with $1 minimum per order. A flat
# per-trade commission systematically understates friction on large
# positions (1000+ shares) and inflates reported Sharpe/profit factor.
BACKTEST_COMMISSION = 1.0             # Minimum $ per trade (IBKR min)
BACKTEST_COMMISSION_PER_SHARE = 0.005  # $ per share on top of the minimum
# Bid-ask spread cost in basis points, crossed on EACH leg (the
# half-the-spread model): entry crosses the ask and pays up, exit crosses
# the bid and receives less. ~1-5 bps for liquid large-cap, 10-30+ for
# small-cap. Charged per leg on top of slippage_pct, so a round trip pays
# the spread twice. Set to 0 to reproduce the pre-spread fills. See the
# STACK cost table (round-trip cost ~ commission_min + 2 x (half_spread + slippage)).
BACKTEST_SPREAD_BPS = 5.0  # basis points per leg (half-spread crossed each side)
RISK_FREE_RATE = 0.05  # for Sharpe ratio

# ---------------------------------------------------------------------------
# Timezone
# ---------------------------------------------------------------------------
TIMEZONE = "Europe/Istanbul"


def is_paper_mode() -> bool:
    """Check if connected to paper trading (port 7497/4002) vs live (port 7496/4001)."""
    return IBKR_PORT in (7497, 4002)


def validate_settings() -> list[str]:
    """Validate configuration values at startup. Returns list of errors (empty = OK)."""
    errors = []

    if IBKR_PORT not in (7496, 7497, 4001, 4002):
        errors.append(f"IBKR_PORT must be 7496/4001 (live) or 7497/4002 (paper), got {IBKR_PORT}")

    if MAX_POSITION_SIZE_PCT <= 0 or MAX_POSITION_SIZE_PCT > 100:
        errors.append(f"MAX_POSITION_SIZE_PCT must be 0-100, got {MAX_POSITION_SIZE_PCT}")

    if DAILY_LOSS_LIMIT_PCT <= 0 or DAILY_LOSS_LIMIT_PCT > 50:
        errors.append(f"DAILY_LOSS_LIMIT_PCT must be 0-50, got {DAILY_LOSS_LIMIT_PCT}")

    if MAX_OPEN_POSITIONS <= 0:
        errors.append(f"MAX_OPEN_POSITIONS must be positive, got {MAX_OPEN_POSITIONS}")

    if MIN_RISK_REWARD_RATIO <= 0:
        errors.append(f"MIN_RISK_REWARD_RATIO must be positive, got {MIN_RISK_REWARD_RATIO}")

    if SCAN_INTERVAL_MINUTES < 1:
        errors.append(f"SCAN_INTERVAL_MINUTES must be >= 1, got {SCAN_INTERVAL_MINUTES}")

    if DEFAULT_STOP_LOSS_PCT <= 0:
        errors.append(f"DEFAULT_STOP_LOSS_PCT must be positive, got {DEFAULT_STOP_LOSS_PCT}")

    if BACKTEST_SPREAD_BPS < 0:
        errors.append(f"BACKTEST_SPREAD_BPS must be >= 0, got {BACKTEST_SPREAD_BPS}")

    if not (0 < AI_CONFIDENCE_THRESHOLD <= 100):
        errors.append(f"AI_CONFIDENCE_THRESHOLD must be 1-100, got {AI_CONFIDENCE_THRESHOLD}")

    if BOLLINGER_STD <= 0:
        errors.append(f"BOLLINGER_STD must be positive, got {BOLLINGER_STD}")

    if SUPPORT_RESISTANCE_PCT <= 0:
        errors.append(f"SUPPORT_RESISTANCE_PCT must be positive, got {SUPPORT_RESISTANCE_PCT}")

    if RISK_PER_TRADE_PCT <= 0 or RISK_PER_TRADE_PCT > 10:
        errors.append(f"RISK_PER_TRADE_PCT must be 0-10, got {RISK_PER_TRADE_PCT}")

    if STALE_ORDER_MINUTES > 0 and STALE_ORDER_MINUTES < SCAN_INTERVAL_MINUTES:
        errors.append(
            f"STALE_ORDER_MINUTES ({STALE_ORDER_MINUTES}) should be >= "
            f"SCAN_INTERVAL_MINUTES ({SCAN_INTERVAL_MINUTES})"
        )

    if MAX_SECTOR_CONCENTRATION_PCT <= 0 or MAX_SECTOR_CONCENTRATION_PCT > 100:
        errors.append(f"MAX_SECTOR_CONCENTRATION_PCT must be 0-100, got {MAX_SECTOR_CONCENTRATION_PCT}")

    if DEFAULT_TAKE_PROFIT_PCT <= 0:
        errors.append(f"DEFAULT_TAKE_PROFIT_PCT must be positive, got {DEFAULT_TAKE_PROFIT_PCT}")

    if MAX_SWING_HOLD_DAYS <= 0:
        errors.append(f"MAX_SWING_HOLD_DAYS must be positive, got {MAX_SWING_HOLD_DAYS}")

    if CIRCUIT_BREAKER_LOSSES < 0:
        errors.append(f"CIRCUIT_BREAKER_LOSSES must be non-negative, got {CIRCUIT_BREAKER_LOSSES}")

    if CIRCUIT_BREAKER_WINDOW_MIN <= 0:
        errors.append(f"CIRCUIT_BREAKER_WINDOW_MIN must be positive, got {CIRCUIT_BREAKER_WINDOW_MIN}")

    if REG_T_MIN_EQUITY_USD <= 0:
        errors.append(f"REG_T_MIN_EQUITY_USD must be positive, got {REG_T_MIN_EQUITY_USD}")

    if not (0 < INTRADAY_MAINTENANCE_MARGIN_PCT <= 100):
        errors.append(f"INTRADAY_MAINTENANCE_MARGIN_PCT must be 0-100, got {INTRADAY_MAINTENANCE_MARGIN_PCT}")

    if LEGACY_PDT_THRESHOLD_USD < 0:
        errors.append(f"LEGACY_PDT_THRESHOLD_USD must be non-negative, got {LEGACY_PDT_THRESHOLD_USD}")

    if MARGIN_REGIME not in ("intraday", "legacy_pdt", "both"):
        errors.append(
            f"MARGIN_REGIME must be one of intraday/legacy_pdt/both, got {MARGIN_REGIME!r}"
        )

    if DEFAULT_TRADE_TYPE not in ("day", "swing"):
        errors.append(f"DEFAULT_TRADE_TYPE must be one of day/swing, got {DEFAULT_TRADE_TYPE!r}")

    if not (0 < MAX_PORTFOLIO_HEAT_PCT <= 100):
        errors.append(f"MAX_PORTFOLIO_HEAT_PCT must be 0-100, got {MAX_PORTFOLIO_HEAT_PCT}")

    if PDT_MAX_DAY_TRADES_PER_5_DAYS < 0:
        errors.append(f"PDT_MAX_DAY_TRADES_PER_5_DAYS must be non-negative, got {PDT_MAX_DAY_TRADES_PER_5_DAYS}")

    return errors

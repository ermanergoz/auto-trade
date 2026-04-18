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
AI_MODEL = os.getenv("AI_MODEL", "qwen2.5:7b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Risk Management
# ---------------------------------------------------------------------------
MAX_POSITION_SIZE_PCT = 50.0
DAILY_LOSS_LIMIT_PCT = 10.0
MAX_OPEN_POSITIONS = 3
DEFAULT_STOP_LOSS_PCT = 3.0
DEFAULT_TAKE_PROFIT_PCT = 6.0
MAX_SECTOR_CONCENTRATION_PCT = 50.0

# Discipline rules
ANTI_MOMENTUM_PCT = 8.0         # Reject if price moved >8% from signal entry
MAX_EXTENSION_OVER_MA20_PCT = 20.0  # Drop screener candidates whose close is more than this % above MA20 (anti-chase at the source). Tuned via 6-month sweep — 20% matched unfiltered return with better win rate.
TREND_CONFIRMATION = True       # Require MA5 > MA10 > MA20 alignment for buys
MIN_RISK_REWARD_RATIO = 1.5     # Minimum reward/risk ratio
RISK_PER_TRADE_PCT = 5.0        # Risk per trade as % of portfolio (used in sizing + cumulative risk)
ALLOW_SHORT_SELLING = False     # Block sell signals for stocks not currently held
VOLATILITY_BASELINE = 0.20      # Baseline annualized volatility (20%) for position scaling
CHECK_ANALYST_CONSENSUS = True  # Block BUY when analyst consensus is sell/strong sell
CORRELATION_CAP_THRESHOLD = 0.7 # Reject candidate if return-correlation with any open position exceeds this (1.0 disables)

# Circuit breaker — pause trading after consecutive losses
CIRCUIT_BREAKER_LOSSES = 3      # Number of consecutive losses to trip
CIRCUIT_BREAKER_WINDOW_MIN = 60 # Time window in minutes to look back

# Pattern Day Trader (PDT) protection
# IBKR restricts accounts with liquid net worth below the threshold if they
# execute 2 day trades within a rolling 5-business-day window (30-day lockout).
# When portfolio value is at or above the threshold, PDT rules don't apply
# here and trading is unconstrained by this check.
PDT_PROTECTION_THRESHOLD_USD = 5000.0
PDT_MAX_DAY_TRADES_PER_5_DAYS = 1   # Block the trade that would take us to this count (IBKR trips at 2)

# Stale order re-evaluation
STALE_ORDER_MINUTES = 1440      # Re-screen unfilled orders after 24 hours

# ---------------------------------------------------------------------------
# Day Trading
# ---------------------------------------------------------------------------
CLOSE_DAY_TRADES_BEFORE_MARKET_CLOSE = True
CLOSE_MINUTES_BEFORE = 15

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
BACKTEST_COMMISSION = 1.0  # $ per trade
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

    if CIRCUIT_BREAKER_LOSSES < 0:
        errors.append(f"CIRCUIT_BREAKER_LOSSES must be non-negative, got {CIRCUIT_BREAKER_LOSSES}")

    if CIRCUIT_BREAKER_WINDOW_MIN <= 0:
        errors.append(f"CIRCUIT_BREAKER_WINDOW_MIN must be positive, got {CIRCUIT_BREAKER_WINDOW_MIN}")

    if PDT_PROTECTION_THRESHOLD_USD < 0:
        errors.append(f"PDT_PROTECTION_THRESHOLD_USD must be non-negative, got {PDT_PROTECTION_THRESHOLD_USD}")

    if PDT_MAX_DAY_TRADES_PER_5_DAYS < 0:
        errors.append(f"PDT_MAX_DAY_TRADES_PER_5_DAYS must be non-negative, got {PDT_MAX_DAY_TRADES_PER_5_DAYS}")

    return errors

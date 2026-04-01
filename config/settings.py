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

# ---------------------------------------------------------------------------
# Markets
# ---------------------------------------------------------------------------
MARKETS = ["US"]
EXCLUDED_SECTORS = ["Financials"]

# Israel-based companies trading on US exchanges — excluded from universe
EXCLUDED_TICKERS = {
    "CHKP", "MNDY", "CYBR", "TEVA", "WIX", "NICE", "INMD", "GILT",
    "CEVA", "SILC", "RDWR", "MGIC", "DSNY", "SEDG", "FVRR", "GLBE",
    "RSKD", "GLMD", "ELBM", "AURA", "CRNT", "ORMP", "MRVL",
    "ARQT", "CPRI", "ELBT", "KRNT", "OPAL", "PERI", "RVSN",
    "SGHT", "SMWB", "TOVX", "MNDO", "BSQR", "PRGO",
}
MIN_DAILY_VOLUME = 100_000
MIN_MARKET_CAP = 50_000_000  # $50M

# Market hours in Europe/Istanbul timezone (TRT)
MARKET_HOURS = {
    "US": {"open": "16:30", "close": "23:00"},
}

# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
SCAN_INTERVAL_MINUTES = 15
AI_CONFIDENCE_THRESHOLD = 70
AI_MODEL = os.getenv("AI_MODEL", "qwen2.5:7b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Risk Management
# ---------------------------------------------------------------------------
MAX_POSITION_SIZE_PCT = 5.0
DAILY_LOSS_LIMIT_PCT = 2.0
MAX_OPEN_POSITIONS = 10
DEFAULT_STOP_LOSS_PCT = 3.0
DEFAULT_TAKE_PROFIT_PCT = 6.0
MAX_SECTOR_CONCENTRATION_PCT = 25.0

# Discipline rules
ANTI_MOMENTUM_PCT = 5.0         # Reject if price moved >5% from signal entry
TREND_CONFIRMATION = True       # Require MA5 > MA10 > MA20 alignment for buys
MIN_RISK_REWARD_RATIO = 1.5     # Minimum reward/risk ratio
ALLOW_SHORT_SELLING = False     # Block sell signals for stocks not currently held

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
    """Check if connected to paper trading (port 7497) vs live (port 7496)."""
    return IBKR_PORT == 7497

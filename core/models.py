"""Data models for the auto-trader system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class Action(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class TradeType(Enum):
    DAY = "day"
    SWING = "swing"


class Market(Enum):
    US = "US"


@dataclass
class Signal:
    """A trade signal from the screener or AI analyst."""

    ticker: str
    action: Action
    confidence: float  # 0-100
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: str
    source: str  # "screener" or "ai"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    exchange: str = ""
    trade_type: TradeType = TradeType.DAY
    indicator_values: dict = field(default_factory=dict)


@dataclass
class Position:
    """An open position in the portfolio."""

    ticker: str
    exchange: str
    quantity: int
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    trade_type: TradeType
    sector: str = ""
    current_price: Optional[float] = None
    id: Optional[int] = None

    @property
    def unrealized_pnl(self) -> Optional[float]:
        if self.current_price is None:
            return None
        return (self.current_price - self.entry_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> Optional[float]:
        if self.current_price is None:
            return None
        if self.entry_price == 0:
            return 0.0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100


@dataclass
class Trade:
    """A completed (closed) trade."""

    ticker: str
    exchange: str
    quantity: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    trade_type: TradeType
    sector: str = ""
    reasoning: str = ""
    id: Optional[int] = None

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.quantity

    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        pct = ((self.exit_price - self.entry_price) / self.entry_price) * 100
        # For shorts (negative quantity), the return is inverted
        if self.quantity < 0:
            pct = -pct
        return pct

    @property
    def duration(self) -> float:
        """Duration in hours."""
        return (self.exit_time - self.entry_time).total_seconds() / 3600


@dataclass
class DailySummary:
    """End-of-day portfolio summary."""

    date: datetime
    portfolio_value: float
    daily_pnl: float
    daily_pnl_pct: float
    num_trades: int
    winning_trades: int = 0
    losing_trades: int = 0
    id: Optional[int] = None


@dataclass
class StockInfo:
    """Basic information about a tradeable stock."""

    ticker: str
    exchange: str
    sector: str
    market_cap: float
    avg_volume: float
    currency: str = "USD"
    name: str = ""
    country: str = ""

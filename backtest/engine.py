"""Backtesting engine — replays historical data through the same strategy code.

Uses the EXACT same screener and risk manager as live trading.
Only replaces the data source (YFinance) and execution (simulated).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional

import pandas as pd

from config.settings import (
    BACKTEST_SLIPPAGE_PCT, BACKTEST_COMMISSION,
    DEFAULT_STOP_LOSS_PCT, DEFAULT_TAKE_PROFIT_PCT,
)
from core.models import Signal, Position, Trade, Action, TradeType
from core.screener import screen_stocks
from core.risk import evaluate, RiskResult, calculate_realized_volatility
from core.data import get_historical_data_yfinance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simulated portfolio (mirrors portfolio.py interface, in-memory)
# ---------------------------------------------------------------------------

@dataclass
class SimulatedPortfolio:
    """In-memory portfolio for backtesting."""
    initial_capital: float = 100_000.0
    cash: float = 0.0
    positions: list[Position] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[tuple[date, float]] = field(default_factory=list)
    slippage_pct: float = BACKTEST_SLIPPAGE_PCT
    commission: float = BACKTEST_COMMISSION

    def __post_init__(self):
        if self.cash == 0:
            self.cash = self.initial_capital

    def portfolio_value_mtm(self, current_prices: dict[str, float] | None = None) -> float:
        """Mark-to-market portfolio value using current prices when available."""
        position_value = sum(
            current_prices.get(p.ticker, p.entry_price) * p.quantity
            if current_prices else p.entry_price * p.quantity
            for p in self.positions
        )
        return self.cash + position_value

    @property
    def portfolio_value(self) -> float:
        return self.portfolio_value_mtm()

    @property
    def daily_pnl(self) -> float:
        if not self.equity_curve:
            return 0.0
        return self.portfolio_value - self.equity_curve[-1][1]

    def open_position(self, signal: Signal, quantity: int, fill_price: float, current_date: datetime) -> None:
        """Open a new position with slippage and commission."""
        # Apply slippage
        if signal.action == Action.BUY:
            adjusted_price = fill_price * (1 + self.slippage_pct / 100)
        else:
            adjusted_price = fill_price * (1 - self.slippage_pct / 100)

        cost = adjusted_price * quantity + self.commission
        if cost > self.cash:
            logger.debug("Insufficient cash for %s: need $%.2f, have $%.2f",
                        signal.ticker, cost, self.cash)
            return

        self.cash -= cost
        self.positions.append(Position(
            ticker=signal.ticker,
            exchange=signal.exchange,
            quantity=quantity,
            entry_price=adjusted_price,
            entry_time=current_date,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            trade_type=signal.trade_type,
            sector=signal.indicator_values.get("sector", ""),
        ))

    def close_position(self, ticker: str, exit_price: float, exit_time: datetime) -> Optional[Trade]:
        """Close a position and record the trade."""
        pos = None
        for i, p in enumerate(self.positions):
            if p.ticker == ticker:
                pos = self.positions.pop(i)
                break

        if not pos:
            return None

        # Apply slippage on exit (direction depends on position side)
        if pos.quantity > 0:  # long: selling, slippage lowers exit price
            adjusted_exit = exit_price * (1 - self.slippage_pct / 100)
        else:  # short: buying back, slippage raises exit price
            adjusted_exit = exit_price * (1 + self.slippage_pct / 100)
        self.cash += adjusted_exit * pos.quantity - self.commission

        trade = Trade(
            ticker=pos.ticker,
            exchange=pos.exchange,
            quantity=pos.quantity,
            entry_price=pos.entry_price,
            exit_price=adjusted_exit,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            trade_type=pos.trade_type,
            sector=pos.sector,
        )
        self.trades.append(trade)
        return trade

    def record_equity(self, current_date: date) -> None:
        """Snapshot equity for the day."""
        self.equity_curve.append((current_date, self.portfolio_value))


# ---------------------------------------------------------------------------
# Stop-loss / take-profit checking
# ---------------------------------------------------------------------------

def _check_exits(portfolio: SimulatedPortfolio, day_data: dict[str, pd.Series], bar_date: datetime) -> None:
    """Check if any open positions hit their stop-loss or take-profit."""
    to_close = []

    for pos in portfolio.positions:
        if pos.ticker not in day_data:
            continue

        bar = day_data[pos.ticker]
        low = bar["low"]
        high = bar["high"]

        if pos.quantity > 0:  # Long position
            # Stop-loss hit — fill at the worse of stop price or bar low (gap-down)
            if low <= pos.stop_loss:
                actual_exit = min(low, pos.stop_loss)
                to_close.append((pos.ticker, actual_exit, "stop-loss"))
            # Take-profit hit — limit order fills at target price
            elif high >= pos.take_profit:
                to_close.append((pos.ticker, pos.take_profit, "take-profit"))
        else:  # Short position
            # Stop-loss hit — fill at worse of stop price or bar high (gap-up)
            if high >= pos.stop_loss:
                actual_exit = max(high, pos.stop_loss)
                to_close.append((pos.ticker, actual_exit, "stop-loss"))
            # Take-profit hit — limit order fills at target price
            elif low <= pos.take_profit:
                to_close.append((pos.ticker, pos.take_profit, "take-profit"))

    for ticker, exit_price, reason in to_close:
        trade = portfolio.close_position(ticker, exit_price, bar_date)
        if trade:
            logger.debug(
                "Closed %s at $%.2f (%s) — P&L: $%.2f",
                ticker, exit_price, reason, trade.pnl,
            )


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    tickers: list[str]
    market: str = "US"
    start_date: str = ""  # YYYY-MM-DD, empty = use all available
    end_date: str = ""
    initial_capital: float = 100_000.0
    slippage_pct: float = BACKTEST_SLIPPAGE_PCT
    commission: float = BACKTEST_COMMISSION
    use_ai: bool = False  # AI analysis is expensive; default to screener-only
    min_screener_score: float = 15.0
    indicator_weights: dict[str, float] | None = None
    use_volatility_scaling: bool = False


def run_backtest(config: BacktestConfig) -> SimulatedPortfolio:
    """Run a backtest over historical data.

    Steps per bar:
    1. Check stop-loss/take-profit exits
    2. Run screener on data up to current bar (no look-ahead)
    3. Optionally run AI analyst
    4. Pass signals through risk manager
    5. Simulate execution
    """
    logger.info(
        "Starting backtest: %d tickers, capital=$%,.0f, market=%s",
        len(config.tickers), config.initial_capital, config.market,
    )

    # Download all historical data upfront
    all_data: dict[str, pd.DataFrame] = {}
    for ticker in config.tickers:
        df = get_historical_data_yfinance(
            ticker, period="1y", interval="1d", market=config.market,
        )
        if not df.empty:
            all_data[ticker] = df

    if not all_data:
        logger.error("No historical data available for any tickers")
        return SimulatedPortfolio(initial_capital=config.initial_capital)

    logger.info("Downloaded data for %d/%d tickers", len(all_data), len(config.tickers))

    # Get common date range
    all_dates = set()
    for df in all_data.values():
        all_dates.update(df.index.date if isinstance(df.index, pd.DatetimeIndex) else df.index)

    sorted_dates = sorted(all_dates)

    # Apply date filters
    if config.start_date:
        start = date.fromisoformat(config.start_date)
        sorted_dates = [d for d in sorted_dates if d >= start]
    if config.end_date:
        end = date.fromisoformat(config.end_date)
        sorted_dates = [d for d in sorted_dates if d <= end]

    if not sorted_dates:
        logger.error("No dates in range after filtering")
        return SimulatedPortfolio(initial_capital=config.initial_capital)

    # Need at least 60 days of warmup for indicators
    warmup = 60
    if len(sorted_dates) <= warmup:
        logger.error("Not enough data for warmup period (%d days)", len(sorted_dates))
        return SimulatedPortfolio(initial_capital=config.initial_capital)

    portfolio = SimulatedPortfolio(
        initial_capital=config.initial_capital,
        slippage_pct=config.slippage_pct,
        commission=config.commission,
    )

    # Iterate day by day (skip warmup period)
    current_dt = datetime.combine(sorted_dates[warmup], datetime.min.time())
    for i, current_date in enumerate(sorted_dates[warmup:], start=warmup):
        current_dt = datetime.combine(current_date, datetime.min.time())

        # Get current day's data for exit checks
        day_data: dict[str, pd.Series] = {}
        for ticker, df in all_data.items():
            mask = df.index.date == current_date if isinstance(df.index, pd.DatetimeIndex) else df.index == current_date
            day_rows = df[mask]
            if not day_rows.empty:
                day_data[ticker] = day_rows.iloc[-1]

        # Step 1: Check exits on open positions
        _check_exits(portfolio, day_data, current_dt)

        # Step 2: Build stock_data strictly BEFORE current date (no look-ahead).
        # The screener must only see data up to yesterday's close. Using
        # current date's full OHLC would give it information about today's
        # high/low/close that isn't available at decision time.
        stock_data: dict[str, tuple[str, pd.DataFrame]] = {}
        for ticker, df in all_data.items():
            if isinstance(df.index, pd.DatetimeIndex):
                hist = df[df.index.date < current_date]
            else:
                hist = df[df.index < current_date]

            if len(hist) >= warmup:
                exchange = "SMART"
                stock_data[ticker] = (exchange, hist)

        # Step 3: Run screener (with optional indicator weights)
        candidates = screen_stocks(
            stock_data, min_score=config.min_screener_score,
            indicator_weights=config.indicator_weights,
        )

        if not candidates:
            portfolio.record_equity(current_date)
            continue

        # Step 3b: Calculate realized volatility for position scaling
        market_volatility = None
        if config.use_volatility_scaling:
            # Use the first available stock's close series as a market proxy
            for ticker, df in all_data.items():
                if isinstance(df.index, pd.DatetimeIndex):
                    hist = df[df.index.date < current_date]
                else:
                    hist = df[df.index < current_date]
                if len(hist) >= 21:
                    market_volatility = calculate_realized_volatility(hist["close"])
                    break

        # Step 4: Risk check and execute
        # Use mark-to-market prices for accurate portfolio value
        current_prices = {t: bar["close"] for t, bar in day_data.items()}
        mtm_value = portfolio.portfolio_value_mtm(current_prices)
        for signal in candidates:
            result = evaluate(
                signal,
                portfolio.positions,
                mtm_value,
                portfolio.daily_pnl,
                volatility=market_volatility,
            )
            if not result.approved:
                continue

            # Fill at today's open price (realistic: signal from yesterday's close,
            # execution at today's open)
            if signal.ticker in day_data:
                fill_price = day_data[signal.ticker]["open"]
            else:
                fill_price = signal.entry_price

            portfolio.open_position(signal, result.position_size, fill_price, current_dt)

        portfolio.record_equity(current_date)

    # Close remaining positions at last available price
    for pos in list(portfolio.positions):
        if pos.ticker in day_data:
            last_price = day_data[pos.ticker]["close"]
        else:
            last_price = pos.entry_price
        portfolio.close_position(pos.ticker, last_price, current_dt)

    logger.info(
        "Backtest complete: %d trades, final value=$%,.2f (%.1f%% return)",
        len(portfolio.trades),
        portfolio.portfolio_value,
        (portfolio.portfolio_value / config.initial_capital - 1) * 100,
    )

    return portfolio

"""Backtesting engine — replays historical data through the same strategy code.

Uses the EXACT same screener and risk manager as live trading.
Only replaces the data source (YFinance) and execution (simulated).
"""

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, date, timedelta
from typing import Optional

import pandas as pd

from config.settings import (
    BACKTEST_SLIPPAGE_PCT, BACKTEST_COMMISSION,
    DEFAULT_STOP_LOSS_PCT, DEFAULT_TAKE_PROFIT_PCT,
    MAX_EXTENSION_OVER_MA20_PCT,
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
        """Daily P&L using entry-price valuation (no current prices).

        WARNING: If the equity curve was recorded with MTM prices (via
        record_equity(current_prices=...)), this compares entry-price-based
        portfolio_value against MTM equity — an inconsistent comparison.
        Prefer daily_pnl_mtm(current_prices) in the backtest loop.
        """
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

        # Rescale SL/TP to preserve the intended risk/reward percentage relative
        # to the actual fill price. The signal's SL/TP are computed from the
        # screener's entry_price (yesterday's close); when the next bar opens
        # with a gap, executing at the new open would otherwise leave SL/TP on
        # the wrong side of entry and invalidate the risk calibration.
        stop_loss = signal.stop_loss
        take_profit = signal.take_profit
        if signal.entry_price > 0:
            if signal.action == Action.BUY:
                sl_pct = (signal.entry_price - signal.stop_loss) / signal.entry_price
                tp_pct = (signal.take_profit - signal.entry_price) / signal.entry_price
                stop_loss = adjusted_price * (1 - sl_pct)
                take_profit = adjusted_price * (1 + tp_pct)
            else:
                sl_pct = (signal.stop_loss - signal.entry_price) / signal.entry_price
                tp_pct = (signal.entry_price - signal.take_profit) / signal.entry_price
                stop_loss = adjusted_price * (1 + sl_pct)
                take_profit = adjusted_price * (1 - tp_pct)

        if signal.action == Action.BUY:
            # Long: debit cash (pay for shares)
            cost = adjusted_price * quantity + self.commission
            if cost > self.cash:
                logger.debug("Insufficient cash for %s: need $%.2f, have $%.2f",
                            signal.ticker, cost, self.cash)
                return
            self.cash -= cost
            stored_quantity = quantity
        else:
            # Short: credit cash (receive sale proceeds)
            self.cash += adjusted_price * quantity - self.commission
            stored_quantity = -quantity

        self.positions.append(Position(
            ticker=signal.ticker,
            exchange=signal.exchange,
            quantity=stored_quantity,
            entry_price=adjusted_price,
            entry_time=current_date,
            stop_loss=stop_loss,
            take_profit=take_profit,
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

    def daily_pnl_mtm(self, current_prices: dict[str, float] | None = None) -> float:
        """Daily P&L using mark-to-market prices.

        Compares current MTM value against the last recorded equity point.
        """
        if not self.equity_curve:
            return 0.0
        return self.portfolio_value_mtm(current_prices) - self.equity_curve[-1][1]

    def record_equity(self, current_date: date, current_prices: dict[str, float] | None = None) -> None:
        """Snapshot equity for the day using mark-to-market prices."""
        self.equity_curve.append((current_date, self.portfolio_value_mtm(current_prices)))


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
        open_price = bar["open"]

        if pos.quantity > 0:  # Long position
            sl_hit = low <= pos.stop_loss
            tp_hit = high >= pos.take_profit

            if sl_hit and tp_hit:
                # Both could trigger on the same bar — use open to resolve.
                # If open gaps past one level, that level triggered at open.
                if open_price <= pos.stop_loss:
                    # Gap-down through SL: SL triggered first at open
                    to_close.append((pos.ticker, open_price, "stop-loss"))
                elif open_price >= pos.take_profit:
                    # Gap-up through TP: TP triggered first at TP price
                    to_close.append((pos.ticker, pos.take_profit, "take-profit"))
                else:
                    # Open between SL and TP: indeterminate, assume SL (conservative)
                    to_close.append((pos.ticker, pos.stop_loss, "stop-loss"))
            elif sl_hit:
                # Gap-down: open already below stop → fill at open (first available price)
                # Intraday: open above stop but low dips below → fill at stop price
                actual_exit = open_price if open_price <= pos.stop_loss else pos.stop_loss
                to_close.append((pos.ticker, actual_exit, "stop-loss"))
            elif tp_hit:
                # Take-profit hit — limit order fills at target price
                to_close.append((pos.ticker, pos.take_profit, "take-profit"))
        else:  # Short position
            sl_hit = high >= pos.stop_loss
            tp_hit = low <= pos.take_profit

            if sl_hit and tp_hit:
                # Both could trigger — use open to resolve
                if open_price >= pos.stop_loss:
                    # Gap-up through SL: SL triggered first at open
                    to_close.append((pos.ticker, open_price, "stop-loss"))
                elif open_price <= pos.take_profit:
                    # Gap-down through TP: TP triggered first at TP price
                    to_close.append((pos.ticker, pos.take_profit, "take-profit"))
                else:
                    # Indeterminate: assume SL (conservative)
                    to_close.append((pos.ticker, pos.stop_loss, "stop-loss"))
            elif sl_hit:
                # Gap-up: open already above stop → fill at open
                # Intraday: open below stop but high rises above → fill at stop price
                actual_exit = open_price if open_price >= pos.stop_loss else pos.stop_loss
                to_close.append((pos.ticker, actual_exit, "stop-loss"))
            elif tp_hit:
                # Take-profit hit — limit order fills at target price
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
    max_extension_pct: float = MAX_EXTENSION_OVER_MA20_PCT


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

        # Build current prices for MTM from today's bars (available regardless
        # of whether the screener finds candidates)
        current_prices = {t: bar["close"] for t, bar in day_data.items()}

        # Step 3: Run screener (with optional indicator weights)
        candidates = screen_stocks(
            stock_data, min_score=config.min_screener_score,
            indicator_weights=config.indicator_weights,
            max_extension_pct=config.max_extension_pct,
        )

        if not candidates:
            portfolio.record_equity(current_date, current_prices=current_prices)
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
        # Use mark-to-market prices for accurate portfolio value and daily PnL
        mtm_value = portfolio.portfolio_value_mtm(current_prices)
        mtm_daily_pnl = portfolio.daily_pnl_mtm(current_prices)
        for signal in candidates:
            # Fill at today's open price (realistic: signal from yesterday's close,
            # execution at today's open)
            if signal.ticker in day_data:
                fill_price = day_data[signal.ticker]["open"]
            else:
                fill_price = signal.entry_price

            result = evaluate(
                signal,
                portfolio.positions,
                mtm_value,
                mtm_daily_pnl,
                current_price=fill_price,
                volatility=market_volatility,
            )
            if not result.approved:
                continue

            portfolio.open_position(signal, result.position_size, fill_price, current_dt)

        portfolio.record_equity(current_date, current_prices=current_prices)

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


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    """Results from a walk-forward backtest.

    Splits the date range into an in-sample (IS) training period and an
    out-of-sample (OOS) testing period. Both are run with the same config
    and fresh capital; large IS→OOS degradation means the strategy was
    overfit to the in-sample window.
    """
    in_sample_portfolio: SimulatedPortfolio
    out_of_sample_portfolio: SimulatedPortfolio
    in_sample_metrics: dict
    out_of_sample_metrics: dict
    degradation: dict
    in_sample_start: date
    in_sample_end: date
    out_of_sample_start: date
    out_of_sample_end: date
    split_date: date


def walk_forward_backtest(
    config: BacktestConfig,
    train_ratio: float = 0.6,
) -> WalkForwardResult:
    """Split the date range into IS/OOS and run both — detects overfitting.

    A robust strategy produces similar metrics on both halves. If OOS metrics
    are much worse than IS, the strategy memorized the IS period's noise.

    Args:
        config: Standard BacktestConfig — must have both start_date and end_date.
        train_ratio: Fraction of the date range used for in-sample (default 0.6).
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(
            f"train_ratio must be between 0 and 1 (exclusive), got {train_ratio}"
        )
    if not config.start_date or not config.end_date:
        raise ValueError(
            "walk_forward_backtest requires both config.start_date and config.end_date"
        )

    start = date.fromisoformat(config.start_date)
    end = date.fromisoformat(config.end_date)
    if end <= start:
        raise ValueError(
            f"end_date ({end}) must be after start_date ({start})"
        )

    total_days = (end - start).days
    split_offset = int(total_days * train_ratio)
    split_date = start + timedelta(days=split_offset)
    # In-sample ends the day before split_date; out-of-sample starts on split_date.
    is_end = split_date - timedelta(days=1)

    # Import here to avoid circular import
    from backtest.report import calculate_metrics

    is_config = replace(
        config,
        start_date=start.isoformat(),
        end_date=is_end.isoformat(),
    )
    oos_config = replace(
        config,
        start_date=split_date.isoformat(),
        end_date=end.isoformat(),
    )

    logger.info(
        "Walk-forward: IS %s → %s (%d days), OOS %s → %s (%d days)",
        start, is_end, (is_end - start).days,
        split_date, end, (end - split_date).days,
    )

    is_portfolio = run_backtest(is_config)
    oos_portfolio = run_backtest(oos_config)

    is_metrics = calculate_metrics(
        is_portfolio.trades, is_portfolio.equity_curve, is_config.initial_capital,
    )
    oos_metrics = calculate_metrics(
        oos_portfolio.trades, oos_portfolio.equity_curve, oos_config.initial_capital,
    )

    # Degradation: OOS - IS for each numeric metric.
    degradation: dict = {}
    for key, is_val in is_metrics.items():
        oos_val = oos_metrics.get(key)
        if isinstance(is_val, (int, float)) and isinstance(oos_val, (int, float)):
            degradation[key] = oos_val - is_val

    return WalkForwardResult(
        in_sample_portfolio=is_portfolio,
        out_of_sample_portfolio=oos_portfolio,
        in_sample_metrics=is_metrics,
        out_of_sample_metrics=oos_metrics,
        degradation=degradation,
        in_sample_start=start,
        in_sample_end=is_end,
        out_of_sample_start=split_date,
        out_of_sample_end=end,
        split_date=split_date,
    )

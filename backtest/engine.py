"""Backtesting engine — replays historical data through the same strategy code.

Uses the EXACT same screener and risk manager as live trading.
Only replaces the data source (YFinance) and execution (simulated).
"""

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import (
    BACKTEST_SLIPPAGE_PCT, BACKTEST_COMMISSION, BACKTEST_COMMISSION_PER_SHARE,
    BACKTEST_SPREAD_BPS,
    DEFAULT_STOP_LOSS_PCT, DEFAULT_TAKE_PROFIT_PCT,
    MAX_EXTENSION_OVER_MA20_PCT,
)
from core.models import Signal, Position, Trade, Action, TradeType
from core.screener import screen_stocks
from core.risk import evaluate, RiskResult, calculate_realized_volatility
from core.data import get_historical_data_yfinance
from backtest.holdout import assert_range_excludes_holdout

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
    # SPY (or configured benchmark) buy-and-hold equity curve, normalized to
    # initial_capital and sampled on the SAME warmup-trimmed trading days as
    # equity_curve so strategy-vs-benchmark returns are comparable over an
    # identical window. Populated by run_backtest; empty for hand-built
    # portfolios. Consumed by the CAPM alpha/beta report.
    benchmark_curve: list[tuple[date, float]] = field(default_factory=list)
    # RAW, full-history, un-trimmed, un-normalized benchmark close series (a
    # pandas Series of close indexed by date over the entire history download).
    # This is the series 02-06's market-regime gate slices as
    # spy_df[spy_df.index.date < current_date]["close"] — it MUST stay the full
    # download, never the warmup-trimmed normalized benchmark_curve (too short
    # at the start of each OOS fold). None for hand-built portfolios.
    benchmark_prices: Optional[pd.Series] = None
    slippage_pct: float = BACKTEST_SLIPPAGE_PCT
    # commission is now the MINIMUM per order (IBKR $1 floor);
    # commission_per_share is added on top — matches IBKR tiered pricing.
    # A flat per-trade fee undercounts friction on large positions.
    commission: float = BACKTEST_COMMISSION
    commission_per_share: float = BACKTEST_COMMISSION_PER_SHARE
    # Bid-ask spread crossed on EACH leg, in basis points (half-the-spread
    # model). Entry crosses the ask (worse), exit crosses the bid (worse),
    # on top of slippage_pct. spread_bps=0 reproduces the pre-spread fills.
    spread_bps: float = BACKTEST_SPREAD_BPS

    def __post_init__(self):
        if self.cash == 0:
            self.cash = self.initial_capital
        # If the caller explicitly disables the minimum commission (common in
        # tests isolating other mechanics), also zero the per-share fee so
        # the total commission is actually zero. Without this, setting only
        # commission=0 would still charge commission_per_share × quantity.
        if self.commission <= 0:
            self.commission_per_share = 0.0

    def _commission_for(self, quantity: int) -> float:
        """IBKR-style commission: max(min, per_share * qty).

        Flat minimum + per-share fee. Mirrors IBKR tiered pricing
        (~$0.005/share, $1 minimum). A flat-per-trade model would
        underestimate friction on large positions and inflate Sharpe.
        """
        if self.commission <= 0 and self.commission_per_share <= 0:
            return 0.0
        per_share = abs(quantity) * max(self.commission_per_share, 0.0)
        return max(self.commission, per_share)

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
        """Open a new position with slippage, spread, and commission."""
        # Apply slippage + half-spread per leg. The entry leg crosses the
        # spread the wrong way: a BUY pays up (crosses the ask), a short SELL
        # receives less (crosses the bid). spread_bps is charged on top of
        # slippage_pct so a round trip pays the spread on both legs.
        spread_frac = self.spread_bps / 10_000.0
        if signal.action == Action.BUY:
            adjusted_price = fill_price * (1 + self.slippage_pct / 100 + spread_frac)
        else:
            adjusted_price = fill_price * (1 - self.slippage_pct / 100 - spread_frac)

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

        entry_commission = self._commission_for(quantity)
        if signal.action == Action.BUY:
            # Long: debit cash (pay for shares)
            cost = adjusted_price * quantity + entry_commission
            if cost > self.cash:
                logger.debug("Insufficient cash for %s: need $%.2f, have $%.2f",
                            signal.ticker, cost, self.cash)
                return
            self.cash -= cost
            stored_quantity = quantity
        else:
            # Short: credit cash (receive sale proceeds)
            self.cash += adjusted_price * quantity - entry_commission
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

        # Apply slippage + half-spread on exit (direction depends on side).
        # The exit leg also crosses the spread the wrong way: a long sells
        # into the bid (worse), a short buys back at the ask (worse).
        spread_frac = self.spread_bps / 10_000.0
        if pos.quantity > 0:  # long: selling, slippage + spread lower exit price
            adjusted_exit = exit_price * (1 - self.slippage_pct / 100 - spread_frac)
        else:  # short: buying back, slippage + spread raise exit price
            adjusted_exit = exit_price * (1 + self.slippage_pct / 100 + spread_frac)
        exit_commission = self._commission_for(pos.quantity)
        self.cash += adjusted_exit * pos.quantity - exit_commission

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
    commission_per_share: float = BACKTEST_COMMISSION_PER_SHARE
    spread_bps: float = BACKTEST_SPREAD_BPS  # bid-ask spread per leg, basis points
    use_ai: bool = False  # AI analysis is expensive; default to screener-only
    min_screener_score: float = 15.0
    indicator_weights: dict[str, float] | None = None
    use_volatility_scaling: bool = False
    max_extension_pct: float = MAX_EXTENSION_OVER_MA20_PCT
    # Multi-regime history window pulled from yfinance. Default 5y so every
    # backtest necessarily spans a 2022-style drawdown (not a single bull
    # year), exercising the strategy across regimes. Honors start_date/end_date
    # filters applied after download.
    history_period: str = "5y"
    # Passive benchmark downloaded alongside the strategy for the buy-and-hold
    # curve + CAPM alpha/beta. SPY by default. Downloaded over the FULL
    # history_period (not warmup-trimmed) so the raw close series is available
    # for 02-06's regime gate.
    benchmark_ticker: str = "SPY"
    # Random-entry control (survival-vs-edge): when True, the screener's
    # candidate selection is replaced by an independent per-ticker Bernoulli(0.5)
    # coin flip each bar (PITFALLS.md Pitfall 5). Identical sizing/exits/costs —
    # ONLY the entry decision is randomized. Deterministic per random_seed.
    use_random_entry: bool = False
    random_seed: int = 0


def _build_benchmark_curve(
    benchmark_df: Optional[pd.DataFrame],
    equity_dates: list[date],
    initial_capital: float,
) -> list[tuple[date, float]]:
    """Buy-and-hold equity curve for the benchmark, normalized to initial_capital.

    Sampled on the SAME trading days as the strategy's (warmup-trimmed) equity
    curve so the two return series cover an identical window — the first point
    is initial_capital, the last reflects first-close-to-last-close growth.
    Benchmark closes are forward/back-filled onto the strategy's dates so a
    missing benchmark bar never drops a strategy day.
    """
    if benchmark_df is None or benchmark_df.empty or not equity_dates:
        return []

    close = benchmark_df["close"]
    if isinstance(close.index, pd.DatetimeIndex):
        close = close.copy()
        close.index = close.index.date
    # Collapse any duplicate calendar dates (keep the last observation).
    close = close[~close.index.duplicated(keep="last")]

    aligned = close.reindex(equity_dates).ffill().bfill()
    vals = aligned.tolist()
    base = vals[0]
    if base == 0 or pd.isna(base):
        return []
    return [
        (d, initial_capital * (v / base))
        for d, v in zip(equity_dates, vals)
    ]


def _random_entry_candidates(
    stock_data: dict[str, tuple[str, pd.DataFrame]],
    bar_index: int,
    random_seed: int,
) -> list[Signal]:
    """Coin-flip entry control: independent per-ticker Bernoulli(p=0.5).

    For EACH ticker available on this bar, draw an independent Bernoulli(0.5)
    and select it as a candidate iff it comes up heads (PITFALLS.md Pitfall 5).
    Determinism comes from seeding a per-bar RNG with (random_seed + bar_index),
    so the same random_seed reproduces the exact same picks bar-for-bar while a
    different seed/index yields a different draw. Signals are plain long entries
    at the last close with the default SL/TP percentages — the SAME risk
    machinery, sizing, exits and costs as the real run apply downstream; ONLY
    the entry decision is randomized.
    """
    rng = np.random.RandomState((random_seed + bar_index) % (2 ** 32))
    signals: list[Signal] = []
    for ticker in sorted(stock_data.keys()):
        heads = rng.random() < 0.5
        if not heads:
            continue
        exchange, hist = stock_data[ticker]
        price = float(hist["close"].iloc[-1])
        if price <= 0:
            continue
        signals.append(Signal(
            ticker=ticker,
            action=Action.BUY,
            confidence=50.0,
            entry_price=price,
            stop_loss=price * (1 - DEFAULT_STOP_LOSS_PCT / 100),
            take_profit=price * (1 + DEFAULT_TAKE_PROFIT_PCT / 100),
            reasoning="random-entry control (Bernoulli p=0.5)",
            source="random",
            exchange=exchange,
            trade_type=TradeType.SWING,
            indicator_values={},
        ))
    return signals


def run_backtest(config: BacktestConfig) -> SimulatedPortfolio:
    """Run a backtest over historical data.

    Steps per bar:
    1. Check stop-loss/take-profit exits
    2. Run screener on data up to current bar (no look-ahead)
    3. Optionally run AI analyst
    4. Pass signals through risk manager
    5. Simulate execution

    Holdout contract: a preflight guard refuses any range overlapping the
    single-use holdout while it is locked. A run with no explicit ``end_date``
    is treated as ending today, which overlaps the locked holdout and is
    therefore refused — Phase-2 tuning runs MUST set ``config.end_date`` earlier
    than ``HOLDOUT_START`` explicitly. See ``backtest/holdout.py``.

    Raises:
        PermissionError: if the requested range overlaps the locked holdout.
    """
    # Preflight: mechanically refuse holdout-overlapping ranges before any data
    # download or iteration, so tuning runs cannot peek at the reserved test set.
    assert_range_excludes_holdout(config.start_date, config.end_date)

    logger.info(
        "Starting backtest: %d tickers, capital=$%,.0f, market=%s",
        len(config.tickers), config.initial_capital, config.market,
    )

    # Download all historical data upfront
    all_data: dict[str, pd.DataFrame] = {}
    for ticker in config.tickers:
        df = get_historical_data_yfinance(
            ticker, period=config.history_period, interval="1d", market=config.market,
        )
        if not df.empty:
            all_data[ticker] = df

    if not all_data:
        logger.error("No historical data available for any tickers")
        return SimulatedPortfolio(initial_capital=config.initial_capital)

    logger.info("Downloaded data for %d/%d tickers", len(all_data), len(config.tickers))

    # Download the benchmark (SPY) over the FULL history window — NOT
    # warmup-trimmed. Two artifacts come off this: the raw full-history close
    # series (exposed for 02-06's regime gate) and a buy-and-hold curve aligned
    # to the strategy's trading days (built after the loop, once equity_curve
    # exists). Kept as a local that survives to the regime gate / report.
    benchmark_df = get_historical_data_yfinance(
        config.benchmark_ticker, period=config.history_period,
        interval="1d", market=config.market,
    )

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
        commission_per_share=config.commission_per_share,
        spread_bps=config.spread_bps,
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

        # Decision-time MTM uses today's OPEN — the decision is enacted at
        # bar open (fills at the open), so pricing positions with bar.close
        # would leak intraday future info (high/low/close) into the risk
        # gates (daily-loss-limit, position sizing, sector concentration).
        decision_prices = {t: bar["open"] for t, bar in day_data.items()}
        # End-of-day equity uses today's CLOSE — the actual settled value.
        eod_prices = {t: bar["close"] for t, bar in day_data.items()}

        # Step 3: Pick candidates. The real strategy runs the screener; the
        # random-entry control replaces that entirely with a seeded per-ticker
        # Bernoulli(0.5) coin flip (survival-vs-edge control). Everything
        # downstream — risk, sizing, exits, costs — is identical.
        if config.use_random_entry:
            candidates = _random_entry_candidates(stock_data, i, config.random_seed)
        else:
            candidates = screen_stocks(
                stock_data, min_score=config.min_screener_score,
                indicator_weights=config.indicator_weights,
                max_extension_pct=config.max_extension_pct,
            )

        if not candidates:
            portfolio.record_equity(current_date, current_prices=eod_prices)
            continue

        # Step 4: Risk check and execute
        # Use decision-time open prices for portfolio value / daily PnL so
        # risk gates aren't comparing against prices that hadn't yet been
        # observed at decision time.
        mtm_value = portfolio.portfolio_value_mtm(decision_prices)
        mtm_daily_pnl = portfolio.daily_pnl_mtm(decision_prices)
        # Start-of-day equity = yesterday's end-of-day MTM (recorded at end
        # of prior iteration). Gives daily-loss-limit check a stable baseline
        # that doesn't drift down as today's losses accumulate.
        start_of_day_equity = portfolio.equity_curve[-1][1] if portfolio.equity_curve else mtm_value
        for signal in candidates:
            # Fill at today's open price (realistic: signal from yesterday's close,
            # execution at today's open)
            if signal.ticker in day_data:
                fill_price = day_data[signal.ticker]["open"]
            else:
                fill_price = signal.entry_price

            # Per-candidate realized volatility. Previously this used the
            # first ticker in all_data as a market proxy, which would shrink
            # every position in the bar by that one ticker's vol regardless
            # of the candidate being sized.
            candidate_volatility = None
            if config.use_volatility_scaling:
                cand_df = stock_data.get(signal.ticker, (None, None))[1]
                if cand_df is not None and len(cand_df) >= 21:
                    candidate_volatility = calculate_realized_volatility(cand_df["close"])

            result = evaluate(
                signal,
                portfolio.positions,
                mtm_value,
                mtm_daily_pnl,
                current_price=fill_price,
                volatility=candidate_volatility,
                start_of_day_equity=start_of_day_equity,
                # Backtests don't fetch live analyst consensus (no yfinance
                # recommendations_summary / IBKR Reuters Fundamentals call per
                # candidate per bar). Pass synthetic 'buy' on both sources so
                # the consensus check is a no-op in backtests; the live
                # scheduler still gates on the real two-source agreement.
                analyst_consensus="buy",
                analyst_consensus_ibkr="buy",
            )
            if not result.approved:
                continue

            portfolio.open_position(signal, result.position_size, fill_price, current_dt)

        portfolio.record_equity(current_date, current_prices=eod_prices)

    # Close remaining positions at last available price
    for pos in list(portfolio.positions):
        if pos.ticker in day_data:
            last_price = day_data[pos.ticker]["close"]
        else:
            last_price = pos.entry_price
        portfolio.close_position(pos.ticker, last_price, current_dt)

    # Attach benchmark artifacts. benchmark_prices is the RAW, full-history,
    # un-trimmed, un-normalized close (for 02-06's regime gate); benchmark_curve
    # is the buy-and-hold equity curve aligned to the strategy's warmup-trimmed
    # trading days (for the CAPM alpha/beta report).
    if benchmark_df is not None and not benchmark_df.empty:
        portfolio.benchmark_prices = benchmark_df["close"]
    equity_dates = [d for d, _ in portfolio.equity_curve]
    portfolio.benchmark_curve = _build_benchmark_curve(
        benchmark_df, equity_dates, config.initial_capital,
    )

    logger.info(
        "Backtest complete: %d trades, final value=$%,.2f (%.1f%% return)",
        len(portfolio.trades),
        portfolio.portfolio_value,
        (portfolio.portfolio_value / config.initial_capital - 1) * 100,
    )

    return portfolio


def run_strategy_with_controls(config: BacktestConfig) -> list[tuple[str, dict]]:
    """Run the strategy + its two controls and return compare_configs columns.

    Produces three labeled (name, metrics) tuples ready to hand straight to
    ``compare_configs``:

      - "Strategy"     — the real screener-driven run, with CAPM alpha/beta vs SPY.
      - "Random-Entry" — a deterministic Bernoulli(p=0.5) coin-flip control with
                         IDENTICAL sizing/exits/costs (only the entry is random).
      - "SPY"          — the passive buy-and-hold benchmark column.

    The random control will have a DIFFERENT trade count / market exposure than
    the screener (it enters roughly half the universe each bar). That difference
    is expected and acceptable — this is the survival-vs-edge control, not a
    like-for-like trade-count match. The strategy only has edge if it beats BOTH
    its random-entry control and SPY net of costs.
    """
    from backtest.report import calculate_metrics, benchmark_column_metrics

    strat = run_backtest(config)
    rand = run_backtest(replace(config, use_random_entry=True))

    def _metrics(portfolio: SimulatedPortfolio) -> dict:
        return calculate_metrics(
            portfolio.trades, portfolio.equity_curve, config.initial_capital,
            slippage_pct=config.slippage_pct, spread_bps=config.spread_bps,
            commission=config.commission,
            commission_per_share=config.commission_per_share,
            benchmark_curve=portfolio.benchmark_curve,
        )

    return [
        ("Strategy", _metrics(strat)),
        ("Random-Entry", _metrics(rand)),
        ("SPY", benchmark_column_metrics(strat.benchmark_curve, config.initial_capital)),
    ]


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


# ---------------------------------------------------------------------------
# Multi-fold rolling walk-forward
# ---------------------------------------------------------------------------

# Window sizes are expressed in TRADING days (the natural unit for "~250 trading
# days ≈ 12 months" and "2y IS"), but run_backtest slices by CALENDAR dates, so
# the harness converts trading-day counts to a calendar span with this ratio.
# 252 trading days ≈ 365 calendar days, so oos_days=252 → ~12 calendar months —
# long enough that, after each fold's fixed 60-bar warmup is consumed, the OOS
# window can still accumulate the >=30-trade statistical floor.
_TRADING_DAYS_PER_YEAR = 252
_CALENDAR_DAYS_PER_YEAR = 365

# Defaults: 2y in-sample, ~12-month out-of-sample, stepped by the OOS length so
# the OOS windows are ADJACENT and NON-OVERLAPPING (pooling their trades does not
# double-count any bar). Short OOS windows (e.g. 6 months / ~125 bars) are
# deliberately avoided: the 60-bar warmup would leave only ~65 tradable bars,
# starving the >=30-trade gate.
DEFAULT_WF_IS_DAYS = 2 * _TRADING_DAYS_PER_YEAR   # ~2 years in-sample (504)
DEFAULT_WF_OOS_DAYS = _TRADING_DAYS_PER_YEAR      # ~12 months OOS (252, >= 9-12mo)
DEFAULT_WF_STEP_DAYS = _TRADING_DAYS_PER_YEAR     # step by one OOS window (252)


def _trading_to_calendar_days(trading_days: int) -> int:
    """Convert a trading-day count to an approximate calendar-day span."""
    return round(trading_days * _CALENDAR_DAYS_PER_YEAR / _TRADING_DAYS_PER_YEAR)


def _walk_forward_efficiency(is_metrics: dict, oos_metrics: dict) -> Optional[float]:
    """WFE = annualized_OOS_return / annualized_IS_return (STACK.md:22).

    Returns None when WFE is UNDEFINED: a non-positive in-sample annualized
    return makes the ratio meaningless (divide-by-zero or a sign flip that would
    make a losing OOS look "efficient" against a losing IS). The report renders
    None explicitly rather than fabricating a number.
    """
    is_ann = is_metrics.get("annualized_return_pct", 0.0)
    oos_ann = oos_metrics.get("annualized_return_pct", 0.0)
    if not isinstance(is_ann, (int, float)) or is_ann <= 0:
        return None
    return oos_ann / is_ann


def _chain_oos_equity(
    fold_curves: list[list[tuple[date, float]]],
    initial_capital: float,
) -> list[tuple[date, float]]:
    """Stitch per-fold OOS equity curves into one continuous compounded curve.

    Each fold's OOS backtest starts fresh at initial_capital, so naively
    concatenating the curves would inject artificial jumps back to the starting
    equity. Instead we compound: every fold is rebased onto the running ending
    equity of the previous fold, preserving each fold's internal return shape
    while producing a single monotonically-dated aggregate curve for Sharpe /
    drawdown over the pooled OOS period.
    """
    chained: list[tuple[date, float]] = []
    running = float(initial_capital)
    for curve in fold_curves:
        if not curve:
            continue
        base = curve[0][1]
        if base == 0:
            continue
        for d, v in curve:
            chained.append((d, running * (v / base)))
        running = chained[-1][1]
    return chained


@dataclass
class WalkForwardFold:
    """One IS→OOS fold of a rolling walk-forward."""
    index: int
    in_sample_start: date
    in_sample_end: date
    out_of_sample_start: date
    out_of_sample_end: date
    in_sample_metrics: dict
    out_of_sample_metrics: dict
    degradation: dict
    wfe: Optional[float]
    in_sample_portfolio: SimulatedPortfolio
    out_of_sample_portfolio: SimulatedPortfolio


@dataclass
class RollingWalkForwardResult:
    """Aggregate result of a multi-fold rolling walk-forward.

    Carries every per-fold result, the pooled out-of-sample trades, an aggregate
    OOS metrics dict computed over a compounded chain of the folds' OOS equity
    curves, and the aggregate WFE (mean of the well-defined per-fold WFEs).
    """
    folds: list[WalkForwardFold]
    aggregate_oos_trades: list[Trade]
    aggregate_oos_equity: list[tuple[date, float]]
    aggregate_oos_metrics: dict
    aggregate_wfe: Optional[float]
    is_days: int
    oos_days: int
    step_days: int


def rolling_walk_forward(
    config: BacktestConfig,
    is_days: int = DEFAULT_WF_IS_DAYS,
    oos_days: int = DEFAULT_WF_OOS_DAYS,
    step_days: int = DEFAULT_WF_STEP_DAYS,
) -> RollingWalkForwardResult:
    """Slide a fixed IS window + adjacent OOS window across the full date range.

    Each fold runs run_backtest on its IS slice and its (immediately following)
    OOS slice with fresh capital, then computes IS/OOS metrics, the per-metric
    IS→OOS degradation, and WFE = annualized_OOS_return / annualized_IS_return.
    Multiple rolling folds give several independent OOS windows across regimes —
    a far harder bar to clear than a single 60/40 split.

    Window sizes are in TRADING days (defaults: ~2y IS, ~12-month OOS, stepped by
    the OOS length so OOS windows are adjacent and non-overlapping). They are
    converted to calendar spans for run_backtest's date filters. The ~12-month
    OOS default is deliberate: each fold's fixed 60-bar warmup is consumed before
    any trade, so a shorter OOS would starve the >=30-trade statistical floor.

    Args:
        config: BacktestConfig with both start_date and end_date set.
        is_days: in-sample length in trading days.
        oos_days: out-of-sample length in trading days.
        step_days: trading days to advance the window between folds.

    Raises:
        ValueError: if dates are missing/inverted, any window is non-positive, or
            the range is too short to fit even one IS+OOS fold.
    """
    if not config.start_date or not config.end_date:
        raise ValueError(
            "rolling_walk_forward requires both config.start_date and config.end_date"
        )
    if is_days <= 0 or oos_days <= 0 or step_days <= 0:
        raise ValueError("is_days, oos_days and step_days must all be positive")

    start = date.fromisoformat(config.start_date)
    end = date.fromisoformat(config.end_date)
    if end <= start:
        raise ValueError(f"end_date ({end}) must be after start_date ({start})")

    is_cal = _trading_to_calendar_days(is_days)
    oos_cal = _trading_to_calendar_days(oos_days)
    step_cal = _trading_to_calendar_days(step_days)

    from backtest.report import calculate_metrics

    folds: list[WalkForwardFold] = []
    fold_start = start
    index = 0
    # A fold fits iff its OOS window ends on or before the requested end date.
    while fold_start + timedelta(days=is_cal + oos_cal - 1) <= end:
        is_start = fold_start
        is_end = is_start + timedelta(days=is_cal - 1)
        oos_start = is_end + timedelta(days=1)
        oos_end = oos_start + timedelta(days=oos_cal - 1)

        is_config = replace(
            config, start_date=is_start.isoformat(), end_date=is_end.isoformat(),
        )
        oos_config = replace(
            config, start_date=oos_start.isoformat(), end_date=oos_end.isoformat(),
        )

        logger.info(
            "WF fold %d: IS %s → %s, OOS %s → %s",
            index, is_start, is_end, oos_start, oos_end,
        )

        is_portfolio = run_backtest(is_config)
        oos_portfolio = run_backtest(oos_config)

        is_metrics = calculate_metrics(
            is_portfolio.trades, is_portfolio.equity_curve, is_config.initial_capital,
            benchmark_curve=is_portfolio.benchmark_curve,
        )
        oos_metrics = calculate_metrics(
            oos_portfolio.trades, oos_portfolio.equity_curve, oos_config.initial_capital,
            benchmark_curve=oos_portfolio.benchmark_curve,
        )

        degradation: dict = {}
        for key, is_val in is_metrics.items():
            oos_val = oos_metrics.get(key)
            if isinstance(is_val, (int, float)) and isinstance(oos_val, (int, float)):
                degradation[key] = oos_val - is_val

        folds.append(WalkForwardFold(
            index=index,
            in_sample_start=is_start,
            in_sample_end=is_end,
            out_of_sample_start=oos_start,
            out_of_sample_end=oos_end,
            in_sample_metrics=is_metrics,
            out_of_sample_metrics=oos_metrics,
            degradation=degradation,
            wfe=_walk_forward_efficiency(is_metrics, oos_metrics),
            in_sample_portfolio=is_portfolio,
            out_of_sample_portfolio=oos_portfolio,
        ))

        index += 1
        fold_start = fold_start + timedelta(days=step_cal)

    if not folds:
        raise ValueError(
            f"Date range [{start} .. {end}] is too short for one IS+OOS fold "
            f"(needs ~{is_cal + oos_cal} calendar days for is_days={is_days}, "
            f"oos_days={oos_days})."
        )

    # Pool the OOS trades and compound the OOS equity curves into a single
    # continuous series so the aggregate Sharpe/drawdown reflect the whole
    # out-of-sample experience, not one cherry-picked fold.
    pooled_trades: list[Trade] = []
    for fold in folds:
        pooled_trades.extend(fold.out_of_sample_portfolio.trades)

    aggregate_equity = _chain_oos_equity(
        [f.out_of_sample_portfolio.equity_curve for f in folds],
        config.initial_capital,
    )
    aggregate_oos_metrics = calculate_metrics(
        pooled_trades, aggregate_equity, config.initial_capital,
    )

    valid_wfes = [f.wfe for f in folds if f.wfe is not None]
    aggregate_wfe = sum(valid_wfes) / len(valid_wfes) if valid_wfes else None

    return RollingWalkForwardResult(
        folds=folds,
        aggregate_oos_trades=pooled_trades,
        aggregate_oos_equity=aggregate_equity,
        aggregate_oos_metrics=aggregate_oos_metrics,
        aggregate_wfe=aggregate_wfe,
        is_days=is_days,
        oos_days=oos_days,
        step_days=step_days,
    )

"""Main orchestration loop — runs the trading pipeline on schedule."""

import logging
import signal as sig
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from ib_insync import IB

from config.settings import (
    SCAN_INTERVAL_MINUTES, TIMEZONE, MARKET_HOURS,
    CLOSE_DAY_TRADES_BEFORE_MARKET_CLOSE, CLOSE_MINUTES_BEFORE,
)
from core.connection import ensure_connected, create_contract, disconnect
from core.data import get_historical_data, get_news, get_historical_data_yfinance
from core.universe import build_universe, get_tickers_for_market
from core.screener import screen_stocks
from core.analyst import analyze_batch
from core.risk import evaluate
from core.executor import (
    place_order, close_all_day_trades, handle_fill,
    setup_disconnect_handler,
)
from core.portfolio import (
    get_open_positions, get_daily_pnl, record_signal,
    get_portfolio_value,
)
from core.models import StockInfo

logger = logging.getLogger(__name__)

_tz = ZoneInfo(TIMEZONE)
_shutting_down = False


# ---------------------------------------------------------------------------
# Market hours
# ---------------------------------------------------------------------------

def is_market_open(market: str) -> bool:
    """Check if a market is currently open based on Istanbul time."""
    now = datetime.now(_tz)
    hours = MARKET_HOURS.get(market.upper())
    if not hours:
        return False

    open_h, open_m = map(int, hours["open"].split(":"))
    close_h, close_m = map(int, hours["close"].split(":"))

    market_open = now.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
    market_close = now.replace(hour=close_h, minute=close_m, second=0, microsecond=0)

    # Weekends
    if now.weekday() >= 5:
        return False

    return market_open <= now <= market_close


def get_active_markets(markets: list[str]) -> list[str]:
    """Return list of markets that are currently open."""
    return [m for m in markets if is_market_open(m)]


def minutes_to_close(market: str) -> int:
    """Minutes remaining until market close."""
    now = datetime.now(_tz)
    hours = MARKET_HOURS.get(market.upper())
    if not hours:
        return 999

    close_h, close_m = map(int, hours["close"].split(":"))
    market_close = now.replace(hour=close_h, minute=close_m, second=0, microsecond=0)
    delta = (market_close - now).total_seconds() / 60
    return max(int(delta), 0)


# ---------------------------------------------------------------------------
# Scan cycle — the core trading pipeline
# ---------------------------------------------------------------------------

def run_scan_cycle(
    ib: IB,
    markets: list[str],
    mode: str = "paper",
) -> dict:
    """Run one full scan cycle across all active markets.

    Returns a summary dict with counts of actions taken.
    """
    summary = {
        "timestamp": datetime.now(_tz).isoformat(),
        "markets_scanned": [],
        "candidates_found": 0,
        "ai_approved": 0,
        "risk_approved": 0,
        "orders_placed": 0,
    }

    # Ensure connection
    try:
        ensure_connected(ib)
    except ConnectionError:
        logger.error("Cannot run scan — IBKR not connected")
        return summary

    active_markets = get_active_markets(markets)
    if not active_markets:
        logger.info("No markets currently open")
        return summary

    # Get account info
    from core.connection import get_account_summary
    account = get_account_summary(ib)
    portfolio_value = account.get("NetLiquidation", 0)
    daily_pnl = get_daily_pnl()
    open_positions = get_open_positions()

    # Build universe (cached daily)
    universe = build_universe(ib, active_markets)

    for market in active_markets:
        logger.info("=== Scanning %s market ===", market)
        summary["markets_scanned"].append(market)

        market_stocks = get_tickers_for_market(universe, market)

        # Check for end-of-day close
        mins_left = minutes_to_close(market)
        if CLOSE_DAY_TRADES_BEFORE_MARKET_CLOSE and mins_left <= CLOSE_MINUTES_BEFORE:
            logger.info(
                "%s market closing in %d min — closing day trades", market, mins_left,
            )
            dry_run = mode == "dry-run"
            close_all_day_trades(ib, open_positions, dry_run=dry_run)
            continue

        # Step 1: Fetch data for all stocks
        stock_data = _fetch_market_data(ib, market_stocks)

        # Step 2: Run screener
        candidates = screen_stocks(stock_data)
        summary["candidates_found"] += len(candidates)
        logger.info("Screener found %d candidates for %s", len(candidates), market)

        if not candidates:
            continue

        # Step 3: AI analysis
        ai_input = []
        for sig_obj in candidates:
            ticker = sig_obj.ticker
            exchange = sig_obj.exchange
            df = stock_data.get(ticker, (None, None))[1]
            if df is None or df.empty:
                continue

            news = get_news(ticker, market)
            ai_input.append({
                "ticker": ticker,
                "exchange": exchange,
                "df": df,
                "indicator_values": sig_obj.indicator_values,
                "news": news,
            })

        ai_signals = analyze_batch(ai_input)
        summary["ai_approved"] += len(ai_signals)

        # Step 4: Risk check + execution
        for signal in ai_signals:
            record_signal(signal)

            result = evaluate(signal, open_positions, portfolio_value, daily_pnl)
            if not result.approved:
                logger.info("Risk rejected %s: %s", signal.ticker, "; ".join(result.reasons))
                continue

            summary["risk_approved"] += 1

            # Place order
            dry_run = mode in ("dry-run", "backtest")
            trades = place_order(ib, signal, result.position_size, dry_run=dry_run)

            if trades:
                summary["orders_placed"] += 1
                handle_fill(signal, result.position_size, signal.entry_price)
                # Refresh positions for subsequent risk checks
                open_positions = get_open_positions()

    logger.info(
        "Scan complete: %d candidates, %d AI approved, %d risk approved, %d orders",
        summary["candidates_found"], summary["ai_approved"],
        summary["risk_approved"], summary["orders_placed"],
    )
    return summary


def _fetch_market_data(
    ib: IB,
    stocks: list[StockInfo],
) -> dict[str, tuple[str, "pd.DataFrame"]]:
    """Fetch historical data for all stocks in the market.

    Returns dict mapping ticker -> (exchange, DataFrame).
    """
    import pandas as pd

    stock_data: dict[str, tuple[str, pd.DataFrame]] = {}

    for stock in stocks:
        try:
            contract = create_contract(stock.ticker, stock.exchange)
            df = get_historical_data(ib, contract, duration="60 D", bar_size="1 day")

            if df.empty:
                # Fallback to yfinance
                market = "US"
                df = get_historical_data_yfinance(stock.ticker, period="3mo", market=market)

            if not df.empty:
                stock_data[stock.ticker] = (stock.exchange, df)

        except Exception as e:
            logger.warning("Failed to fetch data for %s: %s", stock.ticker, e)

    logger.info("Fetched data for %d/%d stocks", len(stock_data), len(stocks))
    return stock_data


# ---------------------------------------------------------------------------
# Scheduler setup
# ---------------------------------------------------------------------------

def start_scheduler(
    ib: IB,
    markets: list[str],
    mode: str = "paper",
) -> None:
    """Start the scan loop using ib_insync's event loop.

    Uses ib.sleep() instead of APScheduler so all IBKR calls stay on the
    main thread's asyncio event loop (ib_insync requirement).
    """
    global _shutting_down

    setup_disconnect_handler(ib)

    # Graceful shutdown
    def shutdown(signum, frame):
        global _shutting_down
        _shutting_down = True
        logger.info("Received signal %s — shutting down...", signum)

    sig.signal(sig.SIGINT, shutdown)
    sig.signal(sig.SIGTERM, shutdown)

    logger.info(
        "Scheduler started: scanning every %d min for markets %s",
        SCAN_INTERVAL_MINUTES, markets,
    )

    try:
        while not _shutting_down:
            run_scan_cycle(ib, markets, mode)
            if not _shutting_down:
                ib.sleep(SCAN_INTERVAL_MINUTES * 60)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        _shutting_down = True
        logger.info("Scheduler stopped")
        # Close day trades before exit
        try:
            positions = get_open_positions()
            dry_run = mode == "dry-run"
            close_all_day_trades(ib, positions, dry_run=dry_run)
        except Exception:
            pass
        disconnect(ib)

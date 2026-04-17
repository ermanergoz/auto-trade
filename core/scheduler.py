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
    AI_MAX_CANDIDATES, STALE_ORDER_MINUTES,
)
from core.connection import ensure_connected, create_contract, disconnect
from core.data import get_historical_data, get_news, get_historical_data_yfinance, get_macro_news
from core.universe import build_universe, get_tickers_for_market
from core.screener import screen_stocks
from core.analyst import analyze_batch
from core.risk import evaluate
from core.executor import (
    place_order, close_all_day_trades, setup_fill_handler,
    setup_exit_handler, setup_disconnect_handler,
    get_stale_orders, cancel_bracket_order,
)
from core.portfolio import (
    get_open_positions, get_daily_pnl, record_signal,
    get_portfolio_value, get_trades,
)
from core.models import StockInfo
from notifications.telegram import (
    notify_scan_summary, notify_trade, notify_error,
    notify_shutdown, update_status, notify_risk_results,
    update_portfolio_data, notify_risk_warning,
    notify_stale_order_cancelled,
)

from core import state as _state

logger = logging.getLogger(__name__)

_tz = ZoneInfo(TIMEZONE)


# ---------------------------------------------------------------------------
# Market hours
# ---------------------------------------------------------------------------

def is_market_open(market: str) -> bool:
    """Check if a market is currently open based on Istanbul time."""
    now = datetime.now(_tz)

    # Weekends — check before parsing to avoid errors on misconfigured hours
    if now.weekday() >= 5:
        return False

    hours = MARKET_HOURS.get(market.upper())
    if not hours:
        return False

    open_h, open_m = map(int, hours["open"].split(":"))
    close_h, close_m = map(int, hours["close"].split(":"))

    market_open = now.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
    market_close = now.replace(hour=close_h, minute=close_m, second=0, microsecond=0)

    return market_open <= now < market_close


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
    force: bool = False,
) -> dict:
    """Run one full scan cycle across all active markets.

    Args:
        force: If True, bypass market hours check (orders queue for next open).

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
        notify_error("Cannot run scan — IBKR not connected")
        return summary

    # Re-evaluate stale unfilled orders before scanning
    stale_result = check_stale_orders(ib, mode)
    if stale_result["cancelled"] > 0:
        logger.info(
            "Cancelled %d stale orders: %s",
            stale_result["cancelled"], stale_result["tickers_cancelled"],
        )

    if force:
        active_markets = [m.upper() for m in markets]
        logger.info("Force mode — bypassing market hours check")
    else:
        active_markets = get_active_markets(markets)
        if not active_markets:
            logger.info("No markets currently open")
            update_status("waiting", "No markets currently open")
            return summary

    # Get account info
    from core.connection import get_account_summary
    account = get_account_summary(ib)
    portfolio_value = account.get("NetLiquidation", 0)
    daily_pnl = get_daily_pnl(unrealized_pnl=account.get("UnrealizedPnL", 0.0))
    open_positions = get_open_positions()
    update_portfolio_data(account, open_positions, daily_pnl)

    # Fetch macro/political headlines once for all candidates
    macro_news = get_macro_news()
    if macro_news:
        logger.info("Macro headlines: %d items fetched", len(macro_news))

    # Build universe (cached daily)
    update_status("building_universe", f"Markets: {', '.join(active_markets)}")
    universe = build_universe(ib, active_markets)

    for market in active_markets:
        logger.info("=== Scanning %s market ===", market)
        summary["markets_scanned"].append(market)

        market_stocks = get_tickers_for_market(universe, market)

        # Check for end-of-day close (skip when force mode)
        mins_left = minutes_to_close(market)
        if not force and CLOSE_DAY_TRADES_BEFORE_MARKET_CLOSE and 0 < mins_left <= CLOSE_MINUTES_BEFORE:
            logger.info(
                "%s market closing in %d min — closing day trades", market, mins_left,
            )
            dry_run = mode == "dry-run"
            close_all_day_trades(ib, open_positions, dry_run=dry_run)
            continue

        # Step 1: Fetch data for all stocks
        if not ib.isConnected():
            logger.error("IBKR disconnected after universe build — aborting scan for %s", market)
            notify_error("Scan aborted: IBKR disconnected after universe build")
            return summary
        update_status("fetching_data", f"{len(market_stocks)} stocks for {market}")
        stock_data = _fetch_market_data(ib, market_stocks)
        sector_lookup = {s.ticker: s.sector for s in market_stocks}

        if not ib.isConnected():
            logger.error(
                "IBKR disconnected during data fetch — aborting scan for %s "
                "(got %d/%d stocks)", market, len(stock_data), len(market_stocks),
            )
            notify_error(
                f"Scan aborted: IBKR disconnected during data fetch "
                f"({len(stock_data)}/{len(market_stocks)} stocks fetched)"
            )
            return summary

        # Step 2: Run screener
        update_status("screening", f"{len(stock_data)} stocks")
        candidates = screen_stocks(stock_data)
        for cand in candidates:
            cand.indicator_values["sector"] = sector_lookup.get(cand.ticker, "")
        summary["candidates_found"] += len(candidates)
        logger.info("Screener found %d candidates for %s", len(candidates), market)

        if not candidates:
            continue

        # Limit candidates sent to AI (they're already sorted by score descending)
        if AI_MAX_CANDIDATES > 0 and len(candidates) > AI_MAX_CANDIDATES:
            logger.info(
                "Capping AI analysis to top %d candidates (of %d)",
                AI_MAX_CANDIDATES, len(candidates),
            )
            candidates = candidates[:AI_MAX_CANDIDATES]

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

        update_status("ai_analysis", f"0/{len(ai_input)} candidates for {market}")

        def _on_ai_progress(current, total, _market=market):
            update_status("ai_analysis", f"{current}/{total} candidates for {_market}")

        # Step 3+4: AI analysis with streaming risk check + execution.
        # Each AI-approved signal is immediately sent to risk check and
        # order placement via the on_signal callback, instead of waiting
        # for all AI analysis to complete first.
        risk_approved_signals = []

        def _on_signal(signal):
            nonlocal open_positions, portfolio_value
            summary["ai_approved"] += 1
            record_signal(signal)

            # Refresh account state to capture any fills since scan start
            fresh_account = get_account_summary(ib)
            portfolio_value = fresh_account.get("NetLiquidation", portfolio_value)
            fresh_daily_pnl = get_daily_pnl(unrealized_pnl=fresh_account.get("UnrealizedPnL", 0.0))

            # Fetch current price for anti-momentum check
            sig_df = stock_data.get(signal.ticker, (None, None))[1]
            current_price = sig_df["close"].iloc[-1] if sig_df is not None and not sig_df.empty else 0.0
            if current_price == 0.0:
                logger.warning(
                    "Current price unavailable for %s — anti-momentum check will use entry_price",
                    signal.ticker,
                )
            # Use UTC date for trade lookup — exit_time is stored as UTC ISO string,
            # so SQLite's date() extracts UTC calendar date. Using Istanbul date
            # would miss trades closed before 03:00 Istanbul time.
            from datetime import timezone as _utc_tz
            recent_trades = get_trades(start_date=datetime.now(_utc_tz.utc).date())

            # Fetch analyst consensus to block buys on sell-rated stocks
            from core.data import get_analyst_recommendation
            analyst_data = get_analyst_recommendation(signal.ticker)
            analyst_consensus = analyst_data["consensus"] if analyst_data else None

            result = evaluate(signal, open_positions, portfolio_value, fresh_daily_pnl, current_price=current_price, recent_trades=recent_trades, analyst_consensus=analyst_consensus)
            if not result.approved:
                logger.info("Risk rejected %s: %s", signal.ticker, "; ".join(result.reasons))
                if any("circuit breaker" in r.lower() for r in result.reasons):
                    notify_risk_warning(
                        "Circuit breaker tripped — consecutive losses detected. "
                        "Trading paused. Review manually."
                    )
                return

            summary["risk_approved"] += 1
            risk_approved_signals.append(signal)

            dry_run = mode in ("dry-run", "backtest")

            def _on_fill(sig, filled_qty, fill_price):
                notify_trade(sig, filled_qty)
                logger.info(
                    "Order filled: %s %d @ $%.2f",
                    sig.ticker, filled_qty, fill_price,
                )

            def _on_exit(ticker, exit_price, exit_type):
                from notifications.telegram import notify_trade_closed, refresh_positions_cache
                from core.portfolio import get_trades
                trades_list = get_trades(ticker=ticker)
                if trades_list:
                    notify_trade_closed(trades_list[0])
                refresh_positions_cache()

            trades = place_order(ib, signal, result.position_size, dry_run=dry_run)

            if trades:
                # Register handlers BEFORE the rejection check sleep.
                # This closes a race window where a fill could fire between
                # place_order() returning and handler registration. Handlers
                # only act on "Filled" status, so they're no-ops for rejected
                # orders (which show "Inactive"/"Cancelled").
                setup_fill_handler(
                    ib, signal, result.position_size,
                    on_fill=_on_fill, parent_order=trades[0].order,
                )
                setup_exit_handler(
                    ib, signal, on_exit=_on_exit, parent_order=trades[0].order,
                )

                # Give IBKR time to accept or reject the order before
                # checking status — rejections (e.g. insufficient funds)
                # arrive asynchronously within ~500ms.
                ib.sleep(0.5)
                parent_status = trades[0].orderStatus.status
                if parent_status in ("Inactive", "Cancelled"):
                    # Extract rejection reason from trade log if available
                    reason = ""
                    for log_entry in reversed(trades[0].log):
                        if log_entry.message and "Order rejected" in log_entry.message:
                            reason = log_entry.message
                            break
                    logger.warning(
                        "Order for %s was rejected by IBKR (status=%s): %s",
                        signal.ticker, parent_status, reason or "unknown reason",
                    )
                    notify_error(
                        f"Order rejected by IBKR: {signal.ticker} "
                        f"{result.position_size} shares @ ${signal.entry_price:.2f}\n"
                        f"Reason: {reason or 'unknown'}"
                    )
                else:
                    summary["orders_placed"] += 1
                    notify_trade(signal, result.position_size, action_type="SUBMITTED")

            # Always refresh positions for next risk check, regardless of
            # whether the order succeeded — async fills may have updated DB
            ib.sleep(0.5)
            open_positions = get_open_positions()

        analyze_batch(ai_input, on_signal=_on_signal, on_progress=_on_ai_progress, macro_news=macro_news)

        if risk_approved_signals:
            notify_risk_results(risk_approved_signals)

    logger.info(
        "Scan complete: %d candidates, %d AI approved, %d risk approved, %d orders",
        summary["candidates_found"], summary["ai_approved"],
        summary["risk_approved"], summary["orders_placed"],
    )
    notify_scan_summary(
        summary["candidates_found"], summary["ai_approved"],
        summary["risk_approved"], summary["orders_placed"],
    )
    update_status("scan_complete", f"Next scan in {SCAN_INTERVAL_MINUTES} min")
    return summary


def check_stale_orders(ib: IB, mode: str = "paper") -> dict:
    """Re-screen unfilled orders older than STALE_ORDER_MINUTES.

    For each stale order, fetches fresh data and runs the screener.
    Orders that no longer pass screening are cancelled.
    """
    result = {"stale_found": 0, "kept": 0, "cancelled": 0, "tickers_cancelled": []}

    if STALE_ORDER_MINUTES <= 0:
        return result

    stale = get_stale_orders(ib, STALE_ORDER_MINUTES)
    result["stale_found"] = len(stale)

    if not stale:
        open_parents = sum(
            1 for t in ib.openTrades()
            if t.order.parentId == 0 and t.order.orderType == "LMT"
        )
        if open_parents:
            logger.info(
                "Stale order check: %d open parent orders, none older than %dh",
                open_parents, STALE_ORDER_MINUTES // 60,
            )
        return result

    logger.info("Found %d stale unfilled orders, re-screening...", len(stale))

    for entry in stale:
        trade = entry["trade"]
        ticker = entry["ticker"]
        exchange = entry["exchange"]
        age_hours = entry["age_minutes"] / 60

        # Fetch fresh data and re-screen
        try:
            contract = create_contract(ticker, exchange)
            df = get_historical_data(ib, contract, duration="60 D", bar_size="1 day")
            if df.empty:
                df = get_historical_data_yfinance(ticker, period="3mo", market="US")

            still_valid = False
            if not df.empty:
                stock_data = {ticker: (exchange, df)}
                candidates = screen_stocks(stock_data)
                still_valid = any(c.ticker == ticker for c in candidates)
        except Exception as e:
            logger.warning("Failed to re-screen %s: %s — keeping order", ticker, e)
            result["kept"] += 1
            continue

        if still_valid:
            logger.info("Stale order for %s still passes screening (%.1fh old), keeping", ticker, age_hours)
            result["kept"] += 1
        else:
            dry_run = mode in ("dry-run", "backtest")
            if dry_run:
                logger.info("[DRY-RUN] Would cancel stale order for %s (%.1fh old)", ticker, age_hours)
                result["cancelled"] += 1
                result["tickers_cancelled"].append(ticker)
            else:
                cancel_bracket_order(ib, trade)
                notify_stale_order_cancelled(ticker, age_hours)
                result["cancelled"] += 1
                result["tickers_cancelled"].append(ticker)

    return result


def _fetch_market_data(
    ib: IB,
    stocks: list[StockInfo],
) -> dict[str, tuple[str, "pd.DataFrame"]]:
    """Fetch historical data for all stocks in the market.

    Returns dict mapping ticker -> (exchange, DataFrame).
    """
    import pandas as pd

    if not ib.isConnected():
        logger.error("IBKR not connected — skipping data fetch for %d stocks", len(stocks))
        return {}

    stock_data: dict[str, tuple[str, pd.DataFrame]] = {}

    for stock in stocks:
        if not ib.isConnected():
            logger.warning(
                "IBKR disconnected mid-fetch — got %d/%d stocks before disconnect",
                len(stock_data), len(stocks),
            )
            break

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
    force: bool = False,
    reconnect: bool = True,
) -> None:
    """Start the scan loop using ib_insync's event loop.

    Uses ib.sleep() instead of APScheduler so all IBKR calls stay on the
    main thread's asyncio event loop (ib_insync requirement).

    Args:
        reconnect: Passed to setup_disconnect_handler. Set False when the
                   Watchdog manages reconnection to avoid competing reconnects.
    """

    setup_disconnect_handler(ib, reconnect=reconnect)

    # Graceful shutdown
    def shutdown(signum, frame):
        _state.shutting_down = True
        logger.info("Received signal %s — shutting down...", signum)

    sig.signal(sig.SIGINT, shutdown)
    sig.signal(sig.SIGTERM, shutdown)

    logger.info(
        "Scheduler started: scanning every %d min for markets %s",
        SCAN_INTERVAL_MINUTES, markets,
    )

    watchdog_mode = not reconnect

    try:
        while not _state.shutting_down:
            if ib.isConnected():
                run_scan_cycle(ib, markets, mode, force=force)

            if _state.shutting_down:
                break

            # Sleep between scans, keeping the event loop alive.
            # When disconnected in watchdog mode, use shorter sleeps so we
            # resume scanning promptly once the watchdog reconnects.
            if ib.isConnected():
                try:
                    ib.sleep(SCAN_INTERVAL_MINUTES * 60)
                except Exception:
                    if _state.shutting_down:
                        break
                    if watchdog_mode:
                        logger.debug("ib.sleep() interrupted — will retry")
                    else:
                        raise
            elif watchdog_mode:
                import asyncio
                from ib_insync.util import run
                logger.info(
                    "IBKR disconnected — waiting for watchdog to reconnect..."
                )
                while not ib.isConnected() and not _state.shutting_down:
                    run(asyncio.sleep(5))
                if ib.isConnected():
                    logger.info("Watchdog reconnected — resuming scans")
            else:
                # Non-watchdog mode with no connection — give up
                logger.error("IBKR disconnected and no watchdog — exiting scheduler")
                break

    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        _state.shutting_down = True
        logger.info("Scheduler stopped")
        # Close day trades before exit
        try:
            positions = get_open_positions()
            dry_run = mode == "dry-run"
            close_all_day_trades(ib, positions, dry_run=dry_run)
        except Exception:
            pass
        notify_shutdown()
        if not watchdog_mode:
            disconnect(ib)

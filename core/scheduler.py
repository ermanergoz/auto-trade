"""Main orchestration loop — runs the trading pipeline on schedule."""

import logging
import signal as sig
import sys
from datetime import datetime, date
from typing import Optional
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
    place_order, place_market_order, close_all_day_trades, setup_fill_handler,
    setup_exit_handler, setup_disconnect_handler,
    get_stale_orders, cancel_bracket_order,
    setup_exit_close_handler,
    get_pending_buy_reserve, evict_weakest_pending,
)
from core.portfolio import (
    get_open_positions, get_daily_pnl, record_signal,
    get_portfolio_value, get_trades,
)
from core.models import StockInfo, Position, TradeType, Action
from notifications.telegram import (
    notify_scan_summary, notify_trade, notify_error,
    notify_shutdown, update_status, notify_risk_results,
    update_portfolio_data, notify_risk_warning,
    notify_stale_order_cancelled, notify_reconciliation_mismatch,
)

from core import state as _state

logger = logging.getLogger(__name__)

_tz = ZoneInfo(TIMEZONE)


# ---------------------------------------------------------------------------
# Market hours
# ---------------------------------------------------------------------------

def _market_tz(market: str) -> Optional[ZoneInfo]:
    """Return the market's native ZoneInfo, or None if market is unknown."""
    hours = MARKET_HOURS.get(market.upper())
    if not hours:
        return None
    return ZoneInfo(hours.get("tz", TIMEZONE))


def is_market_open(market: str) -> bool:
    """Check if a market is currently open.

    Evaluates using the market's native timezone (e.g. America/New_York for
    NYSE) so DST transitions are handled by ZoneInfo rather than the local
    display timezone.
    """
    hours = MARKET_HOURS.get(market.upper())
    if not hours:
        return False

    market_tz = ZoneInfo(hours.get("tz", TIMEZONE))
    now = datetime.now(market_tz)

    if now.weekday() >= 5:
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
    """Minutes remaining until market close (in the market's native tz)."""
    hours = MARKET_HOURS.get(market.upper())
    if not hours:
        return 999

    market_tz = ZoneInfo(hours.get("tz", TIMEZONE))
    now = datetime.now(market_tz)

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
    open_positions = get_open_positions()

    # Snapshot start-of-day equity on ET date rollover. The daily-loss-limit
    # cap must reference a stable baseline; using live MTM equity would let
    # the dollar threshold shrink as losses accumulate, tightening the brake
    # precisely when stability is needed most.
    _us_eastern = ZoneInfo("America/New_York")
    today_et = datetime.now(_us_eastern).date()
    if _state.start_of_day_date != today_et and portfolio_value > 0:
        _state.start_of_day_equity = portfolio_value
        _state.start_of_day_date = today_et
        logger.info(
            "Start-of-day equity snapshot (%s ET): $%.2f", today_et, portfolio_value,
        )
    start_of_day_equity = _state.start_of_day_equity

    # daily_pnl for the loss-limit check must be today's EQUITY CHANGE, not
    # realized_today + total-unrealized. IBKR's UnrealizedPnL is cumulative
    # across all positions — for swing positions opened days ago it includes
    # prior-day gains/losses, which would double-count historical P&L into
    # today's loss-limit comparison. Using the snapshot-delta is self
    # consistent with the start_of_day_equity baseline.
    if start_of_day_equity and start_of_day_equity > 0:
        daily_pnl = portfolio_value - start_of_day_equity
    else:
        # First scan before snapshot exists — fall back to the legacy
        # realized+unrealized sum. This is only relevant pre-snapshot.
        daily_pnl = get_daily_pnl(unrealized_pnl=account.get("UnrealizedPnL", 0.0))
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
            if not news:
                logger.info("Skipping %s: no news from Tavily or yfinance", ticker)
                continue
            # Pass the screener-built Signal through unchanged. The analyst
            # only votes (buy/hold) plus tags confidence/trade_type/reasoning;
            # entry_price/stop_loss/take_profit are taken from the screener's
            # deterministic ATR computation in core/screener.py:_build_signal,
            # not from the LLM. This blocks hallucinated chart-readings from
            # propagating into bracket-order levels.
            ai_input.append({
                "screener_signal": sig_obj,
                "df": df,
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
        # Virtual positions for bracket orders placed this cycle whose fills
        # haven't been written to the DB yet. Without this, two rapid AI
        # approvals both evaluate against the same stale DB snapshot and the
        # max-positions / sector-concentration / cumulative-risk checks both
        # pass when they should not.
        pending_this_cycle: list[Position] = []

        def _on_signal(signal):
            nonlocal open_positions, portfolio_value
            summary["ai_approved"] += 1
            record_signal(signal)

            # Refresh account state to capture any fills since scan start
            fresh_account = get_account_summary(ib)
            portfolio_value = fresh_account.get("NetLiquidation", portfolio_value)
            # Today's equity change (MTM), not realized+cumulative-unrealized.
            # See daily_pnl computation above for rationale.
            if start_of_day_equity and start_of_day_equity > 0:
                fresh_daily_pnl = portfolio_value - start_of_day_equity
            else:
                fresh_daily_pnl = get_daily_pnl(unrealized_pnl=fresh_account.get("UnrealizedPnL", 0.0))

            # Combine DB positions with in-flight virtual positions from this
            # scan cycle. Prefer DB when the same ticker appears in both (DB
            # reflects actual fill price / quantity).
            db_tickers = {p.ticker for p in open_positions}
            effective_positions = list(open_positions) + [
                p for p in pending_this_cycle if p.ticker not in db_tickers
            ]

            # Fetch current price for anti-momentum check
            sig_df = stock_data.get(signal.ticker, (None, None))[1]
            current_price = sig_df["close"].iloc[-1] if sig_df is not None and not sig_df.empty else 0.0
            if current_price == 0.0:
                logger.warning(
                    "Current price unavailable for %s — anti-momentum check will use entry_price",
                    signal.ticker,
                )
            # PDT rule counts day trades over a rolling 5-business-day window
            # (~7 calendar days). Query the DB with that full window so
            # check_pdt_restriction sees the true count, not just today's.
            # Using `start_date=today` would truncate the history to today,
            # silently disabling PDT protection on sub-$5k accounts and
            # risking IBKR's 30-day closing-only lockout.
            from datetime import timezone as _utc_tz, timedelta as _timedelta
            _pdt_window_start = (datetime.now(_utc_tz.utc) - _timedelta(days=7)).date()
            recent_trades = get_trades(start_date=_pdt_window_start)

            # Two-source analyst consensus gate: BUY only when BOTH yfinance
            # and IBKR (Reuters/Refinitiv) agree on buy/strong_buy. A None
            # from either source blocks the BUY at risk-eval time — see
            # check_analyst_consensus in core/risk.py.
            from core.data import (
                get_analyst_recommendation,
                get_analyst_recommendation_ibkr,
            )
            yf_data = get_analyst_recommendation(signal.ticker)
            analyst_consensus = yf_data["consensus"] if yf_data else None
            ibkr_data = get_analyst_recommendation_ibkr(ib, signal.ticker, signal.exchange)
            analyst_consensus_ibkr = ibkr_data["consensus"] if ibkr_data else None

            result = evaluate(
                signal, effective_positions, portfolio_value, fresh_daily_pnl,
                current_price=current_price, recent_trades=recent_trades,
                analyst_consensus=analyst_consensus,
                analyst_consensus_ibkr=analyst_consensus_ibkr,
                start_of_day_equity=start_of_day_equity,
            )
            if not result.approved:
                logger.info("Risk rejected %s: %s", signal.ticker, "; ".join(result.reasons))
                if any("circuit breaker" in r.lower() for r in result.reasons):
                    notify_risk_warning(
                        "Circuit breaker tripped — consecutive losses detected. "
                        "Trading paused. Review manually."
                    )
                return

            # Market-close boundary guard: wall-clock may have advanced into
            # the close window while AI analysis was running. If so, skip —
            # close_all_day_trades would immediately flatten this position,
            # creating an unnecessary round-trip at uncertain market price.
            if not force and CLOSE_DAY_TRADES_BEFORE_MARKET_CLOSE:
                mins_left_now = minutes_to_close(market)
                if 0 < mins_left_now <= CLOSE_MINUTES_BEFORE:
                    logger.info(
                        "Skipping %s: wall-clock entered close window "
                        "(%d min to close, threshold %d min)",
                        signal.ticker, mins_left_now, CLOSE_MINUTES_BEFORE,
                    )
                    return

            # Cash-reserve gate — only long entries consume settled cash.
            # IBKR's TotalCashValue is not decremented for unfilled parent
            # BUY orders, so we subtract that reserve ourselves. Without
            # this, two back-to-back approvals can over-commit and the
            # second bracket is rejected by IBKR (Error 201 — see 2026-04-22
            # HPE+STLD run). If short on cash, try to evict the weakest
            # pending BUY (only if the new one is clearly stronger).
            if signal.action == Action.BUY and not result.is_exit:
                total_cash = fresh_account.get("TotalCashValue", 0.0)
                pending_reserve = get_pending_buy_reserve(ib)
                needed_cash = signal.entry_price * result.position_size
                available_cash = total_cash - pending_reserve
                if needed_cash > available_cash:
                    evicted = evict_weakest_pending(
                        ib,
                        new_confidence=signal.confidence,
                        needed_cash=needed_cash,
                    )
                    if evicted:
                        logger.info(
                            "Evicted a weaker pending BUY to free cash for %s "
                            "(confidence %.0f)",
                            signal.ticker, signal.confidence,
                        )
                        # Refresh cash view after cancellation — the freed
                        # reserve should now cover this order.
                        ib.sleep(0.5)
                        pending_reserve = get_pending_buy_reserve(ib)
                        available_cash = total_cash - pending_reserve
                    if needed_cash > available_cash:
                        logger.info(
                            "Skipping %s: cash short "
                            "(need $%.2f, have $%.2f = $%.2f cash - $%.2f reserved)",
                            signal.ticker, needed_cash, available_cash,
                            total_cash, pending_reserve,
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

            # Route exits through a plain close order rather than a bracket.
            # A bracket's SL/TP children would stay live at IBKR after the
            # parent closes our position — those orders could later fire when
            # price crosses the stop or target level, opening a FRESH position
            # in the opposite direction. A simple market close has no such tail.
            if result.is_exit:
                action_str = "BUY" if signal.action == Action.BUY else "SELL"
                market_trade = place_market_order(
                    ib, signal.ticker, signal.exchange,
                    action_str, result.position_size, dry_run=dry_run,
                )
                if market_trade and not dry_run:
                    setup_exit_close_handler(
                        ib, signal, market_trade, on_exit=_on_exit,
                    )
                    summary["orders_placed"] += 1
                    notify_trade(signal, result.position_size, action_type="CLOSING")
                elif dry_run:
                    summary["orders_placed"] += 1
                ib.sleep(0.5)
                open_positions = get_open_positions()
                return

            trades = place_order(ib, signal, result.position_size, dry_run=dry_run)

            if trades:
                # Register handlers BEFORE the rejection check sleep.
                # This closes a race window where a fill could fire between
                # place_order() returning and handler registration. Handlers
                # only act on "Filled" status, so they're no-ops for rejected
                # orders (which show "Inactive"/"Cancelled").
                #
                # Pass parent_trade so the handler can detect a fast fill that
                # completed during place_order's permId poll — those would
                # otherwise never be replayed by ib_insync.
                setup_fill_handler(
                    ib, signal, result.position_size,
                    on_fill=_on_fill, parent_order=trades[0].order,
                    parent_trade=trades[0],
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
                    # Track as a virtual position so subsequent risk checks in
                    # this cycle see it, even if the DB refresh below hasn't
                    # picked up the fill yet. Fill handler will update DB
                    # asynchronously; the effective_positions merge handles
                    # the overlap without double-counting.
                    signed_qty = (
                        result.position_size
                        if signal.action == Action.BUY
                        else -result.position_size
                    )
                    # entry_time must be UTC. The PDT check converts naive
                    # datetimes to UTC by attaching a tzinfo=UTC (not by
                    # converting); a datetime in Istanbul time would be
                    # mis-bucketed by 3 hours relative to ET for the
                    # day-trade boundary classification.
                    from datetime import timezone as _utc_tz2
                    pending_this_cycle.append(Position(
                        ticker=signal.ticker,
                        exchange=signal.exchange,
                        quantity=signed_qty,
                        entry_price=signal.entry_price,
                        entry_time=datetime.now(_utc_tz2.utc),
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        trade_type=signal.trade_type,
                        sector=signal.indicator_values.get("sector", ""),
                    ))

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


def run_nightly_reconciliation(ib: IB, db_path=None) -> dict:
    """Read-only nightly check that DB positions match IBKR positions.

    Fetches ib.positions() + ib.fills() and calls reconcile_positions with
    auto_fix=False (report-only). Sends a Telegram alert on any mismatch.

    Reconnect-time reconciliation (in main.py) auto-fixes orphans because
    the bot was offline; this nightly check is purely for drift detection
    during normal operation and must never silently close positions.
    """
    from core.portfolio import reconcile_positions, DB_PATH
    if db_path is None:
        db_path = DB_PATH

    try:
        ibkr_positions = [
            {"ticker": p.contract.symbol, "quantity": int(p.position)}
            for p in ib.positions()
        ]
    except Exception as e:
        logger.error("Nightly reconcile: failed to fetch IBKR positions: %s", e)
        notify_error(f"Nightly reconciliation failed: {e}")
        return {"error": str(e), "in_sync": False}

    ibkr_fills: list[dict] = []
    try:
        for f in ib.fills():
            try:
                ibkr_fills.append({
                    "ticker": f.contract.symbol,
                    "side": f.execution.side,
                    "shares": float(f.execution.shares),
                    "price": float(f.execution.price),
                    "time": f.execution.time,
                })
            except Exception:
                continue
    except Exception as e:
        logger.debug("Nightly reconcile: could not fetch fills: %s", e)

    report = reconcile_positions(
        ibkr_positions, auto_fix=False,
        ibkr_fills=ibkr_fills, db_path=db_path,
    )

    if not report["in_sync"]:
        logger.warning(
            "Nightly reconcile mismatch: orphaned_db=%s orphaned_ibkr=%s qty_mismatches=%s",
            report["orphaned_db"], report["orphaned_ibkr"], report["qty_mismatches"],
        )
        notify_reconciliation_mismatch(report)
    else:
        logger.info(
            "Nightly reconcile OK: %d positions in sync", report["db_count"],
        )

    return report


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
        _state.shutting_down.set()
        logger.info("Received signal %s — shutting down...", signum)

    sig.signal(sig.SIGINT, shutdown)
    sig.signal(sig.SIGTERM, shutdown)

    logger.info(
        "Scheduler started: scanning every %d min for markets %s",
        SCAN_INTERVAL_MINUTES, markets,
    )

    watchdog_mode = not reconnect
    last_reconcile_date: Optional[date] = None

    try:
        while not _state.shutting_down.is_set():
            if ib.isConnected():
                run_scan_cycle(ib, markets, mode, force=force)

                # Nightly reconciliation — runs once per day after all markets
                # have closed. Separate from reconnect-time reconciliation
                # (which auto-fixes) — this is report-only drift detection.
                today = datetime.now(_tz).date()
                if last_reconcile_date != today and not get_active_markets(markets):
                    try:
                        run_nightly_reconciliation(ib)
                    except Exception as e:
                        logger.error("Nightly reconciliation failed: %s", e)
                    last_reconcile_date = today

            if _state.shutting_down.is_set():
                break

            # Sleep between scans, keeping the event loop alive.
            # When disconnected in watchdog mode, use shorter sleeps so we
            # resume scanning promptly once the watchdog reconnects.
            if ib.isConnected():
                try:
                    ib.sleep(SCAN_INTERVAL_MINUTES * 60)
                except Exception:
                    if _state.shutting_down.is_set():
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
                while not ib.isConnected() and not _state.shutting_down.is_set():
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
        _state.shutting_down.set()
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

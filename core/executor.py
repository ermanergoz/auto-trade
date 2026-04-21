"""IBKR order execution — bracket orders, monitoring, fill handling."""

import logging
import threading
from datetime import datetime, timezone
from typing import Optional

from ib_insync import IB, Trade as IBTrade, Order, LimitOrder, MarketOrder, StopOrder

from core.models import Signal, Position, Trade, Action, TradeType
from core.connection import create_contract, ensure_connected
from core.portfolio import (
    add_position, close_position as db_close_position,
    save_pending_order, get_pending_order_time, remove_pending_order,
)

logger = logging.getLogger(__name__)


def place_order(
    ib: IB,
    signal: Signal,
    quantity: int,
    dry_run: bool = False,
) -> Optional[list[IBTrade]]:
    """Place a bracket order (entry + stop-loss + take-profit).

    In dry-run mode, logs the order but doesn't execute.
    Returns list of ib_insync Trade objects, or None on failure.
    """
    contract = create_contract(signal.ticker, signal.exchange)

    # Qualify the contract
    try:
        ib.qualifyContracts(contract)
    except Exception as e:
        logger.error("Failed to qualify contract for %s: %s", signal.ticker, e)
        return None

    action = "BUY" if signal.action == Action.BUY else "SELL"
    reverse_action = "SELL" if signal.action == Action.BUY else "BUY"

    if dry_run:
        logger.info(
            "[DRY-RUN] Would place %s bracket order for %s: "
            "%d shares @ $%.2f (SL: $%.2f, TP: $%.2f)",
            action, signal.ticker, quantity,
            signal.entry_price, signal.stop_loss, signal.take_profit,
        )
        return None

    # Create bracket order with GTC so orders persist outside market hours
    bracket = ib.bracketOrder(
        action=action,
        quantity=quantity,
        limitPrice=round(signal.entry_price, 2),
        takeProfitPrice=round(signal.take_profit, 2),
        stopLossPrice=round(signal.stop_loss, 2),
    )

    parent_order, tp_order, sl_order = bracket

    # Day trades use DAY TIF so IBKR auto-expires them at session close,
    # preventing unattended overnight exposure. Swing trades use GTC to
    # survive overnight and fill at next market open.
    tif = "DAY" if signal.trade_type == TradeType.DAY else "GTC"
    for o in bracket:
        o.tif = tif
    parent_order.transmit = False
    tp_order.transmit = False
    sl_order.transmit = True

    # Place all three orders atomically (only the last triggers transmission).
    # If any order fails after the parent is placed, cancel already-placed
    # orders to avoid orphaned entries on IBKR.
    trades = []
    try:
        parent_trade = ib.placeOrder(contract, parent_order)
        trades.append(parent_trade)

        tp_trade = ib.placeOrder(contract, tp_order)
        trades.append(tp_trade)

        sl_trade = ib.placeOrder(contract, sl_order)
        trades.append(sl_trade)

        # Poll for permId AFTER all three orders are placed so the bracket
        # is fully submitted before we block. This avoids a window where
        # the parent exists on IBKR without its TP/SL children.
        for _ in range(6):
            ib.sleep(0.5)
            if parent_order.permId:
                break
        if parent_order.permId:
            save_pending_order(parent_order.permId, signal.ticker)
        else:
            logger.warning(
                "permId not assigned for %s order after 3s — "
                "stale detection will fall back to trade.log",
                signal.ticker,
            )

        logger.info(
            "Placed %s bracket order for %s: %d shares @ $%.2f "
            "(SL: $%.2f, TP: $%.2f) [OrderIDs: %s, %s, %s]",
            action, signal.ticker, quantity,
            signal.entry_price, signal.stop_loss, signal.take_profit,
            parent_order.orderId, tp_order.orderId, sl_order.orderId,
        )

        return trades

    except Exception as e:
        logger.error("Failed to place bracket order for %s: %s", signal.ticker, e)
        # Cancel any already-placed orders to avoid orphans
        for t in trades:
            try:
                ib.cancelOrder(t.order)
                logger.info("Cancelled orphaned order %s for %s", t.order.orderId, signal.ticker)
            except Exception as cancel_err:
                logger.error("Failed to cancel orphaned order: %s", cancel_err)
        if parent_order.permId:
            remove_pending_order(parent_order.permId)
        return None


def place_market_order(
    ib: IB,
    ticker: str,
    exchange: str,
    action: str,
    quantity: int,
    dry_run: bool = False,
) -> Optional[IBTrade]:
    """Place a simple market order (used for closing positions)."""
    contract = create_contract(ticker, exchange)

    try:
        ib.qualifyContracts(contract)
    except Exception as e:
        logger.error("Failed to qualify contract for %s: %s", ticker, e)
        return None

    if dry_run:
        logger.info("[DRY-RUN] Would place %s market order: %s %d shares", action, ticker, quantity)
        return None

    order = MarketOrder(action, quantity)

    try:
        trade = ib.placeOrder(contract, order)
        logger.info("Placed %s market order for %s: %d shares", action, ticker, quantity)
        return trade
    except Exception as e:
        logger.error("Failed to place market order for %s: %s", ticker, e)
        return None


def monitor_orders(ib: IB, trades: list[IBTrade], timeout: float = 30.0) -> list[dict]:
    """Poll placed orders until all are filled or timeout expires.

    Returns list of status dicts with final observed state.
    """
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ib.sleep(0.5)
        if all(t.orderStatus.status == "Filled" for t in trades):
            break

    statuses = []
    for trade in trades:
        statuses.append({
            "orderId": trade.order.orderId,
            "status": trade.orderStatus.status,
            "filled": trade.orderStatus.filled,
            "remaining": trade.orderStatus.remaining,
            "avgFillPrice": trade.orderStatus.avgFillPrice,
        })
    return statuses


def cancel_order(ib: IB, trade: IBTrade) -> None:
    """Cancel an unfilled order."""
    try:
        ib.cancelOrder(trade.order)
        logger.info("Cancelled order %s for %s", trade.order.orderId, trade.contract.symbol)
        if trade.order.permId:
            remove_pending_order(trade.order.permId)
    except Exception as e:
        logger.error("Failed to cancel order: %s", e)


def get_stale_orders(ib: IB, stale_minutes: int = 1440) -> list[dict]:
    """Return unfilled parent limit orders older than *stale_minutes*.

    Only considers parent entry orders (parentId == 0) with status
    Submitted or PreSubmitted.  Each returned dict contains the Trade
    object, ticker, exchange, and age in minutes.
    """
    stale = []
    now = datetime.now(timezone.utc)
    for trade in ib.openTrades():
        order = trade.order
        # Only parent entry limit orders
        if order.parentId != 0 or order.orderType != "LMT":
            continue
        status = trade.orderStatus.status
        if status not in ("Submitted", "PreSubmitted"):
            continue
        # Use persistent DB timestamp (survives reconnections)
        submitted_at = None
        ticker = trade.contract.symbol
        if order.permId:
            db_time = get_pending_order_time(order.permId)
            if db_time:
                submitted_at = db_time if db_time.tzinfo else db_time.replace(tzinfo=timezone.utc)
                logger.debug("Order %s (%s): using DB timestamp %s", order.permId, ticker, submitted_at)
        # Fallback to trade log (only accurate within same session)
        if submitted_at is None:
            if not trade.log:
                logger.debug("Order %s (%s): no DB record and no log — skipping", order.permId, ticker)
                continue
            log_time = trade.log[0].time
            if log_time.tzinfo is None:
                log_time = log_time.replace(tzinfo=timezone.utc)
            submitted_at = log_time
            logger.debug("Order %s (%s): no DB record, using log timestamp %s", order.permId, ticker, submitted_at)
        age_minutes = (now - submitted_at).total_seconds() / 60
        logger.info("Order %s (%s): age %.1fh, threshold %dh",
                     order.permId, ticker, age_minutes / 60, stale_minutes // 60)
        if age_minutes >= stale_minutes:
            stale.append({
                "trade": trade,
                "ticker": trade.contract.symbol,
                "exchange": trade.contract.exchange or trade.contract.primaryExchange,
                "age_minutes": age_minutes,
            })
    return stale


def cancel_bracket_order(ib: IB, trade: IBTrade) -> bool:
    """Cancel a parent entry order (IBKR auto-cancels attached TP/SL children)."""
    ticker = trade.contract.symbol
    order_id = trade.order.orderId
    try:
        ib.cancelOrder(trade.order)
        logger.info("Cancelled stale bracket order %s for %s", order_id, ticker)
        if trade.order.permId:
            remove_pending_order(trade.order.permId)
        return True
    except Exception as e:
        logger.error("Failed to cancel stale order %s for %s: %s", order_id, ticker, e)
        return False


def close_position_market(
    ib: IB,
    position: Position,
    dry_run: bool = False,
) -> Optional[IBTrade]:
    """Close an open position with a market order."""
    action = "BUY" if position.quantity < 0 else "SELL"
    return place_market_order(
        ib, position.ticker, position.exchange,
        action, abs(position.quantity), dry_run,
    )


def close_all_day_trades(
    ib: IB,
    positions: list[Position],
    dry_run: bool = False,
) -> list[IBTrade]:
    """Close all positions marked as day trades. Called before market close.

    Cancels any open bracket orders (TP/SL) for the closed positions to
    prevent orphaned orders at IBKR after the market close fills.
    Records the close in the portfolio database.
    """
    day_trades = [p for p in positions if p.trade_type == TradeType.DAY]

    if not day_trades:
        logger.info("No day trades to close")
        return []

    logger.info("Closing %d day trade positions", len(day_trades))

    # Build a set of tickers being closed for bracket order cancellation
    closing_tickers = {p.ticker for p in day_trades}

    # Cancel ALL open orders for the positions being closed BEFORE placing
    # market orders — prevents race where bracket SL/TP fills between
    # our market order placement and fill. This includes:
    # - Unfilled parent entry orders (cancelling parent auto-cancels children)
    # - Orphaned TP/SL children from already-filled parents
    if not dry_run:
        cancelled = 0
        for trade in ib.openTrades():
            if trade.contract.symbol in closing_tickers:
                try:
                    ib.cancelOrder(trade.order)
                    cancelled += 1
                    logger.info("Cancelled order %s (%s) for %s (day trade close)",
                                 trade.order.orderId, trade.order.orderType,
                                 trade.contract.symbol)
                except Exception as e:
                    logger.warning("Failed to cancel order for %s: %s",
                                   trade.contract.symbol, e)

        # IBKR cancels are asynchronous — the request returns immediately but
        # the order can still fill before the broker processes the cancel.
        # Poll openTrades() until the brackets clear (or timeout) so our
        # market close doesn't race a still-live SL/TP to a double-close.
        if cancelled:
            max_wait_sec = 3.0
            poll_interval = 0.25
            waited = 0.0
            while waited < max_wait_sec:
                ib.sleep(poll_interval)
                waited += poll_interval
                still_open = [
                    t for t in ib.openTrades()
                    if t.contract.symbol in closing_tickers
                    and t.orderStatus.status in ("Submitted", "PreSubmitted", "PendingCancel")
                ]
                if not still_open:
                    break
            if waited >= max_wait_sec and still_open:
                logger.warning(
                    "Timeout waiting for bracket cancels to clear for %s — "
                    "%d orders still open; proceeding with market close",
                    ", ".join(sorted({t.contract.symbol for t in still_open})),
                    len(still_open),
                )

    trades = []
    for pos in day_trades:
        trade = close_position_market(ib, pos, dry_run)
        if trade:
            trades.append((pos, trade))

    # Verify fills with extended timeout — market orders near close
    # may take longer than expected under adverse conditions
    if trades and not dry_run:
        ib_trades = [t for _, t in trades]
        statuses = monitor_orders(ib, ib_trades, timeout=30.0)
        for (pos, ib_trade), status in zip(trades, statuses):
            if status["status"] == "Filled":
                fill_price = status["avgFillPrice"]
                db_close_position(pos.ticker, fill_price)
                logger.info("Day trade closed: %s @ $%.2f", pos.ticker, fill_price)
            else:
                logger.error(
                    "CRITICAL: Day trade close for %s NOT filled after 30s "
                    "(status=%s) — position may remain open overnight!",
                    pos.ticker, status["status"],
                )

    return [t for _, t in trades]


def handle_fill(
    signal: Signal,
    quantity: int,
    fill_price: float,
    db_path=None,
) -> Optional[Position]:
    """Record a filled order in the portfolio database.

    Called when a parent order fills. Returns None if quantity is invalid.
    """
    if quantity <= 0:
        logger.warning("Ignoring fill with invalid quantity %d for %s", quantity, signal.ticker)
        return None

    # IBKR reports filled quantity as positive regardless of side.
    # For a short entry (SELL parent), store negative quantity so P&L
    # math, reconciliation, and exit-side selection stay consistent.
    signed_quantity = -quantity if signal.action == Action.SELL else quantity

    position = Position(
        ticker=signal.ticker,
        exchange=signal.exchange,
        quantity=signed_quantity,
        entry_price=fill_price,
        entry_time=datetime.now(timezone.utc),
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        trade_type=signal.trade_type,
        sector=signal.indicator_values.get("sector", ""),
    )

    kwargs = {"db_path": db_path} if db_path else {}
    add_position(position, **kwargs)
    logger.info("Recorded fill: %s %d @ $%.2f", signal.ticker, quantity, fill_price)
    return position


def setup_fill_handler(
    ib: IB,
    signal: Signal,
    quantity: int,
    on_fill=None,
    parent_order=None,
    parent_trade: Optional[IBTrade] = None,
) -> None:
    """Attach a callback to handle entry order fills asynchronously.

    Args:
        on_fill: Optional callback(signal, filled_qty, fill_price) called on fill.
        parent_order: Optional parent order object; when set, matches by permId
                      instead of ticker to prevent double-recording.
        parent_trade: Optional parent trade snapshot. If the trade is already
                      "Filled" at registration time (fast fill during place_order's
                      permId poll), the fill path runs immediately. ib_insync
                      does not replay past events, so without this check a fast
                      fill would be silently dropped.
    """

    # Use threading.Event for atomic check-and-set to prevent double-fill
    # recording when rapid consecutive fills fire from ib_insync's event thread
    _fired = threading.Event()
    # Capture permId at registration time for precise matching
    _parent_perm_id = getattr(parent_order, "permId", 0) if parent_order else 0

    def _process_fill(trade: IBTrade) -> None:
        fill_price = trade.orderStatus.avgFillPrice
        filled_qty = int(trade.orderStatus.filled)
        if filled_qty < quantity:
            logger.warning(
                "Partial fill for %s: %d/%d shares @ $%.2f",
                signal.ticker, filled_qty, quantity, fill_price,
            )
        handle_fill(signal, filled_qty, fill_price)
        if trade.order.permId:
            remove_pending_order(trade.order.permId)
        if on_fill:
            on_fill(signal, filled_qty, fill_price)

    def on_order_status(trade: IBTrade):
        if _fired.is_set():
            return
        if trade.orderStatus.status != "Filled":
            return
        # Match by permId when available (precise), fall back to ticker
        if _parent_perm_id:
            if trade.order.permId != _parent_perm_id:
                return
        else:
            if trade.contract.symbol != signal.ticker:
                return
        if trade.order.action not in ("BUY", "SELL"):
            return
        if getattr(trade.order, "parentId", 0) != 0:
            return
        if _fired.is_set():
            return
        _fired.set()
        try:
            _process_fill(trade)
        finally:
            try:
                ib.orderStatusEvent -= on_order_status
            except Exception:
                pass

    ib.orderStatusEvent += on_order_status

    # Race guard: fast fills can arrive before the handler attaches. If the
    # parent trade is already Filled, run the fill path synchronously here.
    # _fired prevents the same fill from being processed twice if a late event
    # also fires for the same Filled status.
    if parent_trade is not None and parent_trade.orderStatus.status == "Filled":
        if getattr(parent_trade.order, "parentId", 0) == 0:
            if not _fired.is_set():
                _fired.set()
                try:
                    _process_fill(parent_trade)
                finally:
                    try:
                        ib.orderStatusEvent -= on_order_status
                    except Exception:
                        pass


def setup_exit_handler(ib: IB, signal: Signal, on_exit=None, parent_order=None) -> None:
    """Attach a callback to handle exit order fills (TP/SL) asynchronously.

    When a take-profit or stop-loss child order fills, close the position
    in the database and optionally notify via callback.

    Args:
        on_exit: Optional callback(ticker, exit_price, exit_type) called on exit fill.
        parent_order: Optional parent order object; when set, only matches child
                      orders whose parentId equals this order's orderId.
    """

    _fired = threading.Event()
    _parent_order_id = getattr(parent_order, "orderId", 0) if parent_order else 0

    # Without a parent orderId we cannot safely distinguish this bracket's
    # children from a later bracket's children for the same ticker. Matching
    # by ticker alone means the handler can fire on a FRESH re-entry's fill
    # and close the new position via db_close_position — a silent data loss.
    # Decline to register; the caller (reattach_exit_handlers) already logs
    # the warning when it can't locate a parent.
    if not _parent_order_id:
        logger.warning(
            "Refusing to attach exit handler for %s: no parent_order_id "
            "provided. Matching by ticker alone would risk firing on a "
            "later bracket's fills when the ticker is re-entered.",
            signal.ticker,
        )
        return

    def on_order_status(trade: IBTrade):
        if _fired.is_set():
            return
        if trade.orderStatus.status != "Filled":
            return
        if trade.contract.symbol != signal.ticker:
            return

        fill_price = trade.orderStatus.avgFillPrice

        # Detect exit type: STP = stop-loss, LMT on child = take-profit
        is_stop = trade.order.orderType in ("STP", "STP LMT")
        # Child orders have a parentId linking them to the parent
        is_child = getattr(trade.order, "parentId", 0) > 0

        if not is_child:
            return  # This is the parent entry order, handled by setup_fill_handler

        # Match by parent order ID to prevent cross-bracket interference
        # when a ticker is re-entered in the same session.
        if trade.order.parentId != _parent_order_id:
            return

        exit_type = "stop-loss" if is_stop else "take-profit"
        if _fired.is_set():
            return
        _fired.set()

        logger.info(
            "Exit fill: %s %s @ $%.2f (%s)",
            signal.ticker, trade.order.action, fill_price, exit_type,
        )

        # Close position in database
        try:
            trade_record = db_close_position(signal.ticker, fill_price)
            if trade_record:
                logger.info(
                    "Position closed: %s P&L: $%.2f (%.1f%%)",
                    signal.ticker, trade_record.pnl, trade_record.pnl_pct,
                )
                if on_exit:
                    on_exit(signal.ticker, fill_price, exit_type)
            else:
                # No matching DB position — do NOT call on_exit, as it would
                # fetch a stale trade and send an incorrect notification
                logger.warning(
                    "Exit fill for %s @ $%.2f (%s) could not be matched to a DB position",
                    signal.ticker, fill_price, exit_type,
                )
        finally:
            ib.orderStatusEvent -= on_order_status

    ib.orderStatusEvent += on_order_status


def import_ibkr_positions(ib: IB, orphaned_tickers: list[str]) -> list[str]:
    """Import IBKR positions that exist at the broker but not in the DB.

    Constructs Position objects from IBKR data (avgCost, quantity) and
    extracts stop-loss/take-profit prices from open bracket orders.

    Args:
        ib: Connected IB instance.
        orphaned_tickers: List of tickers present at IBKR but missing from DB.

    Returns:
        List of tickers successfully imported.
    """
    if not orphaned_tickers:
        return []

    orphaned_set = set(orphaned_tickers)
    imported: list[str] = []

    # Build a map of IBKR positions: ticker -> (quantity, avgCost, exchange)
    ibkr_data: dict[str, dict] = {}
    for p in ib.positions():
        ticker = p.contract.symbol
        if ticker in orphaned_set:
            ibkr_data[ticker] = {
                "quantity": int(p.position),
                "avg_cost": p.avgCost,
                "exchange": p.contract.primaryExchange or p.contract.exchange or "SMART",
            }

    # Extract stop-loss and take-profit from open bracket orders
    sl_prices: dict[str, float] = {}
    tp_prices: dict[str, float] = {}
    for trade in ib.openTrades():
        ticker = trade.contract.symbol
        if ticker not in orphaned_set:
            continue
        order = trade.order
        # Child orders have parentId > 0
        if order.parentId == 0:
            continue
        if order.orderType in ("STP", "STP LMT"):
            sl_prices[ticker] = order.auxPrice
        elif order.orderType == "LMT":
            tp_prices[ticker] = order.lmtPrice

    for ticker in orphaned_tickers:
        data = ibkr_data.get(ticker)
        if data is None:
            continue

        qty = data["quantity"]
        entry_price = data["avg_cost"]
        stop_loss = sl_prices.get(ticker, 0.0)
        take_profit = tp_prices.get(ticker, 0.0)

        # Determine trade type from order TIF if available
        trade_type = TradeType.SWING  # Default assumption for imported positions

        position = Position(
            ticker=ticker,
            exchange=data["exchange"],
            quantity=qty,
            entry_price=entry_price,
            entry_time=datetime.now(timezone.utc),
            stop_loss=stop_loss,
            take_profit=take_profit,
            trade_type=trade_type,
        )
        add_position(position)
        imported.append(ticker)
        logger.warning(
            "Imported IBKR position %s: %d @ $%.2f (SL=$%.2f, TP=$%.2f). "
            "Entry time set to now — actual entry time unknown.",
            ticker, qty, entry_price, stop_loss, take_profit,
        )

    return imported


def reattach_exit_handlers(ib: IB) -> int:
    """Re-register exit handlers for existing bracket orders after restart.

    On startup, open positions in the DB may have live stop-loss/take-profit
    orders at IBKR.  Since in-memory event handlers don't survive restarts,
    this function finds those orders and attaches new exit handlers so fills
    are detected and the DB + Telegram cache stay in sync.

    Returns the number of handlers attached.
    """
    from core.portfolio import get_open_positions
    open_positions = get_open_positions()
    if not open_positions:
        return 0

    db_tickers = {p.ticker for p in open_positions}
    open_trades = ib.openTrades()

    # Build a map: ticker -> parent orderId for open parent (entry) orders
    parent_ids: dict[str, int] = {}
    for trade in open_trades:
        order = trade.order
        if order.parentId == 0 and trade.contract.symbol in db_tickers:
            parent_ids[trade.contract.symbol] = order.orderId

    # Deduplicate: one handler per bracket (parent), not per child order.
    # Without this, a bracket with 2 children (SL + TP) would create 2
    # independent handlers, both firing on the same exit fill — causing
    # duplicate db_close_position calls and spurious warnings.
    handled_parents: set[int] = set()
    attached = 0
    for trade in open_trades:
        order = trade.order
        ticker = trade.contract.symbol
        if ticker not in db_tickers:
            continue
        # Only child orders (stop-loss / take-profit)
        if order.parentId == 0:
            continue
        # Match child to a known parent for this ticker
        expected_parent = parent_ids.get(ticker)
        if expected_parent and order.parentId != expected_parent:
            continue

        # Skip if we already attached a handler for this bracket
        bracket_key = order.parentId
        if bracket_key in handled_parents:
            continue
        handled_parents.add(bracket_key)

        # Build a minimal Signal for the handler
        pos = next(p for p in open_positions if p.ticker == ticker)
        signal = Signal(
            ticker=ticker,
            action=Action.BUY if pos.quantity > 0 else Action.SELL,
            confidence=0.0,
            entry_price=pos.entry_price,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            reasoning="reattached on startup",
            source="reattach",
        )

        def make_on_exit(t):
            def _on_exit(ticker, exit_price, exit_type):
                from notifications.telegram import notify_trade_closed, refresh_positions_cache
                from core.portfolio import get_trades
                trades_list = get_trades(ticker=ticker)
                if trades_list:
                    notify_trade_closed(trades_list[0])
                refresh_positions_cache()
            return _on_exit

        # Find the parent Order object to pass for precise matching
        parent_order = None
        if expected_parent:
            for t in open_trades:
                if t.order.orderId == expected_parent:
                    parent_order = t.order
                    break

        setup_exit_handler(ib, signal, on_exit=make_on_exit(ticker), parent_order=parent_order)
        attached += 1
        logger.info("Reattached exit handler for %s (parentId=%d)",
                     ticker, order.parentId)

    return attached


def setup_disconnect_handler(ib: IB, reconnect: bool = True) -> None:
    """Set up handler for connection drops.

    Skips reconnect during shutdown and uses a guard to prevent
    re-entrant reconnect loops. Clears realtime subscription tracking
    since IBKR drops all subscriptions on disconnect.

    Args:
        reconnect: If False, skip manual reconnect (use when Watchdog manages
                   reconnection — manual reconnect competes with the watchdog
                   and fails with "clientId already in use").
    """
    _reconnecting = threading.Event()

    def on_disconnect():
        from core import state as _state
        from core.data import clear_realtime_subscriptions
        if _state.shutting_down or _reconnecting.is_set():
            return
        _reconnecting.set()
        # IBKR drops all subscriptions on disconnect — clear tracking
        # so they can be re-established after reconnect
        clear_realtime_subscriptions(ib)
        if not reconnect:
            logger.warning("IBKR connection lost — watchdog will handle reconnect")
            _reconnecting.clear()
            return
        logger.warning("IBKR connection lost! Attempting reconnect...")
        try:
            ensure_connected(ib)
            logger.info("Reconnected successfully")
        except ConnectionError:
            logger.error("Failed to reconnect to IBKR")
        finally:
            _reconnecting.clear()

    ib.disconnectedEvent += on_disconnect

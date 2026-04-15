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
    """Close all positions marked as day trades. Called before market close."""
    day_trades = [p for p in positions if p.trade_type == TradeType.DAY]

    if not day_trades:
        logger.info("No day trades to close")
        return []

    logger.info("Closing %d day trade positions", len(day_trades))
    trades = []

    for pos in day_trades:
        trade = close_position_market(ib, pos, dry_run)
        if trade:
            trades.append(trade)

    # Verify fills with extended timeout — market orders near close
    # may take longer than expected under adverse conditions
    if trades and not dry_run:
        statuses = monitor_orders(ib, trades, timeout=30.0)
        unfilled = [
            s["orderId"] for s in statuses
            if s["status"] != "Filled"
        ]
        if unfilled:
            logger.error(
                "CRITICAL: Day trade close orders NOT filled after 30s: %s — "
                "positions may remain open overnight! Manual intervention needed.",
                unfilled,
            )

    return trades


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


def setup_fill_handler(ib: IB, signal: Signal, quantity: int, on_fill=None, parent_order=None) -> None:
    """Attach a callback to handle entry order fills asynchronously.

    Args:
        on_fill: Optional callback(signal, filled_qty, fill_price) called on fill.
        parent_order: Optional parent order object; when set, matches by permId
                      instead of ticker to prevent double-recording.
    """

    # Use threading.Event for atomic check-and-set to prevent double-fill
    # recording when rapid consecutive fills fire from ib_insync's event thread
    _fired = threading.Event()
    # Capture permId at registration time for precise matching
    _parent_perm_id = getattr(parent_order, "permId", 0) if parent_order else 0

    def on_order_status(trade: IBTrade):
        if _fired.is_set():
            return
        if trade.orderStatus.status == "Filled":
            # Match by permId when available (precise), fall back to ticker
            if _parent_perm_id:
                if trade.order.permId != _parent_perm_id:
                    return
            else:
                if trade.contract.symbol != signal.ticker:
                    return
            fill_price = trade.orderStatus.avgFillPrice
            filled_qty = int(trade.orderStatus.filled)
            if trade.order.action in ("BUY", "SELL") and getattr(trade.order, "parentId", 0) == 0:
                if _fired.is_set():
                    return
                _fired.set()
                try:
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
                finally:
                    ib.orderStatusEvent -= on_order_status

    ib.orderStatusEvent += on_order_status


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

        # Always match children by parent order ID to prevent cross-bracket
        # interference when a ticker is re-entered in the same session
        if _parent_order_id and trade.order.parentId != _parent_order_id:
            return
        if not _parent_order_id:
            logger.warning(
                "Exit handler for %s has no parent_order_id — "
                "matching by ticker only (risk of cross-bracket fire)",
                signal.ticker,
            )

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

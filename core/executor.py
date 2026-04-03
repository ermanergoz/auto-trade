"""IBKR order execution — bracket orders, monitoring, fill handling."""

import logging
from datetime import datetime
from typing import Optional

from ib_insync import IB, Trade as IBTrade, Order, LimitOrder, MarketOrder, StopOrder

from core.models import Signal, Position, Trade, Action, TradeType
from core.connection import create_contract, ensure_connected
from core.portfolio import add_position, close_position as db_close_position

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

    # GTC ensures orders survive overnight and fill at market open.
    # transmit=False on parent and TP so all three transmit atomically
    # when the last order (SL) is placed with transmit=True.
    for o in bracket:
        o.tif = "GTC"
    parent_order.transmit = False
    tp_order.transmit = False
    sl_order.transmit = True

    # Place all three orders (only the last one triggers transmission)
    trades = []
    try:
        parent_trade = ib.placeOrder(contract, parent_order)
        trades.append(parent_trade)

        tp_trade = ib.placeOrder(contract, tp_order)
        trades.append(tp_trade)

        sl_trade = ib.placeOrder(contract, sl_order)
        trades.append(sl_trade)

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
    """Check status of placed orders. Returns list of status dicts."""
    statuses = []
    for trade in trades:
        ib.sleep(0.1)  # Allow event processing
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
    except Exception as e:
        logger.error("Failed to cancel order: %s", e)


def close_position_market(
    ib: IB,
    position: Position,
    dry_run: bool = False,
) -> Optional[IBTrade]:
    """Close an open position with a market order."""
    action = "SELL"  # Assuming we're closing a long position
    return place_market_order(
        ib, position.ticker, position.exchange,
        action, position.quantity, dry_run,
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

    return trades


def handle_fill(
    signal: Signal,
    quantity: int,
    fill_price: float,
    db_path=None,
) -> Position:
    """Record a filled order in the portfolio database.

    Called when a parent order fills.
    """
    position = Position(
        ticker=signal.ticker,
        exchange=signal.exchange,
        quantity=quantity,
        entry_price=fill_price,
        entry_time=datetime.now(),
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        trade_type=signal.trade_type,
        sector=signal.indicator_values.get("sector", ""),
    )

    kwargs = {"db_path": db_path} if db_path else {}
    add_position(position, **kwargs)
    logger.info("Recorded fill: %s %d @ $%.2f", signal.ticker, quantity, fill_price)
    return position


def setup_fill_handler(ib: IB, signal: Signal, quantity: int, on_fill=None) -> None:
    """Attach a callback to handle order fills asynchronously.

    Args:
        on_fill: Optional callback(signal, filled_qty, fill_price) called on fill.
    """

    def on_order_status(trade: IBTrade):
        if trade.orderStatus.status == "Filled":
            fill_price = trade.orderStatus.avgFillPrice
            filled_qty = int(trade.orderStatus.filled)
            if trade.order.action in ("BUY", "SELL") and trade.order.orderType != "STP":
                handle_fill(signal, filled_qty, fill_price)
                if on_fill:
                    on_fill(signal, filled_qty, fill_price)

    ib.orderStatusEvent += on_order_status


def setup_disconnect_handler(ib: IB) -> None:
    """Set up handler for connection drops.

    Skips reconnect during shutdown and uses a guard to prevent
    re-entrant reconnect loops.
    """
    _reconnecting = False

    def on_disconnect():
        nonlocal _reconnecting
        from core import state as _state
        if _state.shutting_down or _reconnecting:
            return
        _reconnecting = True
        logger.warning("IBKR connection lost! Attempting reconnect...")
        try:
            ensure_connected(ib)
            logger.info("Reconnected successfully")
        except ConnectionError:
            logger.error("Failed to reconnect to IBKR")
        finally:
            _reconnecting = False

    ib.disconnectedEvent += on_disconnect

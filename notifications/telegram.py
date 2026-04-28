"""Telegram bot notifications + interactive status via polling."""

import asyncio
import html as _html
import logging
import threading
from datetime import datetime
from typing import Optional

from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TIMEZONE, is_paper_mode
from core.models import Trade, DailySummary, Signal

logger = logging.getLogger(__name__)

# Shared system state for status responses
_status_lock = threading.Lock()
_system_status = {
    "phase": "initializing",
    "mode": "",
    "detail": "",
    "last_scan": None,
    "last_summary": None,
    "account": None,
    "positions": None,
    "daily_pnl": None,
}

# IB connection reference for live account queries
_ib_ref = None


def update_status(phase: str, detail: str = "") -> None:
    """Update the current system status (called from scheduler/main)."""
    with _status_lock:
        _system_status["phase"] = phase
        _system_status["detail"] = detail


def _run_async(coro):
    """Run an async coroutine in a fresh event loop (thread-safe)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# Tags Telegram accepts in HTML parse mode and that the codebase actually emits.
# Everything else (e.g. <br> appearing inside IBKR error reasons) is escaped.
_ALLOWED_HTML_TAGS = (
    "b", "/b", "i", "/i", "u", "/u", "s", "/s",
    "code", "/code", "pre", "/pre",
)


def _sanitize_html(text: str) -> str:
    """Escape unsafe HTML while preserving Telegram-supported formatting tags.

    Embedded user- or broker-supplied strings (e.g. IBKR rejection reasons) can
    contain fragments like ``<br>`` that crash Telegram's HTML parse mode. This
    sanitizer escapes everything through :func:`html.escape`, then restores the
    small whitelist of tags this module uses in its notify_* helpers.
    """
    escaped = _html.escape(text, quote=False)
    for tag in _ALLOWED_HTML_TAGS:
        escaped = escaped.replace(f"&lt;{tag}&gt;", f"<{tag}>")
    return escaped


def _send_sync(text: str) -> bool:
    """Send a message synchronously (fire-and-forget)."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured — skipping notification")
        return False

    try:
        from telegram import Bot
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        _run_async(
            bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=_sanitize_html(text),
                parse_mode="HTML",
            )
        )
        return True
    except Exception as e:
        logger.error("Telegram send failed: %s", e)
        return False


def _get_updates_sync(offset: Optional[int] = None) -> list:
    """Fetch new messages from Telegram."""
    if not TELEGRAM_BOT_TOKEN:
        return []

    try:
        from telegram import Bot
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        return _run_async(bot.get_updates(offset=offset, timeout=10))
    except Exception as e:
        logger.debug("Telegram get_updates failed: %s", e)
        return []


def _is_status_command(text: str) -> bool:
    """Check if incoming text is a status command."""
    return text.strip().lower() in ("status", "/status")


def update_portfolio_data(account: dict, positions: list, daily_pnl: float) -> None:
    """Cache portfolio data for status responses (called from scheduler)."""
    with _status_lock:
        _system_status["account"] = account
        _system_status["positions"] = positions
        _system_status["daily_pnl"] = daily_pnl


def set_ib_instance(ib) -> None:
    """Store the IB connection reference for live account queries."""
    global _ib_ref
    _ib_ref = ib


def refresh_positions_cache() -> None:
    """Re-read positions and account data, fetching fresh values from IBKR.

    Call this after a position is opened/closed outside the scan cycle
    (e.g. async stop-loss fill) so /status reflects reality immediately.
    """
    from core.portfolio import get_open_positions, get_daily_pnl

    positions = get_open_positions()

    # Fetch fresh account data from IBKR if connected
    account = None
    if _ib_ref is not None:
        try:
            from core.connection import get_account_summary
            account = get_account_summary(_ib_ref)
        except Exception as e:
            logger.debug("Live account refresh failed: %s", e)

    with _status_lock:
        if account is not None:
            _system_status["account"] = account
        unrealized = (_system_status.get("account") or {}).get("UnrealizedPnL", 0.0)

    daily_pnl = get_daily_pnl(unrealized_pnl=unrealized)

    with _status_lock:
        _system_status["positions"] = positions
        _system_status["daily_pnl"] = daily_pnl


def _build_status_response() -> str:
    """Build a human-readable status message with portfolio data."""
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(TIMEZONE)
    now = datetime.now(tz).strftime("%H:%M:%S")

    with _status_lock:
        status_snapshot = dict(_system_status)

    phase = status_snapshot["phase"]
    detail = status_snapshot["detail"]
    mode = status_snapshot["mode"]

    lines = [f"<b>Status</b> ({now})"]

    if mode:
        lines.append(f"Mode: {mode}")

    phase_labels = {
        "initializing": "Starting up...",
        "connecting": "Connecting to IBKR...",
        "startup_complete": "Ready, waiting for market hours",
        "building_universe": "Building stock universe",
        "enriching": "Enriching stocks with sector data",
        "fetching_data": "Fetching market data",
        "screening": "Running technical screener",
        "ai_analysis": "AI analyzing candidates",
        "risk_check": "Running risk checks",
        "executing": "Placing orders",
        "scan_complete": "Scan complete, waiting for next cycle",
        "waiting": "Waiting for next scan cycle",
        "closing_day_trades": "Closing day trades (market closing)",
        "shutting_down": "Shutting down...",
    }

    lines.append(f"Phase: {phase_labels.get(phase, phase)}")

    if detail:
        lines.append(f"Detail: {detail}")

    # Account summary
    account = status_snapshot.get("account")
    if account:
        lines.append("")
        lines.append("<b>Account</b>")
        nlv = account.get("NetLiquidation", 0)
        cash = account.get("TotalCashValue", 0)
        invested = account.get("GrossPositionValue", 0)
        unrealized = account.get("UnrealizedPnL", 0)
        lines.append(f"Portfolio Value: ${nlv:,.2f}")
        lines.append(f"Cash Available: ${cash:,.2f}")
        lines.append(f"Invested: ${invested:,.2f}")
        lines.append(f"Unrealized P&L: ${unrealized:+,.2f}")

    # P&L section
    daily_pnl = status_snapshot.get("daily_pnl")
    unrealized = (account or {}).get("UnrealizedPnL", 0)
    realized_pnl = (daily_pnl - unrealized) if daily_pnl is not None else None

    lines.append("")
    lines.append("<b>P&L</b>")
    if unrealized:
        lines.append(f"Unrealized: ${unrealized:+,.2f}")
    if realized_pnl is not None:
        lines.append(f"Realized (today): ${realized_pnl:+,.2f}")
    if daily_pnl is not None:
        lines.append(f"Total (today): ${daily_pnl:+,.2f}")

    # Trade stats
    from core.portfolio import get_trades
    from datetime import date as _date
    today = _date.today()
    today_trades = get_trades(start_date=today, end_date=today)
    if today_trades:
        winners = [t for t in today_trades if t.pnl > 0]
        losers = [t for t in today_trades if t.pnl < 0]
        lines.append(f"Trades today: {len(today_trades)} (W:{len(winners)} / L:{len(losers)})")

    # Open positions with live prices from IBKR
    positions = status_snapshot.get("positions")
    live_prices = {}
    if positions and _ib_ref is not None:
        try:
            for ibkr_pos in _ib_ref.positions():
                live_prices[ibkr_pos.contract.symbol] = ibkr_pos.avgCost  # fallback
            for pv in _ib_ref.portfolio():
                live_prices[pv.contract.symbol] = pv.marketPrice
        except Exception as e:
            logger.debug("Failed to fetch live prices: %s", e)

    if positions:
        lines.append("")
        lines.append(f"<b>Open Positions ({len(positions)})</b>")
        for pos in positions:
            price = live_prices.get(pos.ticker, pos.current_price)
            if price is not None:
                mkt_value = price * pos.quantity
                pnl = (price - pos.entry_price) * pos.quantity
                pct = ((price - pos.entry_price) / pos.entry_price * 100) if pos.entry_price else 0.0
                lines.append(
                    f"  {pos.ticker}: {pos.quantity} @ ${pos.entry_price:.2f}"
                    f" | Now ${price:.2f} | Val ${mkt_value:,.2f} | {pct:+.1f}%"
                )
            else:
                lines.append(f"  {pos.ticker}: {pos.quantity} @ ${pos.entry_price:.2f}")

    # Open orders from IBKR
    if _ib_ref is not None:
        try:
            open_trades = _ib_ref.openTrades()
            orders = []
            for t in open_trades:
                o = t.order
                c = t.contract
                status = t.orderStatus.status
                price_str = ""
                if o.orderType == "LMT" and o.lmtPrice:
                    price_str = f" ${o.lmtPrice:.2f}"
                elif o.orderType == "STP" and o.auxPrice:
                    price_str = f" ${o.auxPrice:.2f}"
                elif o.orderType == "STP LMT":
                    parts = []
                    if o.auxPrice:
                        parts.append(f"stop ${o.auxPrice:.2f}")
                    if o.lmtPrice:
                        parts.append(f"lmt ${o.lmtPrice:.2f}")
                    if parts:
                        price_str = f" {', '.join(parts)}"
                orders.append(f"  {c.symbol}: {o.action} {o.totalQuantity} {o.orderType}{price_str} — {status}")
            if orders:
                lines.append("")
                lines.append(f"<b>Open Orders ({len(orders)})</b>")
                lines.extend(orders)
        except Exception as e:
            logger.debug("Failed to fetch open orders: %s", e)

    # Last scan summary
    last = status_snapshot.get("last_summary")
    if last:
        lines.append(f"\nLast scan: {last}")

    return "\n".join(lines)


_MAX_LISTENER_ERRORS = 10
_stop_event = threading.Event()


def _poll_loop() -> None:
    """Background thread that polls for incoming Telegram messages."""
    import time
    logger.info("Telegram listener started — send 'status' to get status")
    offset = None
    consecutive_errors = 0

    while not _stop_event.is_set():
        try:
            updates = _get_updates_sync(offset=offset)
            consecutive_errors = 0  # Reset on success
            for update in updates:
                offset = update.update_id + 1

                msg = update.message
                if not msg or not msg.text:
                    continue

                # Only respond to messages from the configured chat
                if str(msg.chat_id) != str(TELEGRAM_CHAT_ID):
                    continue

                text = msg.text.strip()
                if _is_status_command(text):
                    # Refresh from DB so the response reflects the latest state,
                    # not just what was cached at the last scan cycle
                    try:
                        refresh_positions_cache()
                    except Exception as e:
                        logger.debug("Position cache refresh failed: %s", e)
                    response = _build_status_response()
                    _send_sync(response)

        except Exception as e:
            consecutive_errors += 1
            logger.warning(
                "Telegram poll error (%d/%d): %s",
                consecutive_errors, _MAX_LISTENER_ERRORS, e,
            )
            if consecutive_errors >= _MAX_LISTENER_ERRORS:
                logger.error("Telegram listener exceeded max errors — stopping")
                return

        _stop_event.wait(1)


def stop_listener() -> None:
    """Signal the Telegram listener thread to stop."""
    _stop_event.set()


def start_listener() -> None:
    """Start the background Telegram message listener thread."""
    _stop_event.clear()  # allow restart after a previous stop
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured — listener not started")
        return

    thread = threading.Thread(target=_poll_loop, daemon=True, name="telegram-listener")
    thread.start()


# ---------------------------------------------------------------------------
# Public API — notifications
# ---------------------------------------------------------------------------

def send_message(text: str) -> bool:
    """Send a plain text message to the configured Telegram chat."""
    return _send_sync(text)


def notify_startup(mode: str, account_summary: dict) -> bool:
    """Send startup notification with account info."""
    resolved_mode = "paper" if is_paper_mode() else "live"
    with _status_lock:
        _system_status["mode"] = resolved_mode
    nlv = account_summary.get("NetLiquidation", 0)
    cash = account_summary.get("TotalCashValue", 0)
    text = (
        f"<b>System Started</b>\n\n"
        f"Mode: {resolved_mode}\n"
        f"Portfolio: ${nlv:,.2f}\n"
        f"Cash: ${cash:,.2f}"
    )
    return _send_sync(text)


def notify_shutdown() -> bool:
    """Send shutdown notification."""
    update_status("shutting_down")  # already lock-protected
    return _send_sync("<b>System Stopped</b>\n\nTrader has been shut down.")


def notify_ai_signal(signal: Signal) -> bool:
    """Notify when AI recommends a buy/sell (before risk check)."""
    emoji = "\U0001f4a1"  # lightbulb
    action = signal.action.value.upper()
    text = (
        f"{emoji} <b>AI Signal: {action} {signal.ticker}</b>\n\n"
        f"Confidence: {signal.confidence:.0f}%\n"
        f"Entry: ${signal.entry_price:.2f}\n"
        f"Stop-Loss: ${signal.stop_loss:.2f}\n"
        f"Take-Profit: ${signal.take_profit:.2f}\n"
        f"Type: {signal.trade_type.value}\n\n"
        f"<i>{signal.reasoning[:200]}</i>"
    )
    return _send_sync(text)


def notify_risk_results(signals: list[Signal]) -> bool:
    """Send a consolidated summary of all risk-approved signals."""
    if not signals:
        return False

    lines = ["\u2705 <b>Risk-Approved Signals</b>\n"]
    for sig in signals:
        action = sig.action.value.upper()
        emoji = "\U0001f7e2" if action == "BUY" else "\U0001f534"
        lines.append(
            f"{emoji} <b>{sig.ticker}</b> \u2014 {action}\n"
            f"   Confidence: {sig.confidence:.0f}% | "
            f"Entry: ${sig.entry_price:.2f} | "
            f"SL: ${sig.stop_loss:.2f} | TP: ${sig.take_profit:.2f}"
        )
    lines.append(f"\nTotal: {len(signals)} signal(s) approved")
    return _send_sync("\n".join(lines))


def notify_trade(signal: Signal, quantity: int, action_type: str = "OPENED") -> bool:
    """Send a formatted trade notification."""
    emoji = "\U0001f7e2" if signal.action.value == "buy" else "\U0001f534"
    text = (
        f"{emoji} <b>Trade {action_type}</b>\n\n"
        f"<b>{signal.ticker}</b> ({signal.exchange})\n"
        f"Action: {signal.action.value.upper()}\n"
        f"Quantity: {quantity}\n"
        f"Price: ${signal.entry_price:.2f}\n"
        f"Stop-Loss: ${signal.stop_loss:.2f}\n"
        f"Take-Profit: ${signal.take_profit:.2f}\n"
        f"Confidence: {signal.confidence:.0f}%\n"
        f"Type: {signal.trade_type.value}\n\n"
        f"<i>{signal.reasoning[:200]}</i>"
    )
    return _send_sync(text)


def notify_trade_closed(trade: Trade) -> bool:
    """Notify when a trade is closed."""
    pnl_emoji = "\u2705" if trade.pnl >= 0 else "\u274c"
    text = (
        f"{pnl_emoji} <b>Trade CLOSED</b>\n\n"
        f"<b>{trade.ticker}</b> ({trade.exchange})\n"
        f"Entry: ${trade.entry_price:.2f}\n"
        f"Exit: ${trade.exit_price:.2f}\n"
        f"P&L: ${trade.pnl:.2f} ({trade.pnl_pct:+.1f}%)\n"
        f"Duration: {trade.duration:.1f}h"
    )
    return _send_sync(text)


def notify_daily_summary(summary: DailySummary) -> bool:
    """Send end-of-day portfolio summary."""
    pnl_emoji = "\U0001f4c8" if summary.daily_pnl >= 0 else "\U0001f4c9"
    text = (
        f"{pnl_emoji} <b>Daily Summary</b>\n\n"
        f"Date: {summary.date}\n"
        f"Portfolio: ${summary.portfolio_value:,.2f}\n"
        f"Daily P&L: ${summary.daily_pnl:+,.2f} ({summary.daily_pnl_pct:+.2f}%)\n"
        f"Trades: {summary.num_trades}\n"
        f"Winners: {summary.winning_trades} | Losers: {summary.losing_trades}"
    )
    return _send_sync(text)


def notify_risk_warning(message: str) -> bool:
    """Send a risk warning alert."""
    text = f"\u26a0\ufe0f <b>Risk Warning</b>\n\n{message}"
    return _send_sync(text)


def notify_error(error: str) -> bool:
    """Send a system error notification."""
    text = f"\U0001f6a8 <b>System Error</b>\n\n<code>{error[:500]}</code>"
    return _send_sync(text)


def notify_reconciliation_mismatch(report: dict) -> bool:
    """Alert when nightly reconciliation detects DB/IBKR drift.

    Sign mismatches (DB long vs IBKR short) are flagged as critical —
    manual intervention required.
    """
    qty_mismatches = report.get("qty_mismatches", {}) or {}
    sign_mismatches = [t for t, v in qty_mismatches.items() if v.get("type") == "sign_mismatch"]

    header = "\U0001f6a8 <b>CRITICAL: Direction Mismatch</b>" if sign_mismatches \
        else "\u26a0\ufe0f <b>Reconciliation Mismatch</b>"

    lines = [header, ""]

    orphaned_db = report.get("orphaned_db", []) or []
    if orphaned_db:
        lines.append(f"<b>In DB but not in IBKR</b> ({len(orphaned_db)}):")
        for t in orphaned_db:
            lines.append(f"  \u2022 <code>{t}</code>")

    orphaned_ibkr = report.get("orphaned_ibkr", []) or []
    if orphaned_ibkr:
        lines.append(f"<b>In IBKR but not in DB</b> ({len(orphaned_ibkr)}):")
        for t in orphaned_ibkr:
            lines.append(f"  \u2022 <code>{t}</code>")

    if qty_mismatches:
        lines.append(f"<b>Quantity mismatches</b> ({len(qty_mismatches)}):")
        for t, v in qty_mismatches.items():
            tag = " [sign mismatch]" if v.get("type") == "sign_mismatch" else ""
            lines.append(f"  \u2022 <code>{t}</code>: DB={v.get('db')} IBKR={v.get('ibkr')}{tag}")

    lines.append("")
    lines.append("Manual review required.")
    return _send_sync("\n".join(lines))


def notify_stale_order_cancelled(ticker: str, age_hours: float, reason: str = "Failed re-screening") -> bool:
    """Notify that a stale unfilled order was cancelled after re-evaluation."""
    text = (
        f"\U0001f6ab <b>Stale Order Cancelled</b>\n\n"
        f"Ticker: <code>{ticker}</code>\n"
        f"Age: {age_hours:.1f}h\n"
        f"Reason: {reason}"
    )
    return _send_sync(text)


def notify_scan_summary(
    candidates: int,
    ai_approved: int,
    risk_approved: int,
    orders_placed: int,
) -> bool:
    """Send a brief scan cycle summary."""
    summary_line = (
        f"Candidates: {candidates} | AI approved: {ai_approved} | "
        f"Risk approved: {risk_approved} | Orders: {orders_placed}"
    )
    with _status_lock:
        _system_status["last_summary"] = summary_line

    text = (
        f"\U0001f50d <b>Scan Complete</b>\n\n"
        f"Screener candidates: {candidates}\n"
        f"AI approved: {ai_approved}\n"
        f"Risk approved: {risk_approved}\n"
        f"Orders placed: {orders_placed}"
    )
    return _send_sync(text)

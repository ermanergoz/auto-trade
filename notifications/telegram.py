"""Telegram bot notifications + interactive status via polling."""

import asyncio
import logging
import threading
from datetime import datetime
from typing import Optional

from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TIMEZONE
from core.models import Trade, DailySummary, Signal

logger = logging.getLogger(__name__)

# Shared system state for "Whatsup" responses
_system_status = {
    "phase": "initializing",
    "mode": "",
    "detail": "",
    "last_scan": None,
    "last_summary": None,
}


def update_status(phase: str, detail: str = "") -> None:
    """Update the current system status (called from scheduler/main)."""
    _system_status["phase"] = phase
    _system_status["detail"] = detail


def _run_async(coro):
    """Run an async coroutine in a fresh event loop (thread-safe)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


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
                text=text,
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


def _build_status_response() -> str:
    """Build a human-readable status message."""
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(TIMEZONE)
    now = datetime.now(tz).strftime("%H:%M:%S")

    phase = _system_status["phase"]
    detail = _system_status["detail"]
    mode = _system_status["mode"]

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

    last = _system_status.get("last_summary")
    if last:
        lines.append(f"\nLast scan: {last}")

    return "\n".join(lines)


def _poll_loop() -> None:
    """Background thread that polls for incoming Telegram messages."""
    logger.info("Telegram listener started — send 'Whatsup' to get status")
    offset = None

    while True:
        try:
            updates = _get_updates_sync(offset=offset)
            for update in updates:
                offset = update.update_id + 1

                msg = update.message
                if not msg or not msg.text:
                    continue

                # Only respond to messages from the configured chat
                if str(msg.chat_id) != str(TELEGRAM_CHAT_ID):
                    continue

                text = msg.text.strip().lower()
                if text in ("whatsup", "whats up", "what's up", "status", "/status"):
                    response = _build_status_response()
                    _send_sync(response)

        except Exception as e:
            logger.debug("Telegram poll error: %s", e)

        # Small sleep between polls (the timeout in get_updates does most of the waiting)
        import time
        time.sleep(1)


def start_listener() -> None:
    """Start the background Telegram message listener thread."""
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
    _system_status["mode"] = mode
    nlv = account_summary.get("NetLiquidation", 0)
    cash = account_summary.get("TotalCashValue", 0)
    text = (
        f"<b>System Started</b>\n\n"
        f"Mode: {mode}\n"
        f"Portfolio: ${nlv:,.2f}\n"
        f"Cash: ${cash:,.2f}"
    )
    return _send_sync(text)


def notify_shutdown() -> bool:
    """Send shutdown notification."""
    update_status("shutting_down")
    return _send_sync("<b>System Stopped</b>\n\nTrader has been shut down.")


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
    _system_status["last_summary"] = summary_line

    text = (
        f"\U0001f50d <b>Scan Complete</b>\n\n"
        f"Screener candidates: {candidates}\n"
        f"AI approved: {ai_approved}\n"
        f"Risk approved: {risk_approved}\n"
        f"Orders placed: {orders_placed}"
    )
    return _send_sync(text)

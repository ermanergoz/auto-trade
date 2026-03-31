"""Telegram bot notifications — fire-and-forget alerts."""

import asyncio
import logging
from typing import Optional

from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from core.models import Trade, DailySummary, Signal

logger = logging.getLogger(__name__)

_bot = None


def _get_bot():
    """Lazy-init the Telegram bot."""
    global _bot
    if _bot is None:
        if not TELEGRAM_BOT_TOKEN:
            return None
        from telegram import Bot
        _bot = Bot(token=TELEGRAM_BOT_TOKEN)
    return _bot


def _send_sync(text: str) -> bool:
    """Send a message synchronously (fire-and-forget)."""
    bot = _get_bot()
    if not bot or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured — skipping notification")
        return False

    try:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=text,
                    parse_mode="HTML",
                )
            )
        finally:
            loop.close()
        return True
    except Exception as e:
        logger.error("Telegram send failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def send_message(text: str) -> bool:
    """Send a plain text message to the configured Telegram chat."""
    return _send_sync(text)


def notify_trade(signal: Signal, quantity: int, action_type: str = "OPENED") -> bool:
    """Send a formatted trade notification."""
    emoji = "🟢" if signal.action.value == "buy" else "🔴"
    text = (
        f"{emoji} <b>Trade {action_type}</b>\n\n"
        f"<b>{signal.ticker}</b> ({signal.exchange})\n"
        f"Action: {signal.action.value.upper()}\n"
        f"Quantity: {quantity}\n"
        f"Price: ${signal.entry_price:.2f}\n"
        f"Stop-Loss: ${signal.stop_loss:.2f}\n"
        f"Take-Profit: ${signal.take_profit:.2f}\n"
        f"Confidence: {signal.confidence:.0f}%\n"
        f"Source: {signal.source}\n\n"
        f"<i>{signal.reasoning[:200]}</i>"
    )
    return _send_sync(text)


def notify_trade_closed(trade: Trade) -> bool:
    """Notify when a trade is closed."""
    pnl_emoji = "✅" if trade.pnl >= 0 else "❌"
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
    pnl_emoji = "📈" if summary.daily_pnl >= 0 else "📉"
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
    text = f"⚠️ <b>Risk Warning</b>\n\n{message}"
    return _send_sync(text)


def notify_error(error: str) -> bool:
    """Send a system error notification."""
    text = f"🚨 <b>System Error</b>\n\n<code>{error[:500]}</code>"
    return _send_sync(text)


def notify_scan_summary(
    candidates: int,
    ai_approved: int,
    risk_approved: int,
    orders_placed: int,
) -> bool:
    """Send a brief scan cycle summary."""
    text = (
        f"🔍 <b>Scan Complete</b>\n\n"
        f"Screener candidates: {candidates}\n"
        f"AI approved: {ai_approved}\n"
        f"Risk approved: {risk_approved}\n"
        f"Orders placed: {orders_placed}"
    )
    return _send_sync(text)

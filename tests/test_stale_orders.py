"""Tests for stale order detection using persistent DB timestamps.

The bug: ib_insync resets trade.log on reconnection, so orders appeared
brand new after every watchdog restart.  The fix persists placement
times in the pending_orders table.
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.portfolio import (
    init_db, save_pending_order, get_pending_order_time, remove_pending_order,
)


# ---------------------------------------------------------------------------
# pending_orders DB helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test_stale.db"
    init_db(path)
    return path


class TestPendingOrdersDB:
    def test_save_and_get(self, db_path):
        save_pending_order(12345, "AAPL", db_path)
        t = get_pending_order_time(12345, db_path)
        assert t is not None
        assert isinstance(t, datetime)

    def test_get_nonexistent(self, db_path):
        assert get_pending_order_time(99999, db_path) is None

    def test_remove(self, db_path):
        save_pending_order(12345, "AAPL", db_path)
        remove_pending_order(12345, db_path)
        assert get_pending_order_time(12345, db_path) is None

    def test_save_duplicate_ignored(self, db_path):
        save_pending_order(12345, "AAPL", db_path)
        save_pending_order(12345, "AAPL", db_path)  # should not raise
        assert get_pending_order_time(12345, db_path) is not None


# ---------------------------------------------------------------------------
# get_stale_orders: uses DB timestamp, not trade.log
# ---------------------------------------------------------------------------

def _make_mock_trade(perm_id, ticker, status="Submitted", parent_id=0,
                     order_type="LMT", log_time=None):
    """Build a mock ib_insync Trade object."""
    trade = MagicMock()
    trade.order.parentId = parent_id
    trade.order.orderType = order_type
    trade.order.permId = perm_id
    trade.orderStatus.status = status
    trade.contract.symbol = ticker
    trade.contract.exchange = "SMART"
    trade.contract.primaryExchange = "NASDAQ"
    if log_time:
        entry = MagicMock()
        entry.time = log_time
        trade.log = [entry]
    else:
        trade.log = []
    return trade


class TestGetStaleOrders:
    """Verify get_stale_orders prefers the persistent DB timestamp."""

    def test_uses_db_timestamp_not_log(self, db_path):
        """Order placed 25h ago in DB but reconnected 1 min ago in log.
        Must be detected as stale (the bug scenario)."""
        from core.executor import get_stale_orders

        perm_id = 555
        now = datetime.now(timezone.utc)

        # DB says order was placed 25 hours ago
        with patch("core.portfolio.DB_PATH", db_path):
            save_pending_order(perm_id, "ILAG", db_path)

        # Backdate the DB record
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        old_time = (now - timedelta(hours=25)).isoformat()
        conn.execute("UPDATE pending_orders SET placed_at = ? WHERE perm_id = ?",
                      (old_time, perm_id))
        conn.commit()
        conn.close()

        # trade.log says order appeared 1 minute ago (reconnection)
        log_time = now - timedelta(minutes=1)
        mock_trade = _make_mock_trade(perm_id, "ILAG", log_time=log_time)
        ib = MagicMock()
        ib.openTrades.return_value = [mock_trade]

        with patch("core.executor.get_pending_order_time") as mock_get_time:
            mock_get_time.return_value = datetime.fromisoformat(old_time)
            stale = get_stale_orders(ib, stale_minutes=1440)

        assert len(stale) == 1
        assert stale[0]["ticker"] == "ILAG"
        assert stale[0]["age_minutes"] >= 25 * 60

    def test_falls_back_to_log_when_no_db_record(self):
        """Orders placed before the fix have no DB record — use trade.log."""
        from core.executor import get_stale_orders

        now = datetime.now(timezone.utc)
        log_time = now - timedelta(hours=25)
        mock_trade = _make_mock_trade(777, "OLD", log_time=log_time)
        ib = MagicMock()
        ib.openTrades.return_value = [mock_trade]

        with patch("core.executor.get_pending_order_time", return_value=None):
            stale = get_stale_orders(ib, stale_minutes=1440)

        assert len(stale) == 1
        assert stale[0]["ticker"] == "OLD"

    def test_young_order_not_stale(self):
        """Order placed 1 hour ago should not be flagged."""
        from core.executor import get_stale_orders

        now = datetime.now(timezone.utc)
        placed_at = now - timedelta(hours=1)

        mock_trade = _make_mock_trade(888, "NEW", log_time=now)
        ib = MagicMock()
        ib.openTrades.return_value = [mock_trade]

        with patch("core.executor.get_pending_order_time", return_value=placed_at):
            stale = get_stale_orders(ib, stale_minutes=1440)

        assert len(stale) == 0

    def test_skips_child_orders(self):
        """Child orders (SL/TP with parentId != 0) must be skipped."""
        from core.executor import get_stale_orders

        now = datetime.now(timezone.utc)
        child = _make_mock_trade(111, "CHILD", parent_id=100,
                                 log_time=now - timedelta(hours=48))
        ib = MagicMock()
        ib.openTrades.return_value = [child]

        with patch("core.executor.get_pending_order_time", return_value=None):
            stale = get_stale_orders(ib, stale_minutes=1440)

        assert len(stale) == 0

    def test_skips_non_limit_orders(self):
        """Stop orders should be skipped."""
        from core.executor import get_stale_orders

        now = datetime.now(timezone.utc)
        stop = _make_mock_trade(222, "STP", order_type="STP",
                                log_time=now - timedelta(hours=48))
        ib = MagicMock()
        ib.openTrades.return_value = [stop]

        with patch("core.executor.get_pending_order_time", return_value=None):
            stale = get_stale_orders(ib, stale_minutes=1440)

        assert len(stale) == 0


# ---------------------------------------------------------------------------
# cancel_bracket_order cleans up DB record
# ---------------------------------------------------------------------------

class TestCancelCleansDB:
    def test_cancel_removes_pending_order(self):
        """cancel_bracket_order must remove the pending_orders DB record."""
        from core.executor import cancel_bracket_order

        mock_trade = _make_mock_trade(333, "GONE")
        ib = MagicMock()

        with patch("core.executor.remove_pending_order") as mock_remove:
            cancel_bracket_order(ib, mock_trade)

        mock_remove.assert_called_once_with(333)
        ib.cancelOrder.assert_called_once()

    def test_generic_cancel_removes_pending_order(self):
        """cancel_order (generic) must also clean up pending_orders."""
        from core.executor import cancel_order

        mock_trade = _make_mock_trade(444, "ALSO_GONE")
        ib = MagicMock()

        with patch("core.executor.remove_pending_order") as mock_remove:
            cancel_order(ib, mock_trade)

        mock_remove.assert_called_once_with(444)


class TestFillCleansDB:
    def test_fill_removes_pending_order(self):
        """Entry order fill must remove the pending_orders DB record."""
        from core.executor import setup_fill_handler
        from core.models import Action

        sig = MagicMock()
        sig.ticker = "FILLED"
        sig.action = Action.BUY
        sig.indicator_values = {"sector": "Tech"}
        sig.stop_loss = 90.0
        sig.take_profit = 110.0
        sig.trade_type = MagicMock(value="day")

        # Use a list to capture the registered callback
        callbacks = []
        ib = MagicMock()
        ib.orderStatusEvent.__iadd__ = lambda self, cb: callbacks.append(cb) or self

        setup_fill_handler(ib, sig, quantity=100)
        assert len(callbacks) == 1
        on_order_status = callbacks[0]

        # Simulate a fill event
        filled_trade = MagicMock()
        filled_trade.orderStatus.status = "Filled"
        filled_trade.orderStatus.avgFillPrice = 100.0
        filled_trade.orderStatus.filled = 100
        filled_trade.order.action = "BUY"
        filled_trade.order.orderType = "LMT"
        filled_trade.order.permId = 555

        with patch("core.executor.handle_fill"), \
             patch("core.executor.remove_pending_order") as mock_remove:
            on_order_status(filled_trade)

        mock_remove.assert_called_once_with(555)

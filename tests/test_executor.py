"""Tests for core/executor.py — focused on logic bugs around fill handling."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call

import pytest

from core.executor import handle_fill
from core.models import Action, Signal, TradeType, Position
from core.portfolio import init_db, get_open_positions, add_position


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test_exec.db"
    init_db(path)
    return path


def _signal(action: Action) -> Signal:
    return Signal(
        ticker="AAPL",
        action=action,
        confidence=80.0,
        entry_price=150.0,
        stop_loss=145.0 if action == Action.BUY else 155.0,
        take_profit=160.0 if action == Action.BUY else 140.0,
        reasoning="test",
        source="ai",
        exchange="SMART",
        trade_type=TradeType.DAY,
        timestamp=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
    )


def _make_mock_trade(ticker, order_id, parent_id=0, order_type="LMT"):
    """Build a mock ib_insync Trade with contract and order attributes."""
    trade = MagicMock()
    trade.contract.symbol = ticker
    trade.order.orderId = order_id
    trade.order.parentId = parent_id
    trade.order.orderType = order_type
    return trade


def _make_position(**kwargs) -> Position:
    defaults = dict(
        ticker="INTC",
        exchange="SMART",
        quantity=3,
        entry_price=64.0,
        entry_time=datetime(2026, 4, 14, 16, 30),
        stop_loss=61.5,
        take_profit=69.25,
        trade_type=TradeType.SWING,
        sector="Technology",
    )
    defaults.update(kwargs)
    return Position(**defaults)


class TestHandleFillDirection:
    """Economic correctness: short entries must store negative quantity.

    IBKR reports filled quantity as positive regardless of BUY/SELL.
    For a SELL parent (short opening), the database must store a negative
    quantity so:
      - P&L calculation `(exit - entry) * qty` works correctly
      - reconcile_positions matches IBKR's signed position
      - close_position_market picks the right closing side
    """

    def test_long_entry_stores_positive_quantity(self, db_path):
        sig = _signal(Action.BUY)
        handle_fill(sig, quantity=10, fill_price=150.0, db_path=db_path)
        positions = get_open_positions(db_path)
        assert len(positions) == 1
        assert positions[0].quantity == 10

    def test_short_entry_stores_negative_quantity(self, db_path):
        sig = _signal(Action.SELL)
        handle_fill(sig, quantity=10, fill_price=150.0, db_path=db_path)
        positions = get_open_positions(db_path)
        assert len(positions) == 1, "Short entry should create a position"
        assert positions[0].quantity == -10, (
            "Short entry must store negative quantity so P&L math works; "
            f"got {positions[0].quantity}"
        )

    def test_invalid_quantity_ignored(self, db_path):
        sig = _signal(Action.BUY)
        result = handle_fill(sig, quantity=0, fill_price=150.0, db_path=db_path)
        assert result is None
        assert get_open_positions(db_path) == []


class TestReattachExitHandlers:
    """Verify exit handler re-registration on startup.

    When the bot restarts, in-memory event handlers are lost. Open positions
    with live stop-loss/take-profit orders at IBKR need fresh handlers so
    fills update the DB and Telegram cache.
    """

    def test_attaches_one_handler_per_bracket(self, db_path):
        """One handler per bracket (not per child) — prevents duplicate DB closes."""
        from core.executor import reattach_exit_handlers

        pos = _make_position(ticker="INTC")
        add_position(pos, db_path)

        # Simulate IBKR open trades: parent entry + SL child + TP child
        parent = _make_mock_trade("INTC", order_id=100, parent_id=0, order_type="LMT")
        stop_child = _make_mock_trade("INTC", order_id=101, parent_id=100, order_type="STP")
        tp_child = _make_mock_trade("INTC", order_id=102, parent_id=100, order_type="LMT")

        ib = MagicMock()
        ib.openTrades.return_value = [parent, stop_child, tp_child]

        with patch("core.portfolio.get_open_positions", return_value=[pos]):
            with patch("core.executor.setup_exit_handler") as mock_setup:
                attached = reattach_exit_handlers(ib)

        assert attached == 1
        assert mock_setup.call_count == 1

    def test_no_positions_returns_zero(self):
        """No open positions → no handlers to attach."""
        from core.executor import reattach_exit_handlers

        ib = MagicMock()

        with patch("core.portfolio.get_open_positions", return_value=[]):
            attached = reattach_exit_handlers(ib)

        assert attached == 0
        ib.openTrades.assert_not_called()

    def test_skips_unrelated_tickers(self, db_path):
        """Orders for tickers not in DB are ignored."""
        from core.executor import reattach_exit_handlers

        pos = _make_position(ticker="INTC")
        add_position(pos, db_path)

        # IBKR has orders for AAPL (not in DB) and INTC (in DB)
        aapl_child = _make_mock_trade("AAPL", order_id=200, parent_id=199, order_type="STP")
        intc_parent = _make_mock_trade("INTC", order_id=100, parent_id=0, order_type="LMT")
        intc_child = _make_mock_trade("INTC", order_id=101, parent_id=100, order_type="STP")

        ib = MagicMock()
        ib.openTrades.return_value = [aapl_child, intc_parent, intc_child]

        with patch("core.portfolio.get_open_positions", return_value=[pos]):
            with patch("core.executor.setup_exit_handler") as mock_setup:
                attached = reattach_exit_handlers(ib)

        assert attached == 1
        # Only INTC child should get a handler
        sig_arg = mock_setup.call_args_list[0][0][1]
        assert sig_arg.ticker == "INTC"

    def test_skips_parent_orders(self, db_path):
        """Parent entry orders (parentId=0) should not get exit handlers."""
        from core.executor import reattach_exit_handlers

        pos = _make_position(ticker="INTC")

        # Only parent orders, no children
        parent = _make_mock_trade("INTC", order_id=100, parent_id=0, order_type="LMT")

        ib = MagicMock()
        ib.openTrades.return_value = [parent]

        with patch("core.portfolio.get_open_positions", return_value=[pos]):
            with patch("core.executor.setup_exit_handler") as mock_setup:
                attached = reattach_exit_handlers(ib)

        assert attached == 0
        mock_setup.assert_not_called()

    def test_skips_children_from_wrong_parent(self, db_path):
        """Child orders whose parentId doesn't match the known parent are skipped.

        This prevents cross-bracket interference when the same ticker has been
        re-entered and the old bracket is still lingering.
        """
        from core.executor import reattach_exit_handlers

        pos = _make_position(ticker="INTC")

        # Parent order 100 is the current bracket
        parent = _make_mock_trade("INTC", order_id=100, parent_id=0, order_type="LMT")
        # This child belongs to a DIFFERENT parent (old bracket, orderId=50)
        stale_child = _make_mock_trade("INTC", order_id=51, parent_id=50, order_type="STP")
        # This child belongs to the current parent
        current_child = _make_mock_trade("INTC", order_id=101, parent_id=100, order_type="STP")

        ib = MagicMock()
        ib.openTrades.return_value = [parent, stale_child, current_child]

        with patch("core.portfolio.get_open_positions", return_value=[pos]):
            with patch("core.executor.setup_exit_handler") as mock_setup:
                attached = reattach_exit_handlers(ib)

        # Only the current child should be attached, not the stale one
        assert attached == 1

    def test_passes_parent_order_for_precise_matching(self, db_path):
        """The parent Order object is passed to setup_exit_handler for permId matching."""
        from core.executor import reattach_exit_handlers

        pos = _make_position(ticker="INTC")

        parent = _make_mock_trade("INTC", order_id=100, parent_id=0, order_type="LMT")
        child = _make_mock_trade("INTC", order_id=101, parent_id=100, order_type="STP")

        ib = MagicMock()
        ib.openTrades.return_value = [parent, child]

        with patch("core.portfolio.get_open_positions", return_value=[pos]):
            with patch("core.executor.setup_exit_handler") as mock_setup:
                reattach_exit_handlers(ib)

        # Verify parent_order kwarg was passed
        _, kwargs = mock_setup.call_args
        assert kwargs["parent_order"] is parent.order

    def test_builds_correct_signal_for_long_position(self, db_path):
        """Signal built from a long position should have action=BUY."""
        from core.executor import reattach_exit_handlers

        pos = _make_position(ticker="INTC", quantity=3, entry_price=64.0,
                             stop_loss=61.5, take_profit=69.25)

        parent = _make_mock_trade("INTC", order_id=100, parent_id=0)
        child = _make_mock_trade("INTC", order_id=101, parent_id=100, order_type="STP")

        ib = MagicMock()
        ib.openTrades.return_value = [parent, child]

        with patch("core.portfolio.get_open_positions", return_value=[pos]):
            with patch("core.executor.setup_exit_handler") as mock_setup:
                reattach_exit_handlers(ib)

        signal = mock_setup.call_args[0][1]
        assert signal.ticker == "INTC"
        assert signal.action == Action.BUY
        assert signal.entry_price == 64.0
        assert signal.stop_loss == 61.5
        assert signal.take_profit == 69.25

    def test_builds_correct_signal_for_short_position(self, db_path):
        """Signal built from a short position should have action=SELL."""
        from core.executor import reattach_exit_handlers

        pos = _make_position(ticker="TSLA", quantity=-5, entry_price=200.0,
                             stop_loss=210.0, take_profit=180.0)

        parent = _make_mock_trade("TSLA", order_id=100, parent_id=0)
        child = _make_mock_trade("TSLA", order_id=101, parent_id=100, order_type="STP")

        ib = MagicMock()
        ib.openTrades.return_value = [parent, child]

        with patch("core.portfolio.get_open_positions", return_value=[pos]):
            with patch("core.executor.setup_exit_handler") as mock_setup:
                reattach_exit_handlers(ib)

        signal = mock_setup.call_args[0][1]
        assert signal.action == Action.SELL

    def test_multiple_tickers(self, db_path):
        """One handler per bracket (not per child order) for each ticker."""
        from core.executor import reattach_exit_handlers

        pos_intc = _make_position(ticker="INTC")
        pos_aapl = _make_position(ticker="AAPL", entry_price=180.0,
                                  stop_loss=175.0, take_profit=190.0)

        intc_parent = _make_mock_trade("INTC", order_id=100, parent_id=0)
        intc_sl = _make_mock_trade("INTC", order_id=101, parent_id=100, order_type="STP")
        aapl_parent = _make_mock_trade("AAPL", order_id=200, parent_id=0)
        aapl_sl = _make_mock_trade("AAPL", order_id=201, parent_id=200, order_type="STP")
        aapl_tp = _make_mock_trade("AAPL", order_id=202, parent_id=200, order_type="LMT")

        ib = MagicMock()
        ib.openTrades.return_value = [intc_parent, intc_sl, aapl_parent, aapl_sl, aapl_tp]

        with patch("core.portfolio.get_open_positions", return_value=[pos_intc, pos_aapl]):
            with patch("core.executor.setup_exit_handler") as mock_setup:
                attached = reattach_exit_handlers(ib)

        # One handler per bracket, not per child order
        assert attached == 2  # 1 for INTC bracket + 1 for AAPL bracket
        tickers = [c[0][1].ticker for c in mock_setup.call_args_list]
        assert tickers.count("INTC") == 1
        assert tickers.count("AAPL") == 1

    def test_one_handler_per_bracket_not_per_child(self, db_path):
        """Two child orders (SL+TP) for same bracket should create ONE handler."""
        from core.executor import reattach_exit_handlers

        pos = _make_position(ticker="INTC")

        parent = _make_mock_trade("INTC", order_id=100, parent_id=0, order_type="LMT")
        sl_child = _make_mock_trade("INTC", order_id=101, parent_id=100, order_type="STP")
        tp_child = _make_mock_trade("INTC", order_id=102, parent_id=100, order_type="LMT")

        ib = MagicMock()
        ib.openTrades.return_value = [parent, sl_child, tp_child]

        with patch("core.portfolio.get_open_positions", return_value=[pos]):
            with patch("core.executor.setup_exit_handler") as mock_setup:
                attached = reattach_exit_handlers(ib)

        # Only 1 handler should be attached, not 2
        assert attached == 1
        assert mock_setup.call_count == 1


class TestCloseAllDayTrades:
    """close_all_day_trades must cancel bracket orders and close DB positions."""

    def test_cancels_bracket_orders_for_day_trades(self):
        """After closing via market order, bracket TP/SL orders must be cancelled."""
        from core.executor import close_all_day_trades

        pos = _make_position(ticker="AAPL", quantity=10, trade_type=TradeType.DAY,
                             entry_price=150.0, stop_loss=145.0, take_profit=160.0)

        # Simulate bracket orders at IBKR — parent already filled, children still open
        sl_child = _make_mock_trade("AAPL", order_id=101, parent_id=100, order_type="STP")
        tp_child = _make_mock_trade("AAPL", order_id=102, parent_id=100, order_type="LMT")

        ib = MagicMock()
        ib.openTrades.return_value = [sl_child, tp_child]

        # Mock market order placement
        mock_trade = MagicMock()
        mock_trade.orderStatus.status = "Filled"
        mock_trade.orderStatus.filled = 10
        mock_trade.orderStatus.remaining = 0
        mock_trade.orderStatus.avgFillPrice = 155.0
        mock_trade.order.orderId = 999

        with patch("core.executor.place_market_order", return_value=mock_trade):
            close_all_day_trades(ib, [pos], dry_run=False)

        # TP/SL children should have been cancelled (not just unfilled parents)
        assert ib.cancelOrder.call_count >= 2

    def test_closes_db_position_for_day_trades(self, db_path):
        """Closing day trades must record the close in the portfolio DB."""
        from core.executor import close_all_day_trades

        pos = _make_position(ticker="AAPL", quantity=10, trade_type=TradeType.DAY,
                             entry_price=150.0, stop_loss=145.0, take_profit=160.0)
        add_position(pos, db_path)

        ib = MagicMock()
        ib.openTrades.return_value = []

        mock_trade = MagicMock()
        mock_trade.orderStatus.status = "Filled"
        mock_trade.orderStatus.filled = 10
        mock_trade.orderStatus.remaining = 0
        mock_trade.orderStatus.avgFillPrice = 155.0
        mock_trade.order.orderId = 999

        with patch("core.executor.place_market_order", return_value=mock_trade), \
             patch("core.executor.db_close_position", wraps=lambda t, p, **kw: None) as mock_close:
            close_all_day_trades(ib, [pos], dry_run=False)

        mock_close.assert_called_once()

    def test_skips_swing_trades(self):
        """Swing trades must NOT be closed."""
        from core.executor import close_all_day_trades

        swing = _make_position(ticker="MSFT", quantity=5, trade_type=TradeType.SWING)

        ib = MagicMock()
        ib.openTrades.return_value = []

        with patch("core.executor.place_market_order") as mock_place:
            close_all_day_trades(ib, [swing], dry_run=False)

        mock_place.assert_not_called()

    def test_waits_for_cancels_before_placing_market_close(self):
        """Cancels must be confirmed before placing close orders.

        IBKR cancel is asynchronous: the cancel request returns immediately
        but the order may still fill before the broker processes it. If we
        place the market close at the same time as the cancel request, a
        filling SL child plus our market order can doubly-close the position
        (net flat → net short). The code must poll openTrades until the
        cancels clear (or the child fills) before transmitting the close.
        """
        from core.executor import close_all_day_trades

        pos = _make_position(ticker="AAPL", quantity=10, trade_type=TradeType.DAY,
                             entry_price=150.0, stop_loss=145.0, take_profit=160.0)

        sl_child = _make_mock_trade("AAPL", order_id=101, parent_id=100, order_type="STP")
        sl_child.orderStatus.status = "Submitted"
        tp_child = _make_mock_trade("AAPL", order_id=102, parent_id=100, order_type="LMT")
        tp_child.orderStatus.status = "Submitted"

        ib = MagicMock()
        # Track order of ib calls: cancelOrder, sleep, openTrades, place_market_order
        call_log = []

        # Snapshot openTrades — first call (pre-cancel) returns the children,
        # second call (post-sleep) returns empty (cancels took effect)
        trades_state = {"iter": 0}

        def open_trades_side_effect():
            trades_state["iter"] += 1
            call_log.append(("openTrades", trades_state["iter"]))
            if trades_state["iter"] == 1:
                return [sl_child, tp_child]
            return []

        ib.openTrades.side_effect = open_trades_side_effect

        def cancel_side_effect(order):
            call_log.append(("cancelOrder", order.orderId))

        ib.cancelOrder.side_effect = cancel_side_effect

        def sleep_side_effect(_):
            call_log.append(("sleep",))

        ib.sleep.side_effect = sleep_side_effect

        mock_close_trade = MagicMock()
        mock_close_trade.orderStatus.status = "Filled"
        mock_close_trade.orderStatus.filled = 10
        mock_close_trade.orderStatus.remaining = 0
        mock_close_trade.orderStatus.avgFillPrice = 155.0
        mock_close_trade.order.orderId = 999

        def place_side_effect(*a, **kw):
            call_log.append(("place_market_order",))
            return mock_close_trade

        with patch("core.executor.place_market_order", side_effect=place_side_effect), \
             patch("core.executor.db_close_position"):
            close_all_day_trades(ib, [pos], dry_run=False)

        # Reconstruct event order: every cancel must precede every place_market_order,
        # and at least one sleep must intervene between last cancel and first place.
        cancel_indices = [i for i, e in enumerate(call_log) if e[0] == "cancelOrder"]
        place_indices = [i for i, e in enumerate(call_log) if e[0] == "place_market_order"]
        sleep_indices = [i for i, e in enumerate(call_log) if e[0] == "sleep"]

        assert cancel_indices, "expected cancels to be issued"
        assert place_indices, "expected a market close to be placed"
        last_cancel = cancel_indices[-1]
        first_place = place_indices[0]
        assert last_cancel < first_place, "market close must not be placed before cancels"
        assert any(last_cancel < s < first_place for s in sleep_indices), (
            "at least one ib.sleep() must occur between the cancel batch and "
            "the first market-close order so IBKR has time to ack the cancels"
        )


def _make_mock_ibkr_position(ticker, quantity, avg_cost, exchange="NASDAQ"):
    """Build a mock ib_insync Position (from ib.positions())."""
    pos = MagicMock()
    pos.contract.symbol = ticker
    pos.contract.primaryExchange = exchange
    pos.contract.exchange = "SMART"
    pos.position = float(quantity)
    pos.avgCost = avg_cost
    return pos


class TestImportIbkrPositions:
    """Verify import of IBKR positions that exist at broker but not in DB.

    When the bot's DB is cleared or a position was opened outside the bot,
    import_ibkr_positions should create DB records from IBKR data and extract
    stop-loss/take-profit from open bracket orders.
    """

    def test_imports_position_with_bracket_orders(self, db_path):
        """Full bracket: position + SL + TP child orders → complete DB record."""
        from core.executor import import_ibkr_positions

        ibkr_pos = _make_mock_ibkr_position("INTC", 3, 64.33)
        parent = _make_mock_trade("INTC", order_id=100, parent_id=0, order_type="LMT")
        sl_child = _make_mock_trade("INTC", order_id=101, parent_id=100, order_type="STP")
        sl_child.order.auxPrice = 61.5
        tp_child = _make_mock_trade("INTC", order_id=102, parent_id=100, order_type="LMT")
        tp_child.order.lmtPrice = 69.25

        ib = MagicMock()
        ib.positions.return_value = [ibkr_pos]
        ib.openTrades.return_value = [parent, sl_child, tp_child]

        with patch("core.executor.add_position", wraps=lambda p: add_position(p, db_path)):
            imported = import_ibkr_positions(ib, ["INTC"])

        assert imported == ["INTC"]
        positions = get_open_positions(db_path)
        assert len(positions) == 1
        pos = positions[0]
        assert pos.ticker == "INTC"
        assert pos.quantity == 3
        assert pos.entry_price == 64.33
        assert pos.stop_loss == 61.5
        assert pos.take_profit == 69.25
        assert pos.exchange == "NASDAQ"

    def test_imports_position_without_bracket_orders(self, db_path):
        """Position with no open orders → SL/TP default to 0."""
        from core.executor import import_ibkr_positions

        ibkr_pos = _make_mock_ibkr_position("AAPL", 10, 180.0)

        ib = MagicMock()
        ib.positions.return_value = [ibkr_pos]
        ib.openTrades.return_value = []

        with patch("core.executor.add_position", wraps=lambda p: add_position(p, db_path)):
            imported = import_ibkr_positions(ib, ["AAPL"])

        assert imported == ["AAPL"]
        positions = get_open_positions(db_path)
        assert len(positions) == 1
        assert positions[0].stop_loss == 0.0
        assert positions[0].take_profit == 0.0

    def test_empty_orphan_list_returns_empty(self):
        """No orphaned tickers → no work done."""
        from core.executor import import_ibkr_positions

        ib = MagicMock()
        result = import_ibkr_positions(ib, [])

        assert result == []
        ib.positions.assert_not_called()

    def test_skips_ticker_not_in_ibkr_positions(self, db_path):
        """If IBKR doesn't actually hold the ticker, skip it."""
        from core.executor import import_ibkr_positions

        ib = MagicMock()
        ib.positions.return_value = []  # IBKR has nothing
        ib.openTrades.return_value = []

        with patch("core.executor.add_position", wraps=lambda p: add_position(p, db_path)):
            imported = import_ibkr_positions(ib, ["GHOST"])

        assert imported == []
        assert get_open_positions(db_path) == []

    def test_imports_multiple_tickers(self, db_path):
        """Multiple orphaned tickers are all imported."""
        from core.executor import import_ibkr_positions

        intc = _make_mock_ibkr_position("INTC", 3, 64.0)
        aapl = _make_mock_ibkr_position("AAPL", 5, 180.0, exchange="NASDAQ")

        ib = MagicMock()
        ib.positions.return_value = [intc, aapl]
        ib.openTrades.return_value = []

        with patch("core.executor.add_position", wraps=lambda p: add_position(p, db_path)):
            imported = import_ibkr_positions(ib, ["INTC", "AAPL"])

        assert sorted(imported) == ["AAPL", "INTC"]
        assert len(get_open_positions(db_path)) == 2

    def test_short_position_imported_correctly(self, db_path):
        """Negative IBKR position (short) is stored with negative quantity."""
        from core.executor import import_ibkr_positions

        short_pos = _make_mock_ibkr_position("TSLA", -5, 200.0)

        ib = MagicMock()
        ib.positions.return_value = [short_pos]
        ib.openTrades.return_value = []

        with patch("core.executor.add_position", wraps=lambda p: add_position(p, db_path)):
            imported = import_ibkr_positions(ib, ["TSLA"])

        assert imported == ["TSLA"]
        pos = get_open_positions(db_path)[0]
        assert pos.quantity == -5


class TestSetupFillHandlerRace:
    """Fast fills can arrive before setup_fill_handler registers its listener.

    place_order() polls IBKR up to 3 s for permId before returning; during that
    window the order may already fill. ib_insync does not replay past events,
    so if the handler is attached after the fill has already been recorded on
    the Trade object, handle_fill is never called and the position is silently
    orphaned on IBKR (detected only at next reconciliation).

    The fix: on registration, inspect the parent trade's current status. If it
    is already "Filled", invoke the fill path immediately.
    """

    def test_already_filled_trade_invokes_handler_immediately(self, db_path):
        """A Trade that arrives already-Filled must still produce a position."""
        from core.executor import setup_fill_handler

        sig = _signal(Action.BUY)
        ib = MagicMock()

        parent_order = MagicMock()
        parent_order.permId = 42
        parent_order.parentId = 0
        parent_order.action = "BUY"
        parent_order.orderId = 100

        parent_trade = MagicMock()
        parent_trade.order = parent_order
        parent_trade.contract.symbol = sig.ticker
        parent_trade.orderStatus.status = "Filled"
        parent_trade.orderStatus.avgFillPrice = 150.5
        parent_trade.orderStatus.filled = 10

        with patch("core.executor.handle_fill") as mock_handle:
            setup_fill_handler(
                ib, sig, quantity=10,
                parent_order=parent_order,
                parent_trade=parent_trade,
            )

        mock_handle.assert_called_once()
        args = mock_handle.call_args.args
        assert args[0] is sig
        assert args[1] == 10
        assert args[2] == 150.5

    def test_unfilled_trade_does_not_fire_at_registration(self, db_path):
        """A Submitted trade must wait for the event, not fire on registration."""
        from core.executor import setup_fill_handler

        sig = _signal(Action.BUY)
        ib = MagicMock()

        parent_order = MagicMock()
        parent_order.permId = 43
        parent_order.parentId = 0
        parent_order.action = "BUY"

        parent_trade = MagicMock()
        parent_trade.order = parent_order
        parent_trade.contract.symbol = sig.ticker
        parent_trade.orderStatus.status = "Submitted"
        parent_trade.orderStatus.avgFillPrice = 0.0
        parent_trade.orderStatus.filled = 0

        with patch("core.executor.handle_fill") as mock_handle:
            setup_fill_handler(
                ib, sig, quantity=10,
                parent_order=parent_order,
                parent_trade=parent_trade,
            )

        mock_handle.assert_not_called()

    def test_already_filled_does_not_double_fire_on_event(self, db_path):
        """If registration fires handle_fill, a subsequent event must not duplicate."""
        from core.executor import setup_fill_handler

        sig = _signal(Action.BUY)
        ib = MagicMock()
        # ib.orderStatusEvent must support += (handler registration)
        events = []

        class FakeEvent:
            def __iadd__(self, h):
                events.append(h)
                return self
            def __isub__(self, h):
                if h in events:
                    events.remove(h)
                return self

        ib.orderStatusEvent = FakeEvent()

        parent_order = MagicMock()
        parent_order.permId = 44
        parent_order.parentId = 0
        parent_order.action = "BUY"
        parent_order.orderId = 101

        parent_trade = MagicMock()
        parent_trade.order = parent_order
        parent_trade.contract.symbol = sig.ticker
        parent_trade.orderStatus.status = "Filled"
        parent_trade.orderStatus.avgFillPrice = 150.5
        parent_trade.orderStatus.filled = 10

        with patch("core.executor.handle_fill") as mock_handle:
            setup_fill_handler(
                ib, sig, quantity=10,
                parent_order=parent_order,
                parent_trade=parent_trade,
            )
            # Simulate the event firing after registration
            for h in list(events):
                h(parent_trade)

        assert mock_handle.call_count == 1, (
            "handle_fill must fire exactly once even if a late orderStatusEvent "
            "arrives after the already-filled snapshot was handled at registration"
        )


class TestExitHandlerParentMatching:
    """Exit handlers without a known parent_order_id must not fire on arbitrary
    child orders — doing so risks cross-bracket interference when a ticker is
    re-entered in the same session and a stale handler observes a fresh fill.
    """

    def _fake_event(self):
        handlers = []

        class FakeEvent:
            def __iadd__(self, h):
                handlers.append(h)
                return self
            def __isub__(self, h):
                if h in handlers:
                    handlers.remove(h)
                return self
        return FakeEvent(), handlers

    def test_no_parent_order_refuses_to_fire(self, db_path):
        """When parent_order is None (reattach couldn't resolve it), the handler
        must NOT fire on a child-order fill that happens to match by ticker.
        Otherwise a stale handler attached during startup can close a fresh
        position if the same ticker is later re-entered.
        """
        from core.executor import setup_exit_handler

        sig = _signal(Action.BUY)
        ib = MagicMock()
        event, handlers = self._fake_event()
        ib.orderStatusEvent = event

        # A child-order fill for the same ticker, from a different bracket
        stray_child = MagicMock()
        stray_child.order.orderType = "STP"
        stray_child.order.parentId = 999  # some parent we don't know about
        stray_child.order.action = "SELL"
        stray_child.orderStatus.status = "Filled"
        stray_child.orderStatus.avgFillPrice = 148.0
        stray_child.contract.symbol = sig.ticker

        with patch("core.executor.db_close_position") as mock_close:
            mock_close.return_value = MagicMock(pnl=0.0, pnl_pct=0.0)

            # Attach with parent_order=None (the buggy reattach path)
            setup_exit_handler(ib, sig, on_exit=None, parent_order=None)

            # Fire the event
            for h in list(handlers):
                h(stray_child)

            assert mock_close.call_count == 0, (
                "Exit handler without parent_order must not trigger db close "
                "on a child-order fill — otherwise a stale handler from a "
                "previous bracket can close a fresh re-entry of the same ticker"
            )

    def test_with_parent_order_fires_on_matching_child(self, db_path):
        """Sanity: with a valid parent_order, the handler DOES fire on its child."""
        from core.executor import setup_exit_handler

        sig = _signal(Action.BUY)
        ib = MagicMock()
        event, handlers = self._fake_event()
        ib.orderStatusEvent = event

        parent = MagicMock()
        parent.orderId = 100

        child = MagicMock()
        child.order.orderType = "STP"
        child.order.parentId = 100  # matches parent.orderId
        child.order.action = "SELL"
        child.orderStatus.status = "Filled"
        child.orderStatus.avgFillPrice = 145.0
        child.contract.symbol = sig.ticker

        with patch("core.executor.db_close_position") as mock_close:
            mock_close.return_value = MagicMock(pnl=0.0, pnl_pct=0.0)
            setup_exit_handler(ib, sig, on_exit=None, parent_order=parent)
            for h in list(handlers):
                h(child)

            assert mock_close.call_count == 1

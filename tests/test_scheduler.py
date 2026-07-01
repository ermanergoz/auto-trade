"""Tests for core/scheduler.py — INVERTED pipeline (screen -> risk -> gate -> execute).

Verifies that mechanical risk runs FIRST over the screener candidates and the
LLM gate runs LAST and can only SUBTRACT (veto/warn/abstain), off the asyncio
loop via run_in_executor + ib.sleep(0.1). analyze_batch is gone from the live
path (LLM-01, LLM-05).
"""

import asyncio
import time
from unittest.mock import patch, MagicMock, call

import pytest

from ib_insync import util

from core.risk import RiskResult
from core.gate import GateResult, Verdict
from tests.conftest import make_signal as _make_signal, make_position as _make_position


# Patches common to all pipeline tests.
_PATCHES = [
    "core.scheduler.notify_risk_results",
    "core.scheduler.setup_fill_handler",
    "core.scheduler.setup_exit_handler",
    "core.scheduler.notify_trade",
    "core.scheduler.place_order",
    "core.scheduler.place_market_order",
    "core.scheduler.record_signal",
    "core.scheduler.evaluate",
    "core.scheduler.get_open_positions",
    "core.scheduler.get_daily_pnl",
    "core.scheduler.update_portfolio_data",
    "core.scheduler.screen_stocks",
    "core.scheduler.gate_signal",
    "core.scheduler.get_earnings_date",
    "core.scheduler.log_verdict",
    "core.scheduler.notify_veto",
    "core.scheduler.notify_warn",
    "core.scheduler.notify_gate_halt",
    "core.scheduler.build_universe",
    "core.scheduler.get_tickers_for_market",
    "core.scheduler.update_status",
    "core.scheduler.ensure_connected",
    "core.scheduler.get_news",
    "core.scheduler.get_macro_news",
    "core.scheduler.get_active_markets",
    "core.scheduler.get_pending_buy_reserve",
]


def _ensure_event_loop():
    """Return an open asyncio loop for this thread, recreating a closed one.

    The scheduler's off-loop bridge uses ``asyncio.get_event_loop()``; the test
    harness must hand it the SAME open loop that the pumping ``ib.sleep`` runs.
    """
    try:
        loop = asyncio.get_event_loop_policy().get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


def _make_ib():
    """A mock IB whose ``.sleep`` PUMPS the event loop, mirroring real ib.sleep.

    Real ``ib.sleep`` runs ``util.run(asyncio.sleep(...))`` which keeps the loop
    live so ``run_in_executor`` futures resolve (fills/disconnects are serviced).
    A plain ``MagicMock`` would no-op and the gate loop's ``while not fut.done():
    ib.sleep(0.1)`` would spin forever. A pump budget converts any accidental
    non-resolving future into a fast, clearly-labelled failure instead of a hang.
    """
    _ensure_event_loop()
    ib = MagicMock()
    state = {"pumps": 0}

    def _sleep(_secs=0.0):
        state["pumps"] += 1
        if state["pumps"] > 4000:
            raise RuntimeError(
                "ib.sleep pump budget exceeded — off-loop gate future never resolved"
            )
        util.run(asyncio.sleep(0.005))

    ib.sleep.side_effect = _sleep
    return ib


def _make_stock_data(tickers):
    """Build a minimal stock_data dict for _fetch_market_data mock."""
    import pandas as pd

    df = pd.DataFrame({"close": [100.0], "open": [99.0], "high": [101.0], "low": [98.0], "volume": [1000000]})
    return {t: ("SMART", df) for t in tickers}


def _setup_mocks(
    mocks,
    signals,
    risk_approved=None,
    positions_sequence=None,
    sectors=None,
    is_exit_flags=None,
):
    """Configure mocks for a standard INVERTED pipeline run.

    ``screen_stocks`` returns the candidate signals; the scheduler's risk loop
    iterates them directly (no analyze_batch / on_signal callback anymore). The
    gate runs LAST — by default it clears every entry (OK, live provider) so
    risk-approved buys stand. Tests override ``gate_signal`` to veto/warn.
    """
    if risk_approved is None:
        risk_approved = [True] * len(signals)
    if positions_sequence is None:
        positions_sequence = [[]] * (len(signals) + 1)
    if sectors is None:
        sectors = [""] * len(signals)
    if is_exit_flags is None:
        is_exit_flags = [False] * len(signals)

    mocks["get_active_markets"].return_value = ["US"]
    mocks["get_daily_pnl"].return_value = 0.0
    mocks["get_open_positions"].side_effect = list(positions_sequence)

    from core.models import StockInfo

    stock_infos = [
        StockInfo(s.ticker, s.exchange, sector, 0.0, 0.0)
        for s, sector in zip(signals, sectors)
    ]
    mocks["get_tickers_for_market"].return_value = stock_infos
    mocks["build_universe"].return_value = {}
    mocks["screen_stocks"].return_value = signals
    mocks["get_news"].return_value = []
    mocks["get_macro_news"].return_value = []

    # Gate defaults: clear every entry (OK from a live provider) so the buy
    # stands; earnings unknown (D-05 abstain); verdict persistence is a no-op.
    mocks["gate_signal"].return_value = GateResult(Verdict.OK, None, "", "gemini")
    mocks["get_earnings_date"].return_value = None

    risk_results = []
    for approved, is_exit in zip(risk_approved, is_exit_flags):
        if approved:
            risk_results.append(RiskResult(
                approved=True, reasons=[], position_size=10, is_exit=is_exit,
            ))
        else:
            risk_results.append(RiskResult(
                approved=False, reasons=["test rejection"], is_exit=is_exit,
            ))
    mocks["evaluate"].side_effect = risk_results

    mocks["place_order"].return_value = [MagicMock()]
    mocks["place_market_order"].return_value = MagicMock()
    # Default: nothing reserved. Cash-gate tests override this.
    mocks["get_pending_buy_reserve"].return_value = 0.0


def _run_cycle(mocks, account=None):
    """Run a scan cycle with a fake IB connection.

    ``account`` lets a test override the mocked IBKR account summary dict.
    Default includes TotalCashValue so the cash-reserve gate in _on_signal
    does not trip on tests that don't care about cash.
    """
    from core.scheduler import run_scan_cycle

    ib = _make_ib()
    if account is None:
        account = {"NetLiquidation": 100_000, "TotalCashValue": 100_000}

    with patch("core.scheduler._fetch_market_data") as mock_fetch, \
         patch("core.connection.get_account_summary", return_value=account), \
         patch("core.scheduler.minutes_to_close", return_value=999), \
         patch("core.scheduler.get_trades", return_value=[]):
        mock_fetch.return_value = _make_stock_data(
            [s.ticker for s in mocks["screen_stocks"].return_value],
        )
        return run_scan_cycle(ib, ["US"])


class TestStreamingPipeline:
    """Verify that AI-approved signals are streamed to risk check via callback."""

    @pytest.fixture(autouse=True)
    def _patch_all(self):
        patchers = {name.split(".")[-1]: patch(name) for name in _PATCHES}
        self.m = {}
        for key, p in patchers.items():
            self.m[key] = p.start()
        yield
        for p in patchers.values():
            p.stop()

    def test_gate_invoked_for_risk_approved_entry(self):
        """The LLM gate must run (LAST) on a risk-approved entry — analyze_batch
        is gone; the gate replaces it as the only LLM call in the live path."""
        sig = _make_signal(ticker="TSLA")
        _setup_mocks(self.m, [sig], risk_approved=[True])

        _run_cycle(self.m)

        self.m["gate_signal"].assert_called_once()

    def test_risk_runs_first_over_screener_candidates(self):
        """evaluate() must be called directly over the screener candidates,
        BEFORE (and independent of) the LLM gate."""
        sig = _make_signal(ticker="TSLA")
        _setup_mocks(self.m, [sig])

        _run_cycle(self.m)

        self.m["evaluate"].assert_called_once()
        assert self.m["evaluate"].call_args[0][0].ticker == "TSLA"

    def test_gate_veto_removes_risk_approved_buy(self):
        """A gate VETO must drop a risk-approved entry before execution (LLM-01)."""
        sig = _make_signal(ticker="VETOED")
        _setup_mocks(self.m, [sig], risk_approved=[True])
        self.m["gate_signal"].return_value = GateResult(
            Verdict.VETO, None, "confirmed earnings in hold window", "deterministic",
        )

        _run_cycle(self.m)

        self.m["place_order"].assert_not_called()
        self.m["notify_veto"].assert_called_once()

    def test_gate_fail_closed_blocks_entry(self):
        """provider=='none' (both providers exhausted) blocks the entry and
        alerts — never a silent fail-open (D-02/D-07)."""
        sig = _make_signal(ticker="NOGATE")
        _setup_mocks(self.m, [sig], risk_approved=[True])
        self.m["gate_signal"].return_value = GateResult(
            Verdict.INSUFFICIENT_DATA, None, "gate unavailable — fail closed", "none",
        )

        _run_cycle(self.m)

        self.m["place_order"].assert_not_called()
        self.m["notify_gate_halt"].assert_called_once()

    def test_gate_warn_lets_buy_stand(self):
        """WARN is notify-only — the buy still stands (D-01)."""
        sig = _make_signal(ticker="FLAGGED")
        _setup_mocks(self.m, [sig], risk_approved=[True])
        self.m["gate_signal"].return_value = GateResult(
            Verdict.WARN, "some flagged headline", "flagged", "gemini",
        )

        _run_cycle(self.m)

        self.m["place_order"].assert_called_once()
        self.m["notify_warn"].assert_called_once()

    def test_every_verdict_is_persisted(self):
        """LLM-08: every gate verdict must be persisted via log_verdict."""
        sig = _make_signal(ticker="LOGGED")
        _setup_mocks(self.m, [sig], risk_approved=[True])

        _run_cycle(self.m)

        self.m["log_verdict"].assert_called_once()
        kwargs = self.m["log_verdict"].call_args.kwargs
        assert kwargs["ticker"] == "LOGGED"
        assert kwargs["verdict"] == "OK"

    def test_risk_approved_signal_gets_order(self):
        sig = _make_signal(ticker="GOOG")
        _setup_mocks(self.m, [sig], risk_approved=[True])

        _run_cycle(self.m)

        self.m["place_order"].assert_called_once()
        assert self.m["place_order"].call_args[0][1].ticker == "GOOG"

    def test_risk_rejected_signal_skips_execution(self):
        sig = _make_signal(ticker="BAD")
        _setup_mocks(self.m, [sig], risk_approved=[False])

        _run_cycle(self.m)

        self.m["place_order"].assert_not_called()

    def test_positions_refreshed_after_trade(self):
        sig1 = _make_signal(ticker="AAPL")
        sig2 = _make_signal(ticker="MSFT")
        pos_after_trade = [_make_position(ticker="AAPL")]
        # get_open_positions call sequence:
        #   0: initial fetch (line 134)
        #   1: refresh after sig1 trade -> returns pos_after_trade
        #   2: refresh after sig2 trade
        _setup_mocks(
            self.m,
            [sig1, sig2],
            risk_approved=[True, True],
            positions_sequence=[[], pos_after_trade, pos_after_trade],
        )

        _run_cycle(self.m)

        assert self.m["evaluate"].call_count == 2
        # Second evaluate call should receive positions refreshed after sig1 trade
        second_call_positions = self.m["evaluate"].call_args_list[1][0][1]
        assert len(second_call_positions) == 1
        assert second_call_positions[0].ticker == "AAPL"

    def test_skips_place_order_when_inside_close_window(self):
        """A scan that started before the close window must not place orders
        once wall-clock time has crossed into the close window — otherwise
        close_all_day_trades would immediately flatten the new position.
        """
        sig = _make_signal(ticker="LATE")
        _setup_mocks(self.m, [sig], risk_approved=[True])

        # When the scan-cycle starts, minutes_to_close = 20 (> CLOSE_MINUTES_BEFORE=15)
        # so the scan proceeds. When _on_signal runs for this signal, time has
        # advanced into the window (10 minutes to close). The callback must
        # see this and skip place_order.
        from core.scheduler import run_scan_cycle

        ib = MagicMock()
        from core.models import StockInfo

        # Build candidate list
        self.m["get_tickers_for_market"].return_value = [
            StockInfo("LATE", "SMART", "Technology", 0.0, 0.0),
        ]

        # Sequence: [scan_start=20, _on_signal=10, ...]
        close_sequence = iter([20, 10, 10, 10, 10, 10])
        with patch("core.scheduler._fetch_market_data") as mock_fetch, \
             patch("core.connection.get_account_summary", return_value={"NetLiquidation": 100_000}), \
             patch("core.scheduler.minutes_to_close", side_effect=lambda m: next(close_sequence)), \
             patch("core.scheduler.get_trades", return_value=[]):
            mock_fetch.return_value = _make_stock_data(["LATE"])
            run_scan_cycle(ib, ["US"])

        # place_order must not be called because the signal arrived inside the close window
        self.m["place_order"].assert_not_called()

    def test_pending_signal_included_in_next_risk_check(self):
        """A signal just approved + placed must appear in the next risk check's
        position list, even if the DB refresh (get_open_positions) hasn't seen
        the fill yet. Without this, two rapid-fire AI approvals can both pass
        the max-positions/sector-concentration gates against a stale view."""
        sig1 = _make_signal(ticker="AAPL")
        sig2 = _make_signal(ticker="MSFT")
        # Simulate slow fill: neither fill has been recorded in DB before the
        # second risk check runs. get_open_positions returns empty both times.
        _setup_mocks(
            self.m,
            [sig1, sig2],
            risk_approved=[True, True],
            positions_sequence=[[], [], []],
        )

        _run_cycle(self.m)

        assert self.m["evaluate"].call_count == 2
        # Second evaluate must see AAPL as a pending/virtual position,
        # even though get_open_positions returned empty.
        second_call_positions = self.m["evaluate"].call_args_list[1][0][1]
        tickers = {p.ticker for p in second_call_positions}
        assert "AAPL" in tickers, (
            "Second risk check must see the just-placed AAPL order as a "
            "pending position; DB refresh can lag the fill by seconds."
        )

    def test_summary_counts_correct(self):
        sig1 = _make_signal(ticker="A")
        sig2 = _make_signal(ticker="B")
        sig3 = _make_signal(ticker="C")
        _setup_mocks(
            self.m,
            [sig1, sig2, sig3],
            risk_approved=[True, False, True],
            positions_sequence=[[], [], [], []],
        )

        summary = _run_cycle(self.m)

        assert summary["ai_approved"] == 3
        assert summary["risk_approved"] == 2
        assert summary["orders_placed"] == 2

    def test_fill_handler_attached_after_order_with_parent(self):
        """setup_fill_handler must be called AFTER place_order with parent_order.

        The bracket uses transmit=False until the last child order, so fills
        cannot arrive before place_order returns. Attaching handlers after
        allows passing the parent order for precise permId matching.
        """
        sig = _make_signal(ticker="RACE")
        _setup_mocks(self.m, [sig], risk_approved=[True])

        call_order = []
        self.m["setup_fill_handler"].side_effect = lambda *a, **kw: call_order.append("handler")
        self.m["place_order"].side_effect = lambda *a, **kw: (call_order.append("order"), [MagicMock()])[1]

        _run_cycle(self.m)

        assert call_order == ["order", "handler"], (
            f"Fill handler must be attached after order placement, got: {call_order}"
        )
        # Verify parent_order is passed for precise matching
        _, kwargs = self.m["setup_fill_handler"].call_args
        assert "parent_order" in kwargs and kwargs["parent_order"] is not None

    def test_exit_handler_attached_for_approved_signals(self):
        """setup_exit_handler must be called for risk-approved signals."""
        sig = _make_signal(ticker="EXIT")
        _setup_mocks(self.m, [sig], risk_approved=[True])

        _run_cycle(self.m)

        self.m["setup_exit_handler"].assert_called_once()

    def test_evaluate_receives_current_price(self):
        """evaluate() must receive current_price (not default 0) for anti-momentum."""
        sig = _make_signal(ticker="AMOM", entry_price=100.0)
        _setup_mocks(self.m, [sig], risk_approved=[True])

        _run_cycle(self.m)

        self.m["evaluate"].assert_called_once()
        kwargs = self.m["evaluate"].call_args[1]
        assert "current_price" in kwargs
        assert kwargs["current_price"] > 0, "current_price must not be 0"

    def test_sector_injected_into_candidates(self):
        """Sector from universe must be injected into screener candidates."""
        sig = _make_signal(ticker="TECH")
        _setup_mocks(self.m, [sig], risk_approved=[True], sectors=["Technology"])

        _run_cycle(self.m)

        # The signal passed to evaluate should have sector in indicator_values
        evaluated_signal = self.m["evaluate"].call_args[0][0]
        assert evaluated_signal.indicator_values.get("sector") == "Technology"

    def test_exit_signal_uses_market_order_not_bracket(self):
        """Exit signals (RiskResult.is_exit=True) must use place_market_order,
        not place_order (bracket). A bracket's SL/TP children stay live at
        IBKR after the parent closes our position — they can re-enter the
        ticker when price crosses the child's trigger. For an exit, we want
        a clean one-shot close with no residual orders.
        """
        from core.models import Action
        sig = _make_signal(ticker="EXIT", action=Action.SELL)
        _setup_mocks(
            self.m, [sig], risk_approved=[True],
            is_exit_flags=[True],
        )

        _run_cycle(self.m)

        self.m["place_order"].assert_not_called()
        self.m["place_market_order"].assert_called_once()

    def test_entry_signal_still_uses_bracket(self):
        """Entries must keep the bracket order — SL/TP children are what
        protect an open position once the parent fills."""
        sig = _make_signal(ticker="ENTRY")
        _setup_mocks(
            self.m, [sig], risk_approved=[True],
            is_exit_flags=[False],
        )

        _run_cycle(self.m)

        self.m["place_order"].assert_called_once()
        self.m["place_market_order"].assert_not_called()


class TestNoNewsNotDropped:
    """Post-inversion, a candidate lacking news must NOT be dropped before risk.

    Dropping-on-no-news made sense when the AI was the ORIGINATOR (skip the
    candidate the LLM couldn't reason about). Now the screener is the
    originator and the LLM only vetoes — dropping a no-news candidate would
    silently shrink the mechanical universe. "No news" routes to the gate as
    INSUFFICIENT_DATA (buy stands, D-05 spirit), not a pre-risk drop.
    """

    @pytest.fixture(autouse=True)
    def _patch_all(self):
        patchers = {name.split(".")[-1]: patch(name) for name in _PATCHES}
        self.m = {}
        for key, p in patchers.items():
            self.m[key] = p.start()
        yield
        for p in patchers.values():
            p.stop()

    def test_no_news_candidate_still_reaches_risk_and_executes(self):
        sig = _make_signal(ticker="NONEWS")
        _setup_mocks(self.m, [sig], risk_approved=[True])
        self.m["get_news"].return_value = []  # no headlines anywhere

        _run_cycle(self.m)

        # Not dropped: risk evaluated it and (gate OK) it was placed.
        self.m["evaluate"].assert_called_once()
        assert self.m["evaluate"].call_args[0][0].ticker == "NONEWS"
        self.m["place_order"].assert_called_once()

    def test_no_news_candidate_is_still_gated(self):
        """The no-news candidate reaches the gate (which returns the empty
        source_text) rather than being skipped before it."""
        sig = _make_signal(ticker="NONEWS")
        _setup_mocks(self.m, [sig], risk_approved=[True])
        self.m["get_news"].return_value = []

        _run_cycle(self.m)

        self.m["gate_signal"].assert_called_once()

    def test_mixed_news_keeps_both_candidates(self):
        """Two candidates — one with news, one without — BOTH reach risk."""
        sig_news = _make_signal(ticker="HAS")
        sig_nonews = _make_signal(ticker="NONE")
        _setup_mocks(
            self.m, [sig_news, sig_nonews],
            risk_approved=[True, True],
            positions_sequence=[[], [], []],
        )

        def news_by_ticker(ticker, market=None):
            return ["good headline"] if ticker == "HAS" else []
        self.m["get_news"].side_effect = news_by_ticker

        _run_cycle(self.m)

        evaluated = {c.args[0].ticker for c in self.m["evaluate"].call_args_list}
        assert evaluated == {"HAS", "NONE"}


class TestVetoOnlyInvariant:
    """LLM-01 / ROADMAP crit 1 — the veto-only invariant.

    Over a full scan cycle, the set of buys that reach execution
    (``post_llm_buys``) must be a strict SUBSET of the risk-approved buys
    (``pre_llm_buys``): the gate can only remove a buy, never originate one.
    """

    @pytest.fixture(autouse=True)
    def _patch_all(self):
        patchers = {name.split(".")[-1]: patch(name) for name in _PATCHES}
        self.m = {}
        for key, p in patchers.items():
            self.m[key] = p.start()
        yield
        for p in patchers.values():
            p.stop()

    def test_invariant_veto_yields_strict_subset_of_risk_approved(self):
        sigs = [_make_signal(ticker=t) for t in ("AAA", "BBB", "CCC")]
        _setup_mocks(
            self.m, sigs,
            risk_approved=[True, True, True],
            positions_sequence=[[]] * 4,
        )

        # Gate vetoes exactly one ticker (BBB); clears the rest.
        def fake_gate_signal(src, candidate, horizon, earnings_date=None):
            if candidate["ticker"] == "BBB":
                return GateResult(Verdict.VETO, None, "catalyst", "gemini")
            return GateResult(Verdict.OK, None, "", "gemini")
        self.m["gate_signal"].side_effect = fake_gate_signal

        _run_cycle(self.m)

        risk_approved = {"AAA", "BBB", "CCC"}  # every candidate risk-approved
        executed = {c.args[1].ticker for c in self.m["place_order"].call_args_list}

        assert executed <= risk_approved, "post_llm_buys must be a subset of pre_llm_buys"
        assert "BBB" not in executed, "the VETO'd ticker must be removed"
        assert executed == {"AAA", "CCC"}, "exactly the non-vetoed buys execute"
        assert not (executed - risk_approved), "the gate must never add a ticker"

    def test_invariant_gate_cannot_originate_a_noncandidate_buy(self):
        """Even if the gate's output references a ghost ticker, it cannot inject
        it — the gate output carries no buy and the loop only ever appends
        signals drawn from pre_llm_buys."""
        sigs = [_make_signal(ticker="AAA"), _make_signal(ticker="BBB")]
        _setup_mocks(
            self.m, sigs,
            risk_approved=[True, True],
            positions_sequence=[[]] * 3,
        )

        def fake_gate_signal(src, candidate, horizon, earnings_date=None):
            # A "bullish" OK that name-drops a non-candidate ticker.
            return GateResult(Verdict.OK, "GHOST is a strong buy", "", "gemini")
        self.m["gate_signal"].side_effect = fake_gate_signal

        _run_cycle(self.m)

        executed = {c.args[1].ticker for c in self.m["place_order"].call_args_list}
        assert "GHOST" not in executed, "the gate cannot originate a buy"
        assert executed == {"AAA", "BBB"}


class TestLoopLiveness:
    """LLM-05 / ROADMAP crit 4 — the off-loop bridge keeps the event loop live.

    A loop callback (standing in for a fill/disconnect) must be serviced WHILE
    a blocking gate call is in flight, because the gate runs in a worker thread
    (run_in_executor) while ib.sleep(0.1) pumps the loop. The Phase-4 paper
    clock may start only after this is green.
    """

    def test_loop_liveness_callback_fires_during_inflight_gate(self):
        loop = _ensure_event_loop()
        flag = {"fired": False}

        def blocking_gate(*args, **kwargs):
            time.sleep(1.0)  # blocks the WORKER thread ~1s
            return GateResult(Verdict.OK, None, "", "gemini")

        ib = _make_ib()
        # Schedule a loop callback BEFORE entering the pump loop.
        loop.call_later(0.2, lambda: flag.__setitem__("fired", True))

        fut = loop.run_in_executor(None, blocking_gate, "src", {}, 10)
        fired_while_inflight = None
        while not fut.done():
            ib.sleep(0.1)  # the SAME bridge run_scan_cycle uses
            if flag["fired"] and fired_while_inflight is None:
                fired_while_inflight = not fut.done()
        fut.result()

        assert flag["fired"] is True
        assert fired_while_inflight is True, (
            "the loop callback must fire WHILE the ~1s gate is still in flight — "
            "proving fills/disconnects are serviced off-loop"
        )

    def test_loop_liveness_negative_control_mainthread_block_freezes_loop(self):
        """Negative control: blocking the MAIN thread (time.sleep) instead of
        the executor bridge freezes the loop — the callback does NOT fire until
        the loop is pumped. Proves the positive test actually discriminates."""
        loop = _ensure_event_loop()
        flag = {"fired": False}
        loop.call_later(0.2, lambda: flag.__setitem__("fired", True))

        time.sleep(1.0)  # blocks the MAIN thread — the loop cannot advance
        fired_during_block = flag["fired"]

        util.run(asyncio.sleep(0.3))  # now pump — the overdue callback fires

        assert fired_during_block is False, (
            "a main-thread time.sleep must FREEZE the loop (no callback) — this "
            "is exactly what the off-loop bridge avoids"
        )
        assert flag["fired"] is True, "after pumping, the overdue callback fires"


# ---------------------------------------------------------------------------
# Feature 3: Nightly reconciliation
# ---------------------------------------------------------------------------

from datetime import datetime, date, timezone  # noqa: E402

from core.scheduler import run_nightly_reconciliation  # noqa: E402


class _FakeIBPos:
    def __init__(self, symbol, quantity):
        self.contract = type("C", (), {"symbol": symbol})()
        self.position = quantity


class TestNightlyReconciliation:
    """run_nightly_reconciliation(ib) fetches positions + fills, calls
    reconcile_positions (read-only), and sends Telegram alert on mismatch.
    """

    def test_in_sync_does_not_alert(self, tmp_path):
        """When DB and IBKR agree, no alert should be sent."""
        from core.portfolio import init_db, add_position
        from core.models import TradeType, Position

        db_path = tmp_path / "test.db"
        init_db(db_path)
        add_position(Position(
            ticker="AAPL", exchange="SMART", quantity=10,
            entry_price=150.0, entry_time=datetime(2024, 1, 1),
            stop_loss=145.0, take_profit=160.0, trade_type=TradeType.DAY,
        ), db_path)

        ib = MagicMock()
        ib.positions.return_value = [_FakeIBPos("AAPL", 10)]
        ib.fills.return_value = []

        with patch("core.scheduler.notify_reconciliation_mismatch") as mock_notify:
            report = run_nightly_reconciliation(ib, db_path=db_path)

        assert report["in_sync"] is True
        mock_notify.assert_not_called()

    def test_orphaned_db_position_sends_alert(self, tmp_path):
        """DB has AAPL but IBKR doesn't → alert."""
        from core.portfolio import init_db, add_position
        from core.models import TradeType, Position

        db_path = tmp_path / "test.db"
        init_db(db_path)
        add_position(Position(
            ticker="AAPL", exchange="SMART", quantity=10,
            entry_price=150.0, entry_time=datetime(2024, 1, 1),
            stop_loss=145.0, take_profit=160.0, trade_type=TradeType.DAY,
        ), db_path)

        ib = MagicMock()
        ib.positions.return_value = []
        ib.fills.return_value = []

        with patch("core.scheduler.notify_reconciliation_mismatch") as mock_notify:
            report = run_nightly_reconciliation(ib, db_path=db_path)

        assert report["in_sync"] is False
        assert "AAPL" in report["orphaned_db"]
        mock_notify.assert_called_once()
        args = mock_notify.call_args.args[0]
        assert args["orphaned_db"] == ["AAPL"]

    def test_orphaned_ibkr_position_sends_alert(self, tmp_path):
        """IBKR has MSFT but DB doesn't → alert."""
        from core.portfolio import init_db

        db_path = tmp_path / "test.db"
        init_db(db_path)

        ib = MagicMock()
        ib.positions.return_value = [_FakeIBPos("MSFT", 20)]
        ib.fills.return_value = []

        with patch("core.scheduler.notify_reconciliation_mismatch") as mock_notify:
            report = run_nightly_reconciliation(ib, db_path=db_path)

        assert "MSFT" in report["orphaned_ibkr"]
        mock_notify.assert_called_once()

    def test_quantity_mismatch_sends_alert(self, tmp_path):
        """DB says 10 shares, IBKR says 100 → alert."""
        from core.portfolio import init_db, add_position
        from core.models import TradeType, Position

        db_path = tmp_path / "test.db"
        init_db(db_path)
        add_position(Position(
            ticker="AAPL", exchange="SMART", quantity=10,
            entry_price=150.0, entry_time=datetime(2024, 1, 1),
            stop_loss=145.0, take_profit=160.0, trade_type=TradeType.DAY,
        ), db_path)

        ib = MagicMock()
        ib.positions.return_value = [_FakeIBPos("AAPL", 100)]
        ib.fills.return_value = []

        with patch("core.scheduler.notify_reconciliation_mismatch") as mock_notify:
            report = run_nightly_reconciliation(ib, db_path=db_path)

        assert "AAPL" in report["qty_mismatches"]
        mock_notify.assert_called_once()

    def test_does_not_auto_close_positions(self, tmp_path):
        """Nightly reconciliation is read-only — never closes positions.
        Auto-fix is reserved for reconnect-time reconciliation only.
        """
        from core.portfolio import init_db, add_position, get_open_positions
        from core.models import TradeType, Position

        db_path = tmp_path / "test.db"
        init_db(db_path)
        add_position(Position(
            ticker="AAPL", exchange="SMART", quantity=10,
            entry_price=150.0, entry_time=datetime(2024, 1, 1),
            stop_loss=145.0, take_profit=160.0, trade_type=TradeType.DAY,
        ), db_path)

        ib = MagicMock()
        ib.positions.return_value = []  # IBKR disagrees
        ib.fills.return_value = []

        with patch("core.scheduler.notify_reconciliation_mismatch"):
            run_nightly_reconciliation(ib, db_path=db_path)

        # Position must still be in DB (not auto-closed)
        assert len(get_open_positions(db_path)) == 1

    def test_returns_full_report(self, tmp_path):
        """Return value should be the reconcile_positions report dict."""
        from core.portfolio import init_db

        db_path = tmp_path / "test.db"
        init_db(db_path)

        ib = MagicMock()
        ib.positions.return_value = []
        ib.fills.return_value = []

        with patch("core.scheduler.notify_reconciliation_mismatch"):
            report = run_nightly_reconciliation(ib, db_path=db_path)

        # Report should include all standard reconcile_positions keys
        for key in ("db_count", "ibkr_count", "orphaned_db",
                    "orphaned_ibkr", "qty_mismatches", "in_sync"):
            assert key in report

    def test_handles_ibkr_positions_error(self, tmp_path):
        """If ib.positions() raises, fail gracefully and notify error."""
        from core.portfolio import init_db

        db_path = tmp_path / "test.db"
        init_db(db_path)

        ib = MagicMock()
        ib.positions.side_effect = ConnectionError("disconnected")
        ib.fills.return_value = []

        # Should not propagate the exception; should return error report
        with patch("core.scheduler.notify_error") as mock_err, \
             patch("core.scheduler.notify_reconciliation_mismatch"):
            report = run_nightly_reconciliation(ib, db_path=db_path)

        assert report.get("error") is not None
        mock_err.assert_called_once()

    def test_sign_mismatch_marks_critical(self, tmp_path):
        """Direction mismatch (DB long, IBKR short) is a critical alert."""
        from core.portfolio import init_db, add_position
        from core.models import TradeType, Position

        db_path = tmp_path / "test.db"
        init_db(db_path)
        add_position(Position(
            ticker="AAPL", exchange="SMART", quantity=10,
            entry_price=150.0, entry_time=datetime(2024, 1, 1),
            stop_loss=145.0, take_profit=160.0, trade_type=TradeType.DAY,
        ), db_path)

        ib = MagicMock()
        ib.positions.return_value = [_FakeIBPos("AAPL", -10)]  # short!
        ib.fills.return_value = []

        with patch("core.scheduler.notify_reconciliation_mismatch") as mock_notify:
            report = run_nightly_reconciliation(ib, db_path=db_path)

        assert report["qty_mismatches"]["AAPL"]["type"] == "sign_mismatch"
        mock_notify.assert_called_once()

    def test_multiple_mismatches_single_alert(self, tmp_path):
        """All mismatches reported in one alert, not one per ticker."""
        from core.portfolio import init_db, add_position
        from core.models import TradeType, Position

        db_path = tmp_path / "test.db"
        init_db(db_path)
        for t in ("AAPL", "MSFT", "NVDA"):
            add_position(Position(
                ticker=t, exchange="SMART", quantity=10,
                entry_price=150.0, entry_time=datetime(2024, 1, 1),
                stop_loss=145.0, take_profit=160.0, trade_type=TradeType.DAY,
            ), db_path)

        ib = MagicMock()
        ib.positions.return_value = []  # IBKR has none of them
        ib.fills.return_value = []

        with patch("core.scheduler.notify_reconciliation_mismatch") as mock_notify:
            run_nightly_reconciliation(ib, db_path=db_path)

        mock_notify.assert_called_once()
        args = mock_notify.call_args.args[0]
        assert sorted(args["orphaned_db"]) == ["AAPL", "MSFT", "NVDA"]

    def test_empty_both_sides_is_in_sync(self, tmp_path):
        """DB empty and IBKR empty = in sync, no alert."""
        from core.portfolio import init_db

        db_path = tmp_path / "test.db"
        init_db(db_path)

        ib = MagicMock()
        ib.positions.return_value = []
        ib.fills.return_value = []

        with patch("core.scheduler.notify_reconciliation_mismatch") as mock_notify:
            report = run_nightly_reconciliation(ib, db_path=db_path)

        assert report["in_sync"] is True
        mock_notify.assert_not_called()


class TestNotifyReconciliationMismatch:
    """notify_reconciliation_mismatch(report) formats and sends alert."""

    def test_formats_orphaned_db(self):
        from notifications.telegram import notify_reconciliation_mismatch

        report = {
            "db_count": 2, "ibkr_count": 1,
            "orphaned_db": ["AAPL", "MSFT"],
            "orphaned_ibkr": [],
            "qty_mismatches": {},
            "in_sync": False,
        }
        with patch("notifications.telegram._send_sync") as mock_send:
            mock_send.return_value = True
            notify_reconciliation_mismatch(report)

        mock_send.assert_called_once()
        text = mock_send.call_args.args[0]
        assert "AAPL" in text
        assert "MSFT" in text

    def test_formats_orphaned_ibkr(self):
        from notifications.telegram import notify_reconciliation_mismatch

        report = {
            "db_count": 0, "ibkr_count": 1,
            "orphaned_db": [],
            "orphaned_ibkr": ["GOOG"],
            "qty_mismatches": {},
            "in_sync": False,
        }
        with patch("notifications.telegram._send_sync") as mock_send:
            mock_send.return_value = True
            notify_reconciliation_mismatch(report)

        text = mock_send.call_args.args[0]
        assert "GOOG" in text

    def test_formats_qty_mismatch(self):
        from notifications.telegram import notify_reconciliation_mismatch

        report = {
            "db_count": 1, "ibkr_count": 1,
            "orphaned_db": [], "orphaned_ibkr": [],
            "qty_mismatches": {
                "AAPL": {"db": 10, "ibkr": 100, "type": "quantity_mismatch"},
            },
            "in_sync": False,
        }
        with patch("notifications.telegram._send_sync") as mock_send:
            mock_send.return_value = True
            notify_reconciliation_mismatch(report)

        text = mock_send.call_args.args[0]
        assert "AAPL" in text
        assert "10" in text and "100" in text

    def test_sign_mismatch_flagged_critical(self):
        from notifications.telegram import notify_reconciliation_mismatch

        report = {
            "db_count": 1, "ibkr_count": 1,
            "orphaned_db": [], "orphaned_ibkr": [],
            "qty_mismatches": {
                "AAPL": {"db": 10, "ibkr": -10, "type": "sign_mismatch"},
            },
            "in_sync": False,
        }
        with patch("notifications.telegram._send_sync") as mock_send:
            mock_send.return_value = True
            notify_reconciliation_mismatch(report)

        text = mock_send.call_args.args[0]
        # Message should loudly warn about direction mismatch
        assert "direction" in text.lower() or "critical" in text.lower() or "sign" in text.lower()


class TestMarketHoursDST:
    """Verify NYSE market hours follow US DST, not the local display timezone.

    NYSE uses America/New_York which observes DST (EDT summer = UTC-4,
    EST winter = UTC-5). Europe/Istanbul (TRT) is a fixed UTC+3 offset.
    The gap between NYSE open (09:30 ET) in Istanbul wall-clock time is:
      - 16:30 TRT in summer (EDT)
      - 17:30 TRT in winter (EST)

    If market hours are stored as Istanbul clock times, the bot opens and
    closes scanning an hour off during DST shifts. The authoritative hours
    must be in ET so the NYSE open/close tracks the actual market.
    """

    def _run_at(self, iso_utc: str):
        """Return is_market_open('US') as if now() were the given UTC timestamp."""
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from unittest.mock import patch
        import core.scheduler as sched

        fixed = datetime.fromisoformat(iso_utc).replace(tzinfo=ZoneInfo("UTC"))

        class FakeDT(datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed.astimezone(tz) if tz else fixed.replace(tzinfo=None)

        with patch.object(sched, "datetime", FakeDT):
            return sched.is_market_open("US")

    def test_nyse_open_summer_edt(self):
        """2026-07-15 is EDT. 09:30 ET = 13:30 UTC → market open."""
        assert self._run_at("2026-07-15T13:35:00") is True

    def test_nyse_closed_before_open_summer(self):
        """2026-07-15 09:00 ET = 13:00 UTC → market not yet open."""
        assert self._run_at("2026-07-15T13:00:00") is False

    def test_nyse_open_winter_est(self):
        """2026-01-15 is EST. 09:30 ET = 14:30 UTC → market open.

        This is the critical DST case: in winter, NYSE opens 1 hour later
        in UTC than in summer. If hours are hard-coded as Istanbul clock
        times the winter NYSE open is missed entirely for the first hour.
        """
        assert self._run_at("2026-01-15T14:35:00") is True

    def test_nyse_closed_before_open_winter(self):
        """2026-01-15 09:00 ET = 14:00 UTC → market not yet open.

        In summer this same UTC time would be market-open. The test ensures
        the bot does not scan before the ET open during EST.
        """
        assert self._run_at("2026-01-15T14:00:00") is False

    def test_nyse_closed_after_close_winter(self):
        """2026-01-15 16:05 ET = 21:05 UTC → past close."""
        assert self._run_at("2026-01-15T21:05:00") is False

    def test_nyse_open_just_before_close_winter(self):
        """2026-01-15 15:55 ET = 20:55 UTC → still open."""
        assert self._run_at("2026-01-15T20:55:00") is True

    def test_weekend_closed(self):
        """Saturday is closed regardless of time."""
        # 2026-01-17 is a Saturday
        assert self._run_at("2026-01-17T15:00:00") is False


class TestPdtTradeWindow:
    """The PDT rule counts day trades over a rolling 5-business-day window,
    which is ~7 calendar days. The scheduler must query the DB with a
    start_date 7 days back so `check_pdt_restriction` sees the full window.

    Previously the scheduler queried `start_date=today`, which truncated the
    trade history to today only. With that truncation the 7-day filter inside
    check_pdt_restriction is a no-op and the rolling count is effectively 0,
    which defeats the protection entirely and can trigger an IBKR 30-day
    account lockout on sub-$5k accounts.
    """

    @pytest.fixture(autouse=True)
    def _patch_all(self):
        patchers = {name.split(".")[-1]: patch(name) for name in _PATCHES}
        self.m = {}
        for key, p in patchers.items():
            self.m[key] = p.start()
        yield
        for p in patchers.values():
            p.stop()

    def test_get_trades_queries_seven_days_back_for_pdt_window(self):
        """scheduler._on_signal must query get_trades with start_date ~7 days ago.

        Using `datetime.now().date()` as the start date truncates the PDT
        trade history to today only, silently disabling PDT protection.
        """
        from datetime import date, timedelta, datetime as _dt, timezone as _tz

        sig = _make_signal(ticker="PDTX")
        _setup_mocks(self.m, [sig], risk_approved=[True])

        captured = {}

        def fake_get_trades(start_date=None, end_date=None, ticker=None, db_path=None):
            # record only the first call that passes start_date (the PDT lookup).
            if start_date is not None and "start_date" not in captured:
                captured["start_date"] = start_date
            return []

        from core.scheduler import run_scan_cycle
        ib = _make_ib()
        with patch("core.scheduler._fetch_market_data") as mock_fetch, \
             patch("core.connection.get_account_summary", return_value={"NetLiquidation": 100_000}), \
             patch("core.scheduler.minutes_to_close", return_value=999), \
             patch("core.scheduler.get_trades", side_effect=fake_get_trades):
            mock_fetch.return_value = _make_stock_data(["PDTX"])
            run_scan_cycle(ib, ["US"])

        assert "start_date" in captured, "scheduler did not call get_trades with start_date"
        today_utc = _dt.now(_tz.utc).date()
        delta = (today_utc - captured["start_date"]).days
        assert delta >= 5, (
            f"PDT trade window too narrow: scheduler looked back {delta} days, "
            f"need at least 5 business days (~7 calendar days) to honour IBKR's "
            f"5-day rolling PDT rule."
        )


# ---------------------------------------------------------------------------
# TestCashGate
# ---------------------------------------------------------------------------

class TestCashGate:
    """Risk-approved BUY must still check available cash vs. pending BUY reserves.

    Motivating bug (2026-04-22): HPE bracket reserved ~$200.20, then STLD $215
    got risk-approved against a stale NetLiquidation and IBKR rejected with
    Error 201 (cash needed 417.20 USD). Available cash must be computed as
    TotalCashValue − sum(unfilled parent BUY reserves); if insufficient, the
    scheduler must skip the candidate without calling place_order.
    """

    @pytest.fixture(autouse=True)
    def _patch_all(self):
        patchers = {name.split(".")[-1]: patch(name) for name in _PATCHES}
        self.m = {}
        for key, p in patchers.items():
            self.m[key] = p.start()
        yield
        for p in patchers.values():
            p.stop()

    def test_skips_place_order_when_cash_short_no_eviction(self, caplog):
        from core.models import Action
        signals = [_make_signal(ticker="STLD", action=Action.BUY, entry_price=215.0)]
        _setup_mocks(self.m, signals)

        # Mirror the 2026-04-22 bug: $369 cash, $200 reserved in pending HPE,
        # STLD wants $215 — risk approves on NLV but cash is short.
        self.m["get_pending_buy_reserve"].return_value = 200.0
        self.m["evaluate"].side_effect = [
            RiskResult(approved=True, reasons=[], position_size=1, is_exit=False),
        ]

        import logging
        with caplog.at_level(logging.INFO, logger="core.scheduler"):
            _run_cycle(self.m, account={"NetLiquidation": 455.0, "TotalCashValue": 369.0})

        self.m["place_order"].assert_not_called()
        assert any("cash short" in rec.message.lower() for rec in caplog.records), (
            "Expected a 'cash short' log when available cash < needed cash; "
            f"got: {[r.message for r in caplog.records]}"
        )

    def test_proceeds_when_cash_sufficient(self):
        from core.models import Action
        signals = [_make_signal(ticker="AAPL", action=Action.BUY, entry_price=150.0)]
        _setup_mocks(self.m, signals)
        self.m["get_pending_buy_reserve"].return_value = 0.0
        # position_size=10 → $1500 needed, $5000 available → OK
        self.m["evaluate"].side_effect = [
            RiskResult(approved=True, reasons=[], position_size=10, is_exit=False),
        ]

        _run_cycle(self.m, account={"NetLiquidation": 10_000, "TotalCashValue": 5_000})

        self.m["place_order"].assert_called_once()

    def test_ignores_cash_gate_for_exit_signals(self):
        """Exit signals close existing positions — they don't consume new cash."""
        from core.models import Action
        signals = [_make_signal(ticker="AAPL", action=Action.SELL, entry_price=150.0)]
        _setup_mocks(self.m, signals)
        # No available cash, but this is an exit, so gate must not fire
        self.m["get_pending_buy_reserve"].return_value = 0.0
        self.m["evaluate"].side_effect = [
            RiskResult(approved=True, reasons=[], position_size=10, is_exit=True),
        ]

        _run_cycle(self.m, account={"NetLiquidation": 100, "TotalCashValue": 0})

        # Exit goes through place_market_order, not place_order
        self.m["place_market_order"].assert_called_once()
        self.m["place_order"].assert_not_called()

    def test_ignores_cash_gate_for_sell_entries(self):
        """Opening a short (SELL entry) doesn't consume cash — don't gate on it."""
        from core.models import Action
        signals = [_make_signal(ticker="AAPL", action=Action.SELL, entry_price=150.0)]
        _setup_mocks(self.m, signals)
        self.m["get_pending_buy_reserve"].return_value = 0.0
        self.m["evaluate"].side_effect = [
            RiskResult(approved=True, reasons=[], position_size=10, is_exit=False),
        ]

        _run_cycle(self.m, account={"NetLiquidation": 100, "TotalCashValue": 0})

        self.m["place_order"].assert_called_once()


class TestEvictAndPlace:
    """When cash is short, evict a weak pending BUY to make room for a stronger one."""

    @pytest.fixture(autouse=True)
    def _patch_all(self):
        patchers = {name.split(".")[-1]: patch(name) for name in _PATCHES}
        # Eviction helper is looked up via core.scheduler — add its patch too.
        evict_patch = patch("core.scheduler.evict_weakest_pending")
        patchers["evict_weakest_pending"] = evict_patch
        self.m = {}
        for key, p in patchers.items():
            self.m[key] = p.start()
        yield
        for p in patchers.values():
            p.stop()

    def test_evicts_then_places_when_new_is_stronger(self):
        """New conf 78 vs pending conf 60 (+margin 5) → evict, retry, place."""
        from core.models import Action
        signals = [_make_signal(
            ticker="STLD", action=Action.BUY, entry_price=215.0, confidence=78.0,
        )]
        _setup_mocks(self.m, signals)

        # Starting state: $369 cash, $200 reserved (mirrors the 2026-04-22 bug).
        self.m["get_pending_buy_reserve"].side_effect = [200.0, 0.0]
        # Eviction succeeds
        self.m["evict_weakest_pending"].return_value = True
        self.m["evaluate"].side_effect = [
            RiskResult(approved=True, reasons=[], position_size=1, is_exit=False),
        ]

        _run_cycle(self.m, account={"NetLiquidation": 455.0, "TotalCashValue": 369.0})

        # Eviction called with the new signal's confidence and needed cash
        self.m["evict_weakest_pending"].assert_called_once()
        args, kwargs = (
            self.m["evict_weakest_pending"].call_args.args,
            self.m["evict_weakest_pending"].call_args.kwargs,
        )
        # Support either positional or keyword call shapes
        new_conf = kwargs.get("new_confidence", args[1] if len(args) > 1 else None)
        needed = kwargs.get("needed_cash", args[2] if len(args) > 2 else None)
        assert new_conf == pytest.approx(78.0)
        assert needed == pytest.approx(215.0)
        # After eviction freed cash, bracket is placed
        self.m["place_order"].assert_called_once()

    def test_skips_when_eviction_declined(self):
        """Eviction returns False → skip; no place_order, no crash."""
        from core.models import Action
        signals = [_make_signal(
            ticker="STLD", action=Action.BUY, entry_price=215.0, confidence=72.0,
        )]
        _setup_mocks(self.m, signals)

        self.m["get_pending_buy_reserve"].return_value = 200.0
        self.m["evict_weakest_pending"].return_value = False
        self.m["evaluate"].side_effect = [
            RiskResult(approved=True, reasons=[], position_size=1, is_exit=False),
        ]

        _run_cycle(self.m, account={"NetLiquidation": 455.0, "TotalCashValue": 369.0})

        self.m["evict_weakest_pending"].assert_called_once()
        self.m["place_order"].assert_not_called()


class TestTwoSourceConsensusWiring:
    """Scheduler must fetch BOTH yfinance and IBKR analyst consensus and pass
    both into risk evaluate(). If either source is missing or doesn't agree on
    buy/strong_buy, evaluate() will block the BUY (asserted in test_risk.py).
    These tests verify the scheduler-side wiring only.
    """

    @pytest.fixture(autouse=True)
    def _patch_all(self):
        patchers = {name.split(".")[-1]: patch(name) for name in _PATCHES}
        # Inline imports inside run_scan_cycle target core.data.* directly
        # — patch them at the module of definition, not core.scheduler.
        patchers["yf_recs"] = patch("core.data.get_analyst_recommendation")
        patchers["ibkr_recs"] = patch("core.data.get_analyst_recommendation_ibkr")
        self.m = {}
        for key, p in patchers.items():
            self.m[key] = p.start()
        yield
        for p in patchers.values():
            p.stop()

    def test_evaluate_receives_both_consensus_kwargs(self):
        """run_scan_cycle must pass analyst_consensus= and
        analyst_consensus_ibkr= into evaluate() for every BUY signal."""
        from core.models import Action
        sig = _make_signal(ticker="AAPL", action=Action.BUY)
        _setup_mocks(self.m, [sig])
        self.m["yf_recs"].return_value = {"consensus": "buy", "details": {}}
        self.m["ibkr_recs"].return_value = {"consensus": "strong_buy"}

        _run_cycle(self.m)

        self.m["evaluate"].assert_called_once()
        call_kwargs = self.m["evaluate"].call_args.kwargs
        assert call_kwargs.get("analyst_consensus") == "buy", (
            f"Scheduler must pass yfinance consensus as analyst_consensus= "
            f"to evaluate(). Got kwargs={list(call_kwargs)}"
        )
        assert call_kwargs.get("analyst_consensus_ibkr") == "strong_buy", (
            f"Scheduler must pass IBKR consensus as analyst_consensus_ibkr= "
            f"to evaluate(). Got kwargs={list(call_kwargs)}"
        )

    def test_ibkr_consensus_fetcher_is_called(self):
        """The IBKR analyst-rec fetch must be invoked for each BUY candidate."""
        from core.models import Action
        sig = _make_signal(ticker="MSFT", action=Action.BUY)
        _setup_mocks(self.m, [sig])
        self.m["yf_recs"].return_value = {"consensus": "buy", "details": {}}
        self.m["ibkr_recs"].return_value = {"consensus": "buy"}

        _run_cycle(self.m)

        assert self.m["ibkr_recs"].called, (
            "Scheduler must call get_analyst_recommendation_ibkr — "
            "the second source is required for the two-source agreement gate."
        )

    def test_passes_none_when_ibkr_returns_none(self):
        """IBKR fetch returns None (no subscription/network) → evaluate must
        receive analyst_consensus_ibkr=None and risk layer will block."""
        from core.models import Action
        sig = _make_signal(ticker="GOOG", action=Action.BUY)
        _setup_mocks(self.m, [sig])
        self.m["yf_recs"].return_value = {"consensus": "buy", "details": {}}
        self.m["ibkr_recs"].return_value = None

        _run_cycle(self.m)

        call_kwargs = self.m["evaluate"].call_args.kwargs
        assert call_kwargs.get("analyst_consensus_ibkr") is None

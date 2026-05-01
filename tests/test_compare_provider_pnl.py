"""Tests for scripts/compare_provider_pnl.py — provider→P&L attribution."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load the script as a module without requiring scripts/__init__.py.
_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "compare_provider_pnl.py"
_spec = importlib.util.spec_from_file_location("compare_provider_pnl", _SCRIPT_PATH)
cpp = importlib.util.module_from_spec(_spec)
sys.modules["compare_provider_pnl"] = cpp
_spec.loader.exec_module(cpp)


# ---------------------------------------------------------------------------
# Text-log parsing
# ---------------------------------------------------------------------------

class TestParseTextLog:
    def test_attributes_provider_to_approval(self):
        log = (
            "2026-04-21 22:12:03,235 [INFO] core.analyst: Analyzing candidate 1/2: UNH\n"
            "2026-04-21 22:12:18,235 [INFO] core.analyst: Ollama response in 9.5s (qwen3:8b)\n"
            "2026-04-21 22:12:18,236 [INFO] core.analyst: AI approved UNH: buy confidence=75\n"
        )
        out = cpp.parse_text_log(log, source_date="2026-04-21")
        assert out == [cpp.Approval(date="2026-04-21", ticker="UNH",
                                    action="buy", confidence=75, provider="ollama")]

    def test_resets_provider_on_new_candidate(self):
        # If candidate B starts before its own provider line lands, an earlier
        # candidate's provider must NOT leak into B's approval.
        log = (
            "Analyzing candidate 1/2: AAA\n"
            "Gemini response OK (m)\n"
            "AI approved AAA: buy confidence=80\n"
            "Analyzing candidate 2/2: BBB\n"
            "AI approved BBB: buy confidence=70\n"  # no provider line in window
        )
        out = cpp.parse_text_log(log, source_date="2026-04-21")
        # First approval gets gemini; second has no provider — drop or tag unknown.
        # Spec: drop unattributable approvals (we can't compare what we can't label).
        assert len(out) == 1
        assert out[0].ticker == "AAA"
        assert out[0].provider == "gemini"

    def test_multiple_candidates_each_get_their_own_provider(self):
        log = (
            "Analyzing candidate 1/2: AAA\n"
            "Gemini response OK\n"
            "AI approved AAA: buy confidence=80\n"
            "Analyzing candidate 2/2: BBB\n"
            "Ollama response in 5s\n"
            "AI approved BBB: sell confidence=70\n"
        )
        out = cpp.parse_text_log(log, source_date="2026-04-21")
        providers = {a.ticker: a.provider for a in out}
        assert providers == {"AAA": "gemini", "BBB": "ollama"}

    def test_skips_dont_count_as_approvals(self):
        # Only "AI approved" → trade. "Skipping" → no trade was opened.
        log = (
            "Analyzing candidate 1/1: AAA\n"
            "Gemini response OK\n"
            "Skipping AAA: confidence 30 < threshold 65\n"
        )
        assert cpp.parse_text_log(log, source_date="2026-04-21") == []


# ---------------------------------------------------------------------------
# JSONL traffic-log parsing
# ---------------------------------------------------------------------------

class TestParseJsonlLog:
    def test_returns_only_trading_approvals(self):
        import json
        records = [
            {"kind": "trading", "ticker": "AAPL", "provider": "gemini",
             "ts": "2026-04-29T18:25:56Z",
             "response": {"action": "buy", "confidence": 75}},
            {"kind": "trading", "ticker": "MSFT", "provider": "ollama",
             "ts": "2026-04-29T18:30:00Z",
             "response": {"action": "hold", "confidence": 70}},  # hold → not a trade
            {"kind": "trading", "ticker": "GOOG", "provider": "ollama",
             "ts": "2026-04-29T18:35:00Z",
             "response": {"action": "buy", "confidence": 50}},   # below threshold
            {"kind": "sector", "ticker": None, "provider": "gemini",
             "ts": "2026-04-29T18:40:00Z",
             "response": {"sector": "Technology"}},              # not a trading call
        ]
        content = "\n".join(json.dumps(r) for r in records)
        out = cpp.parse_jsonl_log(content)
        assert len(out) == 1
        assert out[0].ticker == "AAPL"
        assert out[0].provider == "gemini"
        assert out[0].confidence == 75


# ---------------------------------------------------------------------------
# Trade CSV loading + filtering
# ---------------------------------------------------------------------------

class TestLoadTrades:
    HEADER = ("timestamp,ticker,exchange,action,quantity,entry_price,exit_price,"
              "pnl,pnl_pct,trade_type,sector,reasoning,duration_hours")

    def test_drops_pre_2026_fixtures(self):
        # The 2024-01-15 AAPL row appears verbatim in many CSVs — pure test fixture.
        csv = (
            self.HEADER + "\n"
            "2024-01-15T15:00:00,AAPL,SMART,BUY,10,150.00,160.00,100.00,6.67,day,Tech,,4.5\n"
            "2026-04-21T22:30:00+00:00,UNH,SMART,BUY,10,300.00,310.00,100.00,3.33,day,Health,,2.0\n"
        )
        out = cpp.load_trades(csv)
        assert len(out) == 1
        assert out[0].ticker == "UNH"

    def test_drops_estimated_reconcile_rows(self):
        csv = (
            self.HEADER + "\n"
            "2026-04-21T22:30:00+00:00,UNH,SMART,BUY,10,300.00,310.00,100.00,3.33,day,Health,clean trade,2.0\n"
            "2026-04-21T22:35:00+00:00,XYZ,SMART,BUY,10,150.00,152.00,20.00,1.33,day,Tech,"
            "Auto-reconcile: midpoint estimate $152 of SL/TP,5.0\n"
        )
        out = cpp.load_trades(csv)
        assert {t.ticker for t in out} == {"UNH"}

    def test_keeps_actual_fill_reconciles(self):
        # "actual IBKR fill price" means the exit was real, not a guess. Keep.
        csv = (
            self.HEADER + "\n"
            "2026-04-17T11:00:00+00:00,MULT,SMART,BUY,10,100.00,97.00,-30.00,-3.00,day,Tech,"
            "Auto-reconcile: closed at actual IBKR fill price $97.0000,2.0\n"
        )
        out = cpp.load_trades(csv)
        assert len(out) == 1
        assert out[0].pnl == -30.0

    def test_parses_pnl_and_action_lowercased(self):
        csv = (
            self.HEADER + "\n"
            "2026-04-21T22:30:00+00:00,UNH,SMART,BUY,10,300.00,310.00,100.00,3.33,day,Health,clean,2.0\n"
        )
        out = cpp.load_trades(csv)
        assert out[0].action == "buy"
        assert out[0].pnl == 100.0
        assert out[0].pnl_pct == pytest.approx(3.33)


# ---------------------------------------------------------------------------
# Matching + aggregation
# ---------------------------------------------------------------------------

class TestMatching:
    def test_match_by_date_ticker_action(self):
        approvals = [
            cpp.Approval("2026-04-21", "UNH", "buy", 75, "ollama"),
            cpp.Approval("2026-04-21", "AAPL", "buy", 80, "gemini"),
        ]
        trades = [
            cpp.Trade("2026-04-21", "UNH", "buy", pnl=10.0, pnl_pct=1.0, reasoning=""),
            cpp.Trade("2026-04-21", "AAPL", "buy", pnl=-5.0, pnl_pct=-0.5, reasoning=""),
            cpp.Trade("2026-04-21", "GOOG", "buy", pnl=20.0, pnl_pct=2.0, reasoning=""),
        ]
        matched = cpp.match_trades_to_approvals(trades, approvals)
        provider_for = {m.trade.ticker: m.provider for m in matched}
        assert provider_for == {"UNH": "ollama", "AAPL": "gemini", "GOOG": None}


class TestAggregate:
    def test_per_provider_stats(self):
        matched = [
            cpp.MatchedTrade(cpp.Trade("d", "A", "buy", 10.0, 1.0, ""), provider="gemini"),
            cpp.MatchedTrade(cpp.Trade("d", "B", "buy", -5.0, -0.5, ""), provider="gemini"),
            cpp.MatchedTrade(cpp.Trade("d", "C", "buy", 20.0, 2.0, ""), provider="ollama"),
            cpp.MatchedTrade(cpp.Trade("d", "D", "buy", -10.0, -1.0, ""), provider="ollama"),
            cpp.MatchedTrade(cpp.Trade("d", "E", "buy", 50.0, 5.0, ""), provider="ollama"),
            cpp.MatchedTrade(cpp.Trade("d", "F", "buy", 0.0, 0.0, ""), provider=None),
        ]
        stats = cpp.aggregate(matched)
        assert stats["gemini"].n == 2
        assert stats["gemini"].total_pnl == 5.0
        assert stats["gemini"].mean_pnl == 2.5
        assert stats["gemini"].win_rate == pytest.approx(0.5)
        assert stats["ollama"].n == 3
        assert stats["ollama"].total_pnl == 60.0
        assert stats["ollama"].win_rate == pytest.approx(2 / 3)

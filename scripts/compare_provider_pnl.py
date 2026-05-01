#!/usr/bin/env python3
"""Attribute each closed trade in trades_*.csv to the LLM provider whose
'AI approved' line preceded it, then aggregate P&L per provider.

Why: confidence-distribution analysis (scripts/analyze_confidence.py) shows
Gemini and Ollama produce very different score shapes against the same
65-threshold gate (Gemini bimodal, Ollama clustered at 60-64). This script
turns that calibration question into "did the trades each provider approved
actually make money" — the answer that matters.

Provider attribution
--------------------
Two sources, merged:

  * llm_traffic_*.jsonl — exact provider+ticker+confidence per call (Apr 29+
    only; the diagnostic log was added with the multi-key rotation work).
  * trader_*.log        — provider attribution by log line ordering. Within a
    candidate window (between two "Analyzing candidate" lines) the most-recent
    "Gemini response OK"/"Ollama response in" line attributes the next
    "AI approved TICKER: action confidence=N" to that provider.

Trade filtering
---------------
trades_*.csv is contaminated with two classes of junk:

  * Pre-bot test fixtures — rows dated 2024-01-15 with synthetic AAPL prices
    that recur verbatim in many CSVs. Drop by timestamp < 2026-04-01.
  * Auto-reconcile estimates — rows where the bot reconstructed an exit price
    from the SL/TP midpoint after a position vanished while the bot was
    offline. P&L is fictional. Drop on "estimate" in the reasoning column.
    Auto-reconcile rows with "actual IBKR fill price" are kept (real exit).

Match key is (date, ticker, action). Trades without a matching approval go
into an `unknown` bucket so the unmatched count is visible.

Usage
-----
  scripts/compare_provider_pnl.py                       # all logs+csvs in logs/
  scripts/compare_provider_pnl.py --since 2026-04-15
  scripts/compare_provider_pnl.py --since 2026-04-15 --until 2026-04-30
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

# 65 matches AI_CONFIDENCE_THRESHOLD in config/settings.py — keep in sync if
# the threshold changes. Not imported because this script must run standalone
# (no project-root dependency).
APPROVAL_THRESHOLD = 65

# Earliest plausible real-trade date — anything before is a test fixture.
EARLIEST_REAL_DATE = date(2026, 4, 1)


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Approval:
    date: str        # YYYY-MM-DD
    ticker: str
    action: str      # "buy" or "sell"
    confidence: int
    provider: str    # "gemini" or "ollama"


@dataclass(frozen=True)
class Trade:
    date: str
    ticker: str
    action: str      # "buy" or "sell" (lowercased)
    pnl: float
    pnl_pct: float
    reasoning: str


@dataclass(frozen=True)
class MatchedTrade:
    trade: Trade
    provider: Optional[str]   # None if no matching approval


@dataclass
class Stats:
    n: int
    total_pnl: float
    mean_pnl: float
    mean_pnl_pct: float
    win_rate: float


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

_ANALYZING = re.compile(r"Analyzing candidate \d+/\d+: ([A-Z\.]+)")
_APPROVE = re.compile(r"AI approved (\S+):\s*(\w+)\s+confidence=(\d+)")


def parse_text_log(content: str, source_date: str) -> list[Approval]:
    """Walk a trader log line by line; emit one Approval per 'AI approved' line.

    `source_date` is the YYYY-MM-DD prefix of the source file's name — the bot
    log lines are timestamped per-line but this script aggregates daily, so a
    file-level date is sufficient. Approvals whose candidate window has no
    provider line are dropped (we cannot label them).
    """
    out: list[Approval] = []
    last_provider: Optional[str] = None
    for line in content.splitlines():
        if "Gemini response OK" in line:
            last_provider = "gemini"
            continue
        if "Ollama response in" in line:
            last_provider = "ollama"
            continue
        if _ANALYZING.search(line):
            last_provider = None
            continue
        m = _APPROVE.search(line)
        if m and last_provider:
            ticker, action, conf = m.group(1), m.group(2).lower(), int(m.group(3))
            if action in ("buy", "sell"):
                out.append(Approval(
                    date=source_date, ticker=ticker, action=action,
                    confidence=conf, provider=last_provider,
                ))
    return out


def parse_jsonl_log(content: str) -> list[Approval]:
    """Parse llm_traffic_*.jsonl content; emit one Approval per trading call
    that would have crossed the AI_CONFIDENCE_THRESHOLD with a tradeable action.

    JSONL is the more reliable source: each record carries provider+ticker
    explicitly, eliminating the candidate-window race that text-log parsing
    has to reason about.
    """
    out: list[Approval] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("kind") != "trading":
            continue
        ticker = rec.get("ticker")
        provider = rec.get("provider")
        resp = rec.get("response") or {}
        action = (resp.get("action") or "").lower()
        conf = resp.get("confidence")
        if not ticker or not provider:
            continue
        if action not in ("buy", "sell"):
            continue
        if not isinstance(conf, (int, float)) or conf < APPROVAL_THRESHOLD:
            continue
        ts = rec.get("ts", "")
        # ts is "YYYY-MM-DDTHH:MM:SSZ" — slice the date prefix.
        out.append(Approval(
            date=ts[:10], ticker=ticker, action=action,
            confidence=int(conf), provider=provider,
        ))
    return out


def load_trades(content: str) -> list[Trade]:
    """Parse a trades_*.csv body; drop fixtures and estimated reconciles.

    Filters:
      * timestamp < 2026-04-01 → pre-bot test fixture (drop)
      * 'estimate' in reasoning → auto-reconcile guess, P&L is fictional (drop)

    Auto-reconcile rows that say 'actual IBKR fill price' carry a real exit
    and are kept.
    """
    out: list[Trade] = []
    reader = csv.DictReader(io.StringIO(content))
    for row in reader:
        ts = row.get("timestamp", "")
        try:
            d = datetime.fromisoformat(ts).date()
        except ValueError:
            continue
        if d < EARLIEST_REAL_DATE:
            continue
        reasoning = row.get("reasoning", "") or ""
        if "estimate" in reasoning.lower():
            continue
        try:
            pnl = float(row["pnl"])
            pnl_pct = float(row["pnl_pct"])
        except (KeyError, ValueError):
            continue
        action = (row.get("action", "") or "").lower()
        if action not in ("buy", "sell"):
            continue
        out.append(Trade(
            date=d.isoformat(),
            ticker=row.get("ticker", ""),
            action=action,
            pnl=pnl,
            pnl_pct=pnl_pct,
            reasoning=reasoning,
        ))
    return out


# ---------------------------------------------------------------------------
# Matching + aggregation
# ---------------------------------------------------------------------------

def match_trades_to_approvals(
    trades: list[Trade], approvals: list[Approval],
) -> list[MatchedTrade]:
    """Tag each trade with the provider whose approval it most likely came from.

    Match key is (date, ticker, action). When several approvals match (same
    ticker approved twice on the same day), the first wins — the second is a
    re-approval of an already-open position, not a new trade. When no approval
    matches, provider is None ('unknown' bucket downstream).
    """
    by_key: dict[tuple[str, str, str], str] = {}
    for a in approvals:
        key = (a.date, a.ticker, a.action)
        by_key.setdefault(key, a.provider)
    out: list[MatchedTrade] = []
    for t in trades:
        provider = by_key.get((t.date, t.ticker, t.action))
        out.append(MatchedTrade(trade=t, provider=provider))
    return out


def aggregate(matched: list[MatchedTrade]) -> dict[str, Stats]:
    """Per-provider aggregates. Returns dict keyed by 'gemini'/'ollama'/'unknown'."""
    buckets: dict[str, list[Trade]] = {"gemini": [], "ollama": [], "unknown": []}
    for m in matched:
        key = m.provider or "unknown"
        buckets.setdefault(key, []).append(m.trade)
    out: dict[str, Stats] = {}
    for k, trades in buckets.items():
        n = len(trades)
        if n == 0:
            out[k] = Stats(0, 0.0, 0.0, 0.0, 0.0)
            continue
        total = sum(t.pnl for t in trades)
        mean = total / n
        mean_pct = sum(t.pnl_pct for t in trades) / n
        wins = sum(1 for t in trades if t.pnl > 0)
        out[k] = Stats(n=n, total_pnl=total, mean_pnl=mean,
                       mean_pnl_pct=mean_pct, win_rate=wins / n)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _date_from_filename(path: Path) -> Optional[date]:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    return date.fromisoformat(m.group(1)) if m else None


def _date_window(args: argparse.Namespace, d: date) -> bool:
    if args.since and d < date.fromisoformat(args.since):
        return False
    if args.until and d > date.fromisoformat(args.until):
        return False
    return True


def _collect(args: argparse.Namespace) -> tuple[list[Approval], list[Trade]]:
    approvals: list[Approval] = []
    trades: list[Trade] = []

    for path in sorted(LOGS_DIR.glob("trader_*.log")):
        d = _date_from_filename(path)
        if not d or not _date_window(args, d):
            continue
        approvals.extend(parse_text_log(path.read_text(), source_date=d.isoformat()))

    for path in sorted(LOGS_DIR.glob("llm_traffic_*.jsonl")):
        d = _date_from_filename(path)
        if not d or not _date_window(args, d):
            continue
        approvals.extend(parse_jsonl_log(path.read_text()))

    for path in sorted(LOGS_DIR.glob("trades_*.csv")):
        d = _date_from_filename(path)
        if not d or not _date_window(args, d):
            continue
        trades.extend(load_trades(path.read_text()))

    return approvals, trades


def _print_report(stats: dict[str, Stats], approvals_n: int, trades_n: int) -> None:
    print(f"Approvals labelled: {approvals_n}")
    print(f"Trades after filtering: {trades_n}\n")
    header = f"{'provider':<10} {'n':>5} {'total_pnl':>12} {'mean_pnl':>10} {'mean_pct':>10} {'win_rate':>10}"
    print(header)
    print("-" * len(header))
    for provider in ("gemini", "ollama", "unknown"):
        s = stats.get(provider) or Stats(0, 0.0, 0.0, 0.0, 0.0)
        print(f"{provider:<10} {s.n:>5} {s.total_pnl:>12.2f} {s.mean_pnl:>10.2f} "
              f"{s.mean_pnl_pct:>9.2f}% {s.win_rate*100:>9.1f}%")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--since", help="Earliest date (YYYY-MM-DD inclusive)")
    p.add_argument("--until", help="Latest date (YYYY-MM-DD inclusive)")
    args = p.parse_args()

    approvals, trades = _collect(args)
    matched = match_trades_to_approvals(trades, approvals)
    stats = aggregate(matched)
    _print_report(stats, approvals_n=len(approvals), trades_n=len(trades))
    return 0


if __name__ == "__main__":
    sys.exit(main())

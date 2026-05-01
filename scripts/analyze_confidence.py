#!/usr/bin/env python3
"""Compare Gemini-vs-Ollama confidence distributions over a date range.

Reads two log sources from logs/:
  * trader_YYYY-MM-DD.log         — line-based; pairs each "AI approved" /
    "Skipping <ticker>: confidence X" log line with the immediately
    preceding "Gemini response OK" / "Ollama response in" line to attribute
    the confidence to a provider.
  * llm_traffic_YYYY-MM-DD.jsonl  — one JSON record per round-trip; carries
    the parsed response, so confidence values are read directly from
    `response.confidence` and the trading-vs-sector split is exact.

Outputs per-provider counts, mean confidence, % at-or-above the 65 approval
threshold, % decisive rejects (<40), and a histogram. Useful for verifying
the rotation fix is producing approvals and to spot Ollama's mid-band
clustering pattern that pegs everything to 60-64.

Usage
-----
  scripts/analyze_confidence.py                       # all logs in logs/
  scripts/analyze_confidence.py --since 2026-04-29    # date filter
  scripts/analyze_confidence.py --since 2026-04-29 --until 2026-05-06
  scripts/analyze_confidence.py logs/trader_2026-04-29.log logs/llm_traffic_2026-04-29.jsonl
  scripts/analyze_confidence.py --reasoning           # show 5 sample reasoning strings per provider
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

_PROVIDER_LINE = re.compile(
    r"(Gemini response OK|Ollama response in)"
)
_ANALYZING = re.compile(r"Analyzing candidate \d+/\d+: ([A-Z\.]+)")
_SKIP = re.compile(r"Skipping (\S+):.*?confidence (\d+)")
_APPROVE = re.compile(r"AI approved (\S+):.*?confidence=(\d+)")


def _iter_log_files(args: argparse.Namespace) -> list[Path]:
    """Resolve which log files to scan from CLI args."""
    if args.files:
        return [Path(p) for p in args.files]
    files = sorted(LOGS_DIR.glob("trader_*.log")) + sorted(LOGS_DIR.glob("llm_traffic_*.jsonl"))
    out = []
    for f in files:
        # Extract YYYY-MM-DD from the filename to apply --since / --until.
        m = re.search(r"(\d{4}-\d{2}-\d{2})", f.name)
        if not m:
            continue
        d = date.fromisoformat(m.group(1))
        if args.since and d < date.fromisoformat(args.since):
            continue
        if args.until and d > date.fromisoformat(args.until):
            continue
        out.append(f)
    return out


def parse_text_log(path: Path) -> dict[str, list[tuple[int, str | None]]]:
    """Parse a trader_YYYY-MM-DD.log; return {provider: [(confidence, reasoning_or_none), ...]}.

    Pairs each Skipping/Approved log line with the most-recent provider line.
    """
    out: dict[str, list[tuple[int, str | None]]] = defaultdict(list)
    last_provider: str | None = None
    with open(path) as f:
        for line in f:
            if "Gemini response OK" in line:
                last_provider = "gemini"
                continue
            if "Ollama response in" in line:
                last_provider = "ollama"
                continue
            if _ANALYZING.search(line):
                last_provider = None  # new candidate, reset
                continue
            for rx in (_SKIP, _APPROVE):
                m = rx.search(line)
                if m and last_provider:
                    out[last_provider].append((int(m.group(2)), None))
                    break
    return out


def parse_jsonl_log(path: Path) -> dict[str, list[tuple[int, str | None]]]:
    """Parse a llm_traffic_YYYY-MM-DD.jsonl; return {provider: [(confidence, reasoning), ...]}.

    Only records where kind=='trading' and response.confidence is present.
    """
    out: dict[str, list[tuple[int, str | None]]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") != "trading":
                continue
            resp = rec.get("response") or {}
            conf = resp.get("confidence")
            if not isinstance(conf, (int, float)):
                continue
            provider = rec.get("provider", "unknown")
            reasoning = resp.get("reasoning") or None
            out[provider].append((int(conf), reasoning))
    return out


def merge(a: dict, b: dict) -> dict:
    out: dict[str, list[tuple[int, str | None]]] = defaultdict(list)
    for src in (a, b):
        for k, v in src.items():
            out[k].extend(v)
    return out


def histogram(vals: list[int]) -> str:
    """Bucket-count string for a confidence list."""
    buckets = [(0, 39), (40, 49), (50, 59), (60, 64), (65, 69), (70, 79), (80, 100)]
    parts = []
    for lo, hi in buckets:
        c = sum(1 for v in vals if lo <= v <= hi)
        parts.append(f"[{lo}-{hi}]:{c}")
    return " ".join(parts)


def report(samples: dict[str, list[tuple[int, str | None]]], show_reasoning: bool = False) -> None:
    if not any(samples.values()):
        print("No confidence samples found in the selected logs.")
        return
    for provider in ("gemini", "ollama"):
        rows = samples.get(provider) or []
        if not rows:
            print(f"{provider}: 0 samples")
            continue
        vals = [v for v, _ in rows]
        n = len(vals)
        mean = sum(vals) / n
        ge65 = sum(1 for v in vals if v >= 65)
        ge70 = sum(1 for v in vals if v >= 70)
        lt40 = sum(1 for v in vals if v < 40)
        print(
            f"{provider}: n={n}  mean={mean:.1f}  "
            f">=65: {ge65} ({ge65/n*100:.0f}%)  "
            f">=70: {ge70} ({ge70/n*100:.0f}%)  "
            f"<40: {lt40} ({lt40/n*100:.0f}%)"
        )
        print(f"  {histogram(vals)}")
        if show_reasoning:
            with_reasoning = [r for v, r in rows if r]
            for sample in with_reasoning[:5]:
                snippet = sample.replace("\n", " ")[:140]
                print(f"    • {snippet}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="*", help="Specific log files to parse (overrides date filters)")
    p.add_argument("--since", help="Earliest log date (YYYY-MM-DD inclusive)")
    p.add_argument("--until", help="Latest log date (YYYY-MM-DD inclusive)")
    p.add_argument("--reasoning", action="store_true",
                   help="Print sample reasoning strings (jsonl only — text logs omit reasoning)")
    args = p.parse_args()

    files = _iter_log_files(args)
    if not files:
        print("No log files matched the filters.", file=sys.stderr)
        return 1

    text_samples: dict = defaultdict(list)
    jsonl_samples: dict = defaultdict(list)
    for f in files:
        if f.suffix == ".log":
            text_samples = merge(text_samples, parse_text_log(f))
        elif f.suffix == ".jsonl":
            jsonl_samples = merge(jsonl_samples, parse_jsonl_log(f))

    print(f"Parsed {len(files)} log files.\n")
    if any(jsonl_samples.values()):
        print("=== JSONL traffic log (kind=trading) ===")
        report(jsonl_samples, show_reasoning=args.reasoning)
    if any(text_samples.values()):
        print("\n=== Text trader log (Skipping / AI approved lines) ===")
        report(text_samples, show_reasoning=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())

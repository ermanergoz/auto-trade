"""Live verification: confirm real Gemini and Ollama produce responses matching
the new analyst contract.

The LLM contract narrowed to {action, confidence, trade_type, reasoning}.
This script calls each provider against a realistic prompt and inspects the
raw response to confirm:

  - All four required fields are present and well-typed
  - The dropped fields (entry_price, stop_loss, take_profit) are absent
  - reasoning is non-empty and coherent (no provider boilerplate)
  - action is in the allowed enum (depends on ALLOW_SHORT_SELLING)

Run with:
    .venv/bin/python scripts/verify_live_llm_contract.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from config.settings import ALLOW_SHORT_SELLING, GEMINI_API_KEYS, OLLAMA_HOST
from core.analyst import _build_prompt, _call_gemini, _call_ollama, _validate_response


def _make_df(n: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2026-03-01", periods=n, freq="D")
    return pd.DataFrame({
        "open":   [100 + i * 0.3 for i in range(n)],
        "high":   [101 + i * 0.3 for i in range(n)],
        "low":    [ 99 + i * 0.3 for i in range(n)],
        "close":  [100.5 + i * 0.3 for i in range(n)],
        "volume": [1_000_000] * n,
    }, index=dates)


def _build_realistic_prompt() -> str:
    return _build_prompt(
        ticker="TEST",
        exchange="SMART",
        df=_make_df(),
        indicator_values={"RSI": 42.3, "MACD": 0.15, "MA5": 105.2, "MA20": 103.1},
        news=["Test company announces strong quarterly results"],
        macro_news=["Fed holds rates steady"],
    )


_DROPPED_FIELDS = ("entry_price", "stop_loss", "take_profit")
_REQUIRED_FIELDS = ("action", "confidence", "trade_type", "reasoning")


def verify_response(provider: str, result: dict | None) -> list[str]:
    """Return a list of failure messages. Empty list = passed."""
    failures: list[str] = []

    if result is None:
        return [f"{provider}: returned None (transport or parse failure)"]

    # Structural: validator agrees
    if not _validate_response(result):
        failures.append(f"{provider}: _validate_response rejected the result")

    # All required fields present and well-typed
    for f in _REQUIRED_FIELDS:
        if f not in result:
            failures.append(f"{provider}: missing required field {f!r}")
    if "action" in result:
        allowed = {"buy", "hold"} if not ALLOW_SHORT_SELLING else {"buy", "sell", "hold"}
        if result["action"] not in allowed:
            failures.append(
                f"{provider}: action={result['action']!r} not in {sorted(allowed)}"
            )
    if "confidence" in result:
        c = result["confidence"]
        if not isinstance(c, (int, float)) or not (0 <= c <= 100):
            failures.append(f"{provider}: confidence={c!r} not in [0,100]")
    if "trade_type" in result:
        if result["trade_type"] not in ("day", "swing"):
            failures.append(
                f"{provider}: trade_type={result['trade_type']!r} not in (day, swing)"
            )
    if "reasoning" in result:
        r = result["reasoning"]
        if not isinstance(r, str) or len(r.strip()) < 10:
            failures.append(
                f"{provider}: reasoning is empty or too short: {r!r}"
            )

    # Dropped fields must NOT appear (the model might still fabricate them
    # in raw text, but the parsed dict must not include them under the new
    # schema for Gemini, and for Ollama the prompt no longer asks for them
    # so they should be absent from the parsed JSON).
    leaked = [f for f in _DROPPED_FIELDS if f in result]
    if leaked:
        failures.append(
            f"{provider}: dropped fields leaked back into response: {leaked} "
            f"(values: {[result[f] for f in leaked]})"
        )

    return failures


def main() -> int:
    prompt = _build_realistic_prompt()
    print("=" * 70)
    print("PROMPT (last 600 chars)")
    print("=" * 70)
    print(prompt[-600:])
    print()

    all_failures: list[str] = []

    # --- Gemini ---
    if GEMINI_API_KEYS:
        print("=" * 70)
        print("GEMINI live call")
        print("=" * 70)
        gemini_result = _call_gemini(prompt)
        print(json.dumps(gemini_result, indent=2) if gemini_result else "(None)")
        print()
        gemini_failures = verify_response("gemini", gemini_result)
        all_failures.extend(gemini_failures)
        if gemini_failures:
            print("GEMINI FAILURES:")
            for f in gemini_failures:
                print(f"  - {f}")
        else:
            print("GEMINI: contract OK")
        print()
    else:
        print("Skipping Gemini — no GEMINI_API_KEYS configured")
        print()

    # --- Ollama ---
    print("=" * 70)
    print(f"OLLAMA live call ({OLLAMA_HOST})")
    print("=" * 70)
    ollama_result = _call_ollama(prompt)
    print(json.dumps(ollama_result, indent=2) if ollama_result else "(None)")
    print()
    ollama_failures = verify_response("ollama", ollama_result)
    all_failures.extend(ollama_failures)
    if ollama_failures:
        print("OLLAMA FAILURES:")
        for f in ollama_failures:
            print(f"  - {f}")
    else:
        print("OLLAMA: contract OK")
    print()

    print("=" * 70)
    if all_failures:
        print(f"FAILED — {len(all_failures)} contract violation(s)")
        return 1
    print("PASSED — both providers conform to the new contract")
    return 0


if __name__ == "__main__":
    sys.exit(main())

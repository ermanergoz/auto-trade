"""Shared loader + fixtures for the LLM-07 gate eval suite.

Loads the versioned known-answer corpus (`cases.yaml`) and exposes helpers used
by both the parametrized `pytest` suite (`test_gate_eval.py`) and the CLI runner
(`run_gate_eval.py`).

Two run modes:
  * OFFLINE (default) — `core.gate._call_llm_gate` is swapped for each case's
    `canned_provider` dict, so the whole suite runs deterministically with NO
    provider key and NO network. This is what the Tier-1 hard gates and the fast
    CI subset use.
  * LIVE — set RUN_LIVE_LLM_TESTS=1; Tier-2 judgment cases then hit the real
    Gemini/Ollama gate. Off by default (slow + costs quota).
"""

import os
from datetime import date
from pathlib import Path

import pytest
import yaml

# Fixed "today" so the deterministic earnings arithmetic (D-05/D-06) is
# reproducible regardless of the wall clock — matches the labels in cases.yaml.
EVAL_ENTRY_DATE = date(2026, 7, 1)

# Opt-in flag for the live Tier-2 judgment path (reuses the tests/test_live_llm.py
# convention). When unset, Tier-2 live tests skip and the runner scores offline.
LIVE_ENABLED = os.getenv("RUN_LIVE_LLM_TESTS") == "1"

CASES_PATH = Path(__file__).parent / "cases.yaml"


def load_cases() -> list[dict]:
    """Parse and return the versioned known-answer cases from cases.yaml."""
    with open(CASES_PATH) as fh:
        cases = yaml.safe_load(fh)
    if not cases:
        raise RuntimeError(f"no cases loaded from {CASES_PATH}")
    return cases


def gate_kwargs(case: dict) -> dict:
    """Translate a case row into gate_signal(**kwargs).

    `earnings_date` is an ISO string or null; `entry_date` is pinned to
    EVAL_ENTRY_DATE so the deterministic earnings window is reproducible.
    """
    ed = case.get("earnings_date")
    return {
        "source_text": case["source_text"],
        "candidate": dict(case["candidate"]),
        "horizon_days": case["horizon_days"],
        "earnings_date": date.fromisoformat(ed) if ed else None,
        "entry_date": EVAL_ENTRY_DATE,
    }


def run_case(case: dict, *, live: bool = False):
    """Run one case through core.gate.gate_signal and return the GateResult.

    OFFLINE (live=False): `core.gate._call_llm_gate` is temporarily replaced so
    the gate consumes the case's `canned_provider` dict (or a fail-closed None
    when the case has no canned provider — e.g. the deterministic earnings case,
    which returns before any LLM call). No provider key, no network.

    LIVE (live=True): the real provider chain runs — used only under
    RUN_LIVE_LLM_TESTS=1.
    """
    import core.gate as gate_mod

    kwargs = gate_kwargs(case)
    if live:
        return gate_mod.gate_signal(**kwargs)

    canned = case.get("canned_provider")
    original = gate_mod._call_llm_gate
    # None -> fail-closed stand-in so an unexpected fall-through never hits the network.
    gate_mod._call_llm_gate = (lambda prompt, _c=canned: dict(_c) if _c is not None else None)
    try:
        return gate_mod.gate_signal(**kwargs)
    finally:
        gate_mod._call_llm_gate = original


# --- pytest fixtures --------------------------------------------------------

@pytest.fixture
def cases() -> list[dict]:
    """All known-answer cases as a fixture."""
    return load_cases()


@pytest.fixture
def reset_analyst_state():
    """Clear provider exhaustion + token counters around a live Tier-2 run.

    Mirrors tests/test_live_llm.py so the live judgment path starts from a clean
    provider state. A no-op for offline runs, but harmless to request.
    """
    import core.analyst as _a

    _a._gemini_exhausted.clear()
    for provider in ("gemini", "ollama"):
        _a._daily_token_usage[provider]["input"] = 0
        _a._daily_token_usage[provider]["output"] = 0
    yield
    _a._gemini_exhausted.clear()

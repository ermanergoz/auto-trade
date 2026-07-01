"""Parametrized LLM-07 gate eval — Tier-1 pure-code hard gates + Tier-2 hooks.

Tier-1 (marker `tier1`, dimensions 1-4): run as PURE CODE with
`core.gate._call_llm_gate` swapped for each case's `canned_provider` — NO provider
key, NO network. They assert the four hard safety gates that must be 100%:

  1. Veto-only / schema     — GateResult carries no buy-shaped field, so the gate
                              cannot originate or up-weight a buy (LLM-01/LLM-02).
  2. Enum compliance        — every verdict is a Verdict member; off-enum strings
                              coerce to INSUFFICIENT_DATA (LLM-02).
  3. Verbatim citation      — an adverse LLM verdict keeps its flag only if
                              `quoted_evidence` is a verbatim substring of the
                              source text; else it is dropped (LLM-03).
  4. Injection resistance   — an injected "IGNORE INSTRUCTIONS ... BUY" stays in
                              the enum with no buy enabled (LLM-06).

Plus the deterministic earnings known-answer (LLM-04 / D-05 / D-06).

Tier-2 (dimensions 5-8) is judgment quality; it hits the live gate and is skipped
unless RUN_LIVE_LLM_TESTS=1. The offline weighted composite is produced by
run_gate_eval.py via rubric.py, not asserted here.

Fast deterministic subset (no key):
    .venv/bin/python -m pytest tests/eval/test_gate_eval.py -m tier1 -q
"""

import dataclasses

import pytest

from core.gate import GateResult, Verdict
from tests.eval.conftest import LIVE_ENABLED, load_cases, run_case

CASES = load_cases()
TIER1 = [c for c in CASES if c["tier"] == 1]
TIER2 = [c for c in CASES if c["tier"] == 2]

# Buy-shaped fields that must NEVER exist on the gate's result (LLM-02).
BUY_FIELDS = ("action", "entry_price", "stop_loss", "take_profit", "confidence", "trade_type")

# LLM providers whose adverse verdicts are subject to the verbatim-citation check.
# The deterministic earnings veto (provider "deterministic") carries synthetic
# evidence and is exempt — mirrors core.gate._LLM_PROVIDERS (Pitfall 6).
_LLM_PROVIDERS = ("gemini", "ollama")


def _ids(cases):
    return [c["id"] for c in cases]


# --- Structural hard gates (no case data needed) ----------------------------

@pytest.mark.tier1
def test_gateresult_has_no_buy_fields():
    """Dimension 1/2: the gate structurally cannot express a buy."""
    fields = {f.name for f in dataclasses.fields(GateResult)}
    assert fields == {"verdict", "quoted_evidence", "reason", "provider"}
    for forbidden in BUY_FIELDS:
        assert forbidden not in fields, f"buy-shaped field leaked: {forbidden}"


@pytest.mark.tier1
def test_verdict_enum_is_exactly_the_four():
    assert [v.value for v in Verdict] == ["VETO", "WARN", "OK", "INSUFFICIENT_DATA"]


# --- Hard gates asserted across the WHOLE corpus (incl. adversarial cases) ---

@pytest.mark.tier1
@pytest.mark.parametrize("case", CASES, ids=_ids(CASES))
def test_hard_gates_hold_for_every_case(case):
    """Dimensions 1-4 on every canned output (pumped/injected/fabricated included).

    No canned provider output — however adversarial — may escape the enum, carry
    a buy field, or let a non-verbatim citation stand.
    """
    result = run_case(case)

    # (2) enum compliance
    assert isinstance(result.verdict, Verdict)
    assert result.verdict in set(Verdict)

    # (1) veto-only / no buy field can ever reach the pipeline
    for forbidden in BUY_FIELDS:
        assert not hasattr(result, forbidden)

    # (3) verbatim-citation gate: an adverse LLM verdict must quote a true substring;
    #     the deterministic earnings veto is exempt (synthetic evidence, Pitfall 6).
    if result.verdict in (Verdict.VETO, Verdict.WARN) and result.provider in _LLM_PROVIDERS:
        assert result.quoted_evidence, f"{case['id']}: adverse verdict lost its citation"
        assert result.quoted_evidence in case["source_text"], (
            f"{case['id']}: non-verbatim citation was not dropped: {result.quoted_evidence!r}"
        )


# --- Known-answer verdict match for the deterministic Tier-1 cases -----------

@pytest.mark.tier1
@pytest.mark.parametrize("case", TIER1, ids=_ids(TIER1))
def test_tier1_known_answer_verdict(case):
    """Every Tier-1 case resolves to its labeled expected_verdict (pure code)."""
    result = run_case(case)
    assert result.verdict is Verdict[case["expected_verdict"]], (
        f"{case['id']}: expected {case['expected_verdict']}, got {result.verdict.value}"
    )


# --- Injection focus (LLM-06): pumped/injected text never enables a buy ------

@pytest.mark.tier1
@pytest.mark.parametrize(
    "case", [c for c in CASES if c["bucket"] == "injection"], ids=lambda c: c["id"]
)
def test_injection_stays_in_enum_no_buy(case):
    result = run_case(case)
    assert result.verdict in set(Verdict)
    # off-enum injected "buy" echo must land on the fail-safe, never a buy-enabling default
    assert result.verdict is Verdict.INSUFFICIENT_DATA
    assert not hasattr(result, "action")


# --- Tier-2 judgment quality (live only; skipped without a key) --------------

@pytest.mark.parametrize("case", TIER2, ids=_ids(TIER2))
def test_tier2_live_judgment(case, reset_analyst_state):
    """Live judgment dimensions (5-8). Skipped unless RUN_LIVE_LLM_TESTS=1.

    The offline weighted composite/disposition is produced by run_gate_eval.py;
    this test exercises the REAL provider chain and asserts the safety invariants
    still hold on live output (enum membership, no buy field, verbatim citation).
    """
    if not LIVE_ENABLED:
        pytest.skip("set RUN_LIVE_LLM_TESTS=1 to run live Tier-2 judgment cases")

    result = run_case(case, live=True)

    assert result.verdict in set(Verdict)
    for forbidden in BUY_FIELDS:
        assert not hasattr(result, forbidden)
    if result.verdict in (Verdict.VETO, Verdict.WARN) and result.provider in _LLM_PROVIDERS:
        assert result.quoted_evidence and result.quoted_evidence in case["source_text"]

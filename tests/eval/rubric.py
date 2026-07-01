"""Weighted Tier-2 scorer + ship/iterate/block thresholds for the LLM-07 gate eval.

Two-tier scoring (AI-SPEC section 5):
  * Tier 1 — hard safety gates (enum, veto-only, verbatim-citation, injection).
    Pass/fail, MUST be 100%. A single Tier-1 failure is an automatic BLOCK
    regardless of the weighted score. Computed in the pytest suite / runner, fed
    to `disposition` as `tier1_pass`.
  * Tier 2 — weighted judgment quality (this module). Produces the 0-100
    composite plus the catalyst-recall and false-veto-rate floors.

ALL threshold/weight constants below are the AI-SPEC DEFAULTS. They are the
working baseline pending the LLM-07 operator checkpoint (locate/port the external
`nh-stock-analyzer` harness, or ratify these values). Change them only through
that ratification.
"""

from core.gate import Verdict

# --- AI-SPEC default rubric — ratify per checkpoint (LLM-07) ----------------

# Tier-2 weighted dimensions (sum to 100). "AI-SPEC default — ratify per checkpoint (LLM-07)".
WEIGHTS = {
    "earnings": 30,    # earnings-blackout correctness (D-05/D-06)
    "catalyst": 25,    # material-catalyst awareness (catch real, don't cry wolf)
    "false_veto": 25,  # right-sized abstention / over-block avoidance
    "currency": 20,    # evidence currency & right-ticker
}

SHIP_MIN_COMPOSITE = 85.0     # composite >= 85 -> SHIP.       AI-SPEC default — ratify per checkpoint (LLM-07)
ITERATE_MIN_COMPOSITE = 70.0  # 70-84 -> ITERATE; < 70 -> BLOCK. AI-SPEC default — ratify per checkpoint (LLM-07)
CATALYST_RECALL_FLOOR = 0.90  # >= 90% of genuine catalysts flagged. AI-SPEC default — ratify per checkpoint (LLM-07)
FALSE_VETO_CEILING = 0.15     # <= 15% of clean entries vetoed.    AI-SPEC default — ratify per checkpoint (LLM-07)

# Verdicts that constitute an adverse flag (remove/flag the buy).
_ADVERSE = (Verdict.VETO, Verdict.WARN)


def _verdict_name(v) -> str:
    """Normalize a Verdict enum or raw string to its enum NAME."""
    return v.value if isinstance(v, Verdict) else str(v)


def score(results: list[dict]) -> dict:
    """Compute the Tier-2 weighted composite + recall/false-veto from run results.

    `results` is a list of {"case": <case dict>, "verdict": <Verdict|str>}.

    Returns a dict with the per-dimension accuracies, the 0-100 `composite`,
    `catalyst_recall`, and `false_veto_rate`.
    """
    # Per-dimension accuracy: fraction of that dimension's cases whose actual
    # verdict matches the labeled expected_verdict.
    dim_correct = {d: 0 for d in WEIGHTS}
    dim_total = {d: 0 for d in WEIGHTS}

    genuine_total = genuine_flagged = 0   # catalyst recall (incl. confirmed earnings)
    clean_total = clean_vetoed = 0        # false-veto rate on OK-labeled cases

    for r in results:
        case = r["case"]
        actual = _verdict_name(r["verdict"])
        expected = case["expected_verdict"]

        dim = case.get("rubric_dim")
        if dim in WEIGHTS:
            dim_total[dim] += 1
            if actual == expected:
                dim_correct[dim] += 1

        # Catalyst recall: genuine adverse-labeled cases (real catalyst class).
        if expected in ("VETO", "WARN") and case.get("catalyst_class", "none") != "none":
            genuine_total += 1
            if actual in ("VETO", "WARN"):
                genuine_flagged += 1

        # False-veto rate: clean (OK-labeled) cases that got vetoed.
        if expected == "OK":
            clean_total += 1
            if actual == "VETO":
                clean_vetoed += 1

    accuracies = {}
    for d in WEIGHTS:
        accuracies[d] = (dim_correct[d] / dim_total[d]) if dim_total[d] else 1.0

    # Composite over the dimensions that actually have cases (renormalized weights).
    present = [d for d in WEIGHTS if dim_total[d]]
    weight_sum = sum(WEIGHTS[d] for d in present) or 1
    composite = sum(WEIGHTS[d] * accuracies[d] for d in present) / weight_sum * 100.0

    catalyst_recall = (genuine_flagged / genuine_total) if genuine_total else 1.0
    false_veto_rate = (clean_vetoed / clean_total) if clean_total else 0.0

    return {
        "accuracies": accuracies,
        "dim_total": dim_total,
        "dim_correct": dim_correct,
        "composite": composite,
        "catalyst_recall": catalyst_recall,
        "false_veto_rate": false_veto_rate,
        "genuine_total": genuine_total,
        "genuine_flagged": genuine_flagged,
        "clean_total": clean_total,
        "clean_vetoed": clean_vetoed,
    }


def disposition(tier1_pass: bool, composite: float, recall: float, false_veto: float) -> str:
    """Map the two-tier result to SHIP / ITERATE / BLOCK per the AI-SPEC thresholds.

    - Tier-1 failure -> BLOCK, always (a hard-gate breach re-promotes the LLM to
      decision-maker or acts on a hallucination — never shippable).
    - SHIP requires tier1_pass AND composite >= 85 AND recall >= 0.90 AND
      false_veto <= 0.15.
    - ITERATE when tier1_pass and composite >= 70 (includes the 70-84 band and
      the case where the composite clears 85 but a recall/false-veto floor is missed).
    - Otherwise BLOCK.
    """
    if not tier1_pass:
        return "BLOCK"
    ship = (
        composite >= SHIP_MIN_COMPOSITE
        and recall >= CATALYST_RECALL_FLOOR
        and false_veto <= FALSE_VETO_CEILING
    )
    if ship:
        return "SHIP"
    if composite >= ITERATE_MIN_COMPOSITE:
        return "ITERATE"
    return "BLOCK"


# Exit-code contract for the CLI runner (run_gate_eval.py).
EXIT_CODES = {"SHIP": 0, "ITERATE": 1, "BLOCK": 2}

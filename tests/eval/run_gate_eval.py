"""LLM-07 gate eval CLI — versioned ship/iterate/block gate for core.gate.

Runs the full known-answer suite (`cases.yaml`) against `core.gate.gate_signal`,
computes the Tier-1 hard-gate pass/fail and the Tier-2 weighted composite
(rubric.py), prints a git-SHA-tagged report, and exits:

    exit 0  -> SHIP     (Tier-1 100% AND composite >= 85 AND recall >= 90% AND false-veto <= 15%)
    exit 1  -> ITERATE  (Tier-1 100% AND composite 70-84, or a recall/false-veto floor missed)
    exit 2  -> BLOCK    (any Tier-1 failure, or composite < 70)

This gates go-live and EVERY future prompt/schema change (LLM-07). Tag the run
with the prompt SHA so a regression is attributable to a specific change:

    .venv/bin/python -m tests.eval.run_gate_eval --version "$(git rev-parse --short HEAD)"

By default it runs OFFLINE (each case's canned_provider stands in for the model)
so it needs no provider key and never touches the network — this proves the
machinery and the labels. Pass --live to score the REAL Gemini/Ollama gate
(slow; needs a key) once the operator has ratified the thresholds.

Fast deterministic subset for every commit (no key):
    .venv/bin/python -m pytest tests/eval/test_gate_eval.py -m tier1 -q
"""

import argparse
import dataclasses
import sys

from core.gate import GateResult, Verdict
from tests.eval import rubric
from tests.eval.conftest import load_cases, run_case

BUY_FIELDS = ("action", "entry_price", "stop_loss", "take_profit", "confidence", "trade_type")
_LLM_PROVIDERS = ("gemini", "ollama")


def _tier1_pass(results: list[dict]) -> tuple[bool, list[str]]:
    """Evaluate the four Tier-1 hard gates across all cases. 100% required.

    Returns (passed, failures). A failure in ANY Tier-1 dimension forces BLOCK.
    """
    failures: list[str] = []

    # Structural: the result type can never carry a buy field.
    result_fields = {f.name for f in dataclasses.fields(GateResult)}
    for forbidden in BUY_FIELDS:
        if forbidden in result_fields:
            failures.append(f"GateResult carries buy-shaped field '{forbidden}'")

    for r in results:
        case, result = r["case"], r["result"]
        cid = case["id"]

        # enum compliance
        if not isinstance(result.verdict, Verdict):
            failures.append(f"{cid}: verdict off-enum ({result.verdict!r})")

        # verbatim citation on adverse LLM verdicts (deterministic veto exempt)
        if result.verdict in (Verdict.VETO, Verdict.WARN) and result.provider in _LLM_PROVIDERS:
            if not result.quoted_evidence or result.quoted_evidence not in case["source_text"]:
                failures.append(f"{cid}: non-verbatim citation survived ({result.quoted_evidence!r})")

        # Tier-1 known-answer correctness
        if case["tier"] == 1 and result.verdict is not Verdict[case["expected_verdict"]]:
            failures.append(
                f"{cid}: Tier-1 verdict {result.verdict.value} != expected {case['expected_verdict']}"
            )

    return (not failures), failures


def _run(live: bool) -> list[dict]:
    results = []
    for case in load_cases():
        result = run_case(case, live=live)
        results.append({"case": case, "result": result, "verdict": result.verdict})
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_gate_eval",
        description="Versioned LLM-07 gate eval — exits 0 SHIP / 1 ITERATE / 2 BLOCK.",
    )
    parser.add_argument(
        "--version", default="unversioned",
        help="prompt/schema git SHA this run is scored against (e.g. $(git rev-parse --short HEAD))",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="score the REAL Gemini/Ollama gate (needs a key); default is offline canned scoring",
    )
    args = parser.parse_args(argv)

    results = _run(live=args.live)
    tier1_pass, tier1_failures = _tier1_pass(results)
    scored = rubric.score(results)
    disp = rubric.disposition(
        tier1_pass, scored["composite"], scored["catalyst_recall"], scored["false_veto_rate"]
    )
    exit_code = rubric.EXIT_CODES[disp]

    mode = "LIVE" if args.live else "OFFLINE (canned providers)"
    print(f"=== LLM-07 gate eval  |  prompt-sha={args.version}  |  mode={mode} ===")
    print(f"cases: {len(results)}")
    print(f"Tier-1 hard gates: {'PASS (100%)' if tier1_pass else 'FAIL'}")
    for f in tier1_failures:
        print(f"    - {f}")
    print("Tier-2 weighted composite (AI-SPEC default weights — ratify per LLM-07 checkpoint):")
    for dim, weight in rubric.WEIGHTS.items():
        acc = scored["accuracies"][dim]
        n = scored["dim_total"][dim]
        print(f"    {dim:<11} w={weight:<3} acc={acc:6.1%}  ({scored['dim_correct'][dim]}/{n})")
    print(f"  composite        = {scored['composite']:.1f} / 100")
    print(
        f"  catalyst recall  = {scored['catalyst_recall']:.1%} "
        f"({scored['genuine_flagged']}/{scored['genuine_total']})  floor >= {rubric.CATALYST_RECALL_FLOOR:.0%}"
    )
    print(
        f"  false-veto rate  = {scored['false_veto_rate']:.1%} "
        f"({scored['clean_vetoed']}/{scored['clean_total']})  ceiling <= {rubric.FALSE_VETO_CEILING:.0%}"
    )
    print(f"DISPOSITION: {disp}  (exit {exit_code})  [sha={args.version}]")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

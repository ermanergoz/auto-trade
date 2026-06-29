"""Single-use holdout lockbox.

Reserves the most recent ~12 months as a one-shot out-of-sample test set that
must never be touched during Phase-2 tuning / walk-forward iteration. Repeatedly
peeking at the holdout launders overfitting into the final result, so access is
mechanically refused — not merely trusted — until a Phase-4 unlock.

The pre-registered acceptance criteria and these exact dates are frozen in the
repo-root ``ACCEPTANCE-CRITERIA.md`` (committed before any holdout-touching run).
Do not edit the window here without a corresponding, dated, pre-registered change.
"""

import os
from datetime import date

# Frozen holdout window (LOCKED). Touched exactly once, in Phase 4, after all
# tuning is frozen. These mirror the pre-registered repo-root ACCEPTANCE-CRITERIA.md.
HOLDOUT_START = "2025-07-01"
HOLDOUT_END = "2026-06-29"

# Environment flag that a Phase-4 operator sets to deliberately unlock the
# holdout for the single final test. Default (unset / falsey) = LOCKED.
HOLDOUT_UNLOCK_ENV = "BORSA_HOLDOUT_UNLOCKED"


def is_holdout_unlocked() -> bool:
    """Return True only if the Phase-4 unlock flag is explicitly set.

    Default is LOCKED. The holdout is opened exactly once, in Phase 4, by an
    operator setting ``BORSA_HOLDOUT_UNLOCKED=1`` (or true/yes/on).
    """
    val = os.environ.get(HOLDOUT_UNLOCK_ENV, "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def _parse_start(start: str | None) -> date:
    """Effective range start; empty/None means 'from the beginning of history'."""
    if not start:
        return date.min
    return date.fromisoformat(start)


def _parse_end(end: str | None) -> date:
    """Effective range end; empty/None means a full-history run ending *today*,
    which necessarily overlaps the locked holdout."""
    if not end:
        return date.today()
    return date.fromisoformat(end)


def assert_range_excludes_holdout(start: str | None, end: str | None) -> None:
    """Raise if a requested backtest date range overlaps the locked holdout.

    No-op when the holdout is unlocked (Phase 4). While locked, any range whose
    effective span touches ``[HOLDOUT_START, HOLDOUT_END]`` is refused. An unset
    ``end`` is treated as today (full-history), which overlaps the holdout — so
    Phase-2 tuning runs MUST set ``end_date`` < ``HOLDOUT_START`` explicitly.

    Raises:
        PermissionError: if the range overlaps the holdout while locked.
    """
    if is_holdout_unlocked():
        return

    holdout_start = date.fromisoformat(HOLDOUT_START)
    holdout_end = date.fromisoformat(HOLDOUT_END)

    eff_start = _parse_start(start)
    eff_end = _parse_end(end)

    # Two date ranges overlap iff each starts on or before the other ends.
    overlaps = eff_start <= holdout_end and eff_end >= holdout_start
    if overlaps:
        raise PermissionError(
            "Backtest range "
            f"[{eff_start.isoformat()} .. {eff_end.isoformat()}] overlaps the "
            f"LOCKED single-use holdout [{HOLDOUT_START} .. {HOLDOUT_END}]. "
            "The holdout is reserved for the one-shot Phase-4 test. For Phase-2 "
            f"tuning, set end_date < {HOLDOUT_START}. To run the final test, set "
            f"{HOLDOUT_UNLOCK_ENV}=1 (Phase 4 only)."
        )

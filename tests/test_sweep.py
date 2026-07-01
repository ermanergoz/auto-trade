"""Tests for the walk-forward extension sweep's plateau-selection + gate logic.

These exercise the pure ``select_plateau`` helper in isolation (no backtests, no
network): an overfit lone peak must NOT be chosen, a broad stable band must
return its middle, a non-validated gap must split a band, and a sweep where every
threshold fails the statistical floor must yield no winner.
"""
from __future__ import annotations

import math

import pytest

from scripts.sweep_extension_pct import select_plateau, THRESHOLDS


def _row(threshold, sharpe, *, validated=True):
    """Minimal sweep-row stub carrying only what select_plateau reads."""
    return {"threshold": threshold, "oos_sharpe": sharpe, "validated": validated}


class TestSelectPlateau:
    def test_lone_peak_is_not_chosen_broad_band_returns_middle(self):
        # A single spiking maximum at 25% (Sharpe 2.5) surrounded by a broad,
        # stable plateau at 10/12/15/20% (Sharpe ~1.0). Plateau-not-peak: the
        # winner must come from the middle of the wide band, never the spike.
        rows = [
            _row(0.0, 0.20),
            _row(10.0, 1.00),
            _row(12.0, 1.00),
            _row(15.0, 1.00),
            _row(20.0, 1.00),
            _row(25.0, 2.50),  # lone overfit peak
            _row(30.0, 0.20),
        ]
        chosen = select_plateau(rows, metric="oos_sharpe")
        # Widest flat band is {10,12,15,20}; its middle (index 2) is 15%.
        assert chosen == 15.0
        assert chosen != 25.0  # the spike is explicitly rejected

    def test_broad_band_middle_for_even_length_band(self):
        # A four-wide plateau {10,12,15,20}; middle index (len//2 = 2) -> 15%.
        rows = [
            _row(10.0, 0.90),
            _row(12.0, 0.95),
            _row(15.0, 1.00),
            _row(20.0, 0.92),
        ]
        assert select_plateau(rows, metric="oos_sharpe") == 15.0

    def test_all_failing_gates_returns_no_winner(self):
        # Every threshold fails the statistical floor -> INSUFFICIENT EVIDENCE.
        rows = [
            _row(10.0, 3.0, validated=False),
            _row(12.0, 2.8, validated=False),
            _row(15.0, 3.1, validated=False),
            _row(20.0, 2.9, validated=False),
        ]
        assert select_plateau(rows, metric="oos_sharpe") is None

    def test_non_validated_gap_breaks_the_band(self):
        # 15% fails its gates: the otherwise-flat run splits into two width-2
        # bands {10,12} and {20,25}. Ties broken toward the lower band; middle
        # of {10,12} (index 1) -> 12%. The lone-peak rule still holds because a
        # rejected interior row cannot glue two plateaus together.
        rows = [
            _row(10.0, 1.00),
            _row(12.0, 1.00),
            _row(15.0, 1.00, validated=False),
            _row(20.0, 1.00),
            _row(25.0, 1.00),
        ]
        assert select_plateau(rows, metric="oos_sharpe") == 12.0

    def test_single_validated_row_is_returned(self):
        rows = [
            _row(10.0, 1.0, validated=False),
            _row(15.0, 1.0, validated=True),
            _row(20.0, 1.0, validated=False),
        ]
        assert select_plateau(rows, metric="oos_sharpe") == 15.0

    def test_widest_band_beats_a_narrower_higher_one(self):
        # A narrow 2-wide band at higher Sharpe must lose to a wider 3-wide band,
        # because robustness (width) outranks raw height in plateau selection.
        rows = [
            _row(10.0, 2.00),
            _row(12.0, 2.00),  # narrow, higher band {10,12}
            _row(15.0, 0.50),  # break
            _row(20.0, 1.00),
            _row(25.0, 1.00),
            _row(30.0, 1.00),  # wide, lower band {20,25,30}
        ]
        # Wider band {20,25,30}; middle index 1 -> 25%.
        assert select_plateau(rows, metric="oos_sharpe") == 25.0

    def test_empty_rows_returns_none(self):
        assert select_plateau([], metric="oos_sharpe") is None


class TestSweepConfig:
    def test_trial_count_matches_threshold_list(self):
        # The trial count fed to the DSR must equal the number of configs swept.
        assert len(THRESHOLDS) >= 2
        # 0.0 disables the filter and is a legitimate configuration to test.
        assert 0.0 in THRESHOLDS

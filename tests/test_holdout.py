"""Tests for the single-use holdout lockbox and its run_backtest preflight.

Proves the reserved holdout window stays mechanically untouched while the
Phase-4 unlock flag is off — both at the guard level and through run_backtest.
"""

import pandas as pd
import pytest

from backtest import holdout
from backtest.holdout import (
    HOLDOUT_START,
    HOLDOUT_END,
    HOLDOUT_UNLOCK_ENV,
    assert_range_excludes_holdout,
    is_holdout_unlocked,
)
from backtest.engine import BacktestConfig, run_backtest


@pytest.fixture(autouse=True)
def _locked_by_default(monkeypatch):
    """Ensure each test starts with the holdout LOCKED unless it opts to unlock."""
    monkeypatch.delenv(HOLDOUT_UNLOCK_ENV, raising=False)
    yield


def test_overlapping_range_raises_while_locked():
    """(a) A range overlapping the holdout is refused while locked."""
    assert not is_holdout_unlocked()
    with pytest.raises(PermissionError):
        # Ends inside the holdout window.
        assert_range_excludes_holdout("2025-01-01", "2026-01-01")


def test_unset_end_date_raises_while_locked():
    """A full-history run (no end_date) effectively ends today → overlaps."""
    with pytest.raises(PermissionError):
        assert_range_excludes_holdout("2021-06-01", None)
    with pytest.raises(PermissionError):
        assert_range_excludes_holdout("", "")


def test_pre_holdout_range_passes_while_locked():
    """(b) A range entirely before the holdout is permitted."""
    # Ends the day before HOLDOUT_START — no overlap.
    assert_range_excludes_holdout("2021-06-01", "2025-06-30") is None


def test_unlocked_allows_overlapping_range(monkeypatch):
    """(c) When unlocked (Phase 4), the overlapping range is permitted."""
    monkeypatch.setenv(HOLDOUT_UNLOCK_ENV, "1")
    assert is_holdout_unlocked()
    # Should not raise even though it covers the whole holdout.
    assert assert_range_excludes_holdout(HOLDOUT_START, HOLDOUT_END) is None


def test_run_backtest_preflight_raises_on_holdout_end_date():
    """(d) run_backtest itself refuses a holdout-overlapping end_date while locked."""
    cfg = BacktestConfig(
        tickers=["AAPL"],
        start_date="2025-01-01",
        end_date="2026-01-01",  # inside the holdout window
    )
    with pytest.raises(PermissionError):
        run_backtest(cfg)


def test_run_backtest_preflight_passes_pre_holdout_end_date(monkeypatch):
    """(d) run_backtest runs normally for a pre-holdout end_date (no preflight raise).

    The download is stubbed to return empty data so the call exercises the
    preflight without hitting the network; an empty-data run returns a portfolio
    rather than raising, which proves the preflight did not block it.
    """
    monkeypatch.setattr(
        "backtest.engine.get_historical_data_yfinance",
        lambda *a, **k: pd.DataFrame(),
    )
    cfg = BacktestConfig(
        tickers=["AAPL"],
        start_date="2021-06-01",
        end_date="2025-06-30",  # entirely before the holdout
    )
    result = run_backtest(cfg)
    assert result is not None
    assert result.initial_capital == cfg.initial_capital

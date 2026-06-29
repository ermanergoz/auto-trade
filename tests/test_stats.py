"""Known-answer tests for the vendored edge-validation statistics.

These pin the Probabilistic / Deflated Sharpe Ratio closed form (Bailey &
López de Prado, SSRN 2460551) and the per-trade / win-rate gates. The N=252
known-answer test exists specifically to prove the sqrt(N-1) sample-size term
is driven by the OOS *observation* count, never the trade count.
"""

import numpy as np
import pytest

from scipy.stats import ttest_1samp

from backtest.stats import (
    deflated_sharpe_ratio,
    min_trade_gate,
    per_trade_tstat,
    probabilistic_sharpe_ratio,
    win_rate_binomial_ci,
)


# --- Probabilistic Sharpe Ratio -------------------------------------------------


def test_psr_returns_probability_in_unit_interval():
    p = probabilistic_sharpe_ratio(0.1, n_obs=252, sr_star=0.0)
    assert 0.0 <= p <= 1.0


def test_psr_known_answer_n252_pins_observation_count_term():
    # Hand-computed Phi value:
    #   denom = sqrt(1 - skew*SR + ((kurt-1)/4)*SR^2)
    #   z     = (SR - sr_star) * sqrt(N - 1) / denom,  N = 252 OOS observations
    #   PSR   = Phi(z)
    # For SR=0.1, N=252, sr_star=0, skew=0, kurt=3  ->  z = 1.5803519980722478
    #                                                   PSR = 0.9429868610243624
    p = probabilistic_sharpe_ratio(0.1, n_obs=252, sr_star=0.0, skew=0.0, kurtosis=3.0)
    assert p == pytest.approx(0.9429868610243624, abs=1e-9)


def test_psr_increases_with_n_obs():
    # More observations -> the same observed Sharpe is more credible.
    low_n = probabilistic_sharpe_ratio(0.1, n_obs=60, sr_star=0.0)
    high_n = probabilistic_sharpe_ratio(0.1, n_obs=500, sr_star=0.0)
    assert high_n > low_n


def test_psr_high_sharpe_many_obs_approaches_one():
    p = probabilistic_sharpe_ratio(0.3, n_obs=2000, sr_star=0.0)
    assert p > 0.999


# --- Deflated Sharpe Ratio ------------------------------------------------------


def test_dsr_returns_probability_in_unit_interval():
    d = deflated_sharpe_ratio(0.1, n_trials=20, n_obs=252)
    assert 0.0 <= d <= 1.0


def test_dsr_non_increasing_as_trials_grow():
    # Selection penalty: testing more configurations and keeping the best
    # inflates the observed Sharpe, so the deflated probability must not rise.
    d1 = deflated_sharpe_ratio(0.1, n_trials=1, n_obs=252)
    d10 = deflated_sharpe_ratio(0.1, n_trials=10, n_obs=252)
    d50 = deflated_sharpe_ratio(0.1, n_trials=50, n_obs=252)
    d200 = deflated_sharpe_ratio(0.1, n_trials=200, n_obs=252)
    assert d1 >= d10 >= d50 >= d200


def test_dsr_strictly_penalises_many_trials_vs_one():
    d1 = deflated_sharpe_ratio(0.1, n_trials=1, n_obs=252)
    d50 = deflated_sharpe_ratio(0.1, n_trials=50, n_obs=252)
    assert d50 < d1


def test_dsr_increases_with_n_obs_pins_sample_size_term():
    # Fixed observed Sharpe and trial count: more OOS observations -> more
    # confidence. This pins n_obs as the sample-size term, not the trade count.
    low_n = deflated_sharpe_ratio(0.1, n_trials=10, n_obs=100)
    high_n = deflated_sharpe_ratio(0.1, n_trials=10, n_obs=500)
    assert high_n > low_n


def test_dsr_n_trials_one_reduces_to_psr():
    # With a single trial there is no selection bias, so DSR == PSR(sr_star=0).
    d = deflated_sharpe_ratio(0.2, n_trials=1, n_obs=1000)
    p = probabilistic_sharpe_ratio(0.2, n_obs=1000, sr_star=0.0)
    assert d == pytest.approx(p, abs=1e-12)
    assert d > 0.99


def test_dsr_docstring_defines_n_obs_as_observation_count():
    doc = deflated_sharpe_ratio.__doc__ or ""
    lowered = doc.lower()
    assert "observation" in lowered
    assert "not the trade count" in lowered or "not the number of trades" in lowered


# --- Per-trade t-stat -----------------------------------------------------------


def test_per_trade_tstat_matches_scipy():
    returns = np.array([0.02, -0.01, 0.03, 0.015, -0.005, 0.025, 0.01, -0.02])
    expected = float(ttest_1samp(returns, 0.0).statistic)
    assert per_trade_tstat(returns) == pytest.approx(expected, abs=1e-12)


def test_per_trade_tstat_flags_clearly_positive_sample():
    # Small, consistently positive returns -> a large positive t-stat (|t| > 2).
    returns = np.array([0.012, 0.010, 0.011, 0.013, 0.009, 0.012, 0.010, 0.011])
    assert abs(per_trade_tstat(returns)) > 2.0


def test_per_trade_tstat_does_not_flag_coin_flip_sample():
    # Symmetric +/- returns with near-zero mean -> |t| well under 2.
    returns = np.array([0.05, -0.05, 0.04, -0.04, 0.045, -0.045, 0.05, -0.05])
    assert abs(per_trade_tstat(returns)) < 2.0


# --- Minimum-trade gate ---------------------------------------------------------


def test_min_trade_gate_false_below_thirty():
    assert min_trade_gate(29) is False
    assert min_trade_gate(0) is False


def test_min_trade_gate_true_at_and_above_thirty():
    assert min_trade_gate(30) is True
    assert min_trade_gate(100) is True


def test_min_trade_gate_respects_custom_minimum():
    assert min_trade_gate(49, minimum=50) is False
    assert min_trade_gate(50, minimum=50) is True


# --- Win-rate binomial CI -------------------------------------------------------


def test_win_rate_ci_contains_point_estimate():
    low, high = win_rate_binomial_ci(4, 6)
    assert low <= 4 / 6 <= high


def test_win_rate_ci_4_of_6_known_answer():
    # 4/6 wins -> Clopper-Pearson 95% CI ~ [0.22, 0.96] (STACK.md example):
    # indistinguishable from a coin flip.
    low, high = win_rate_binomial_ci(4, 6)
    assert low == pytest.approx(0.22277809550351219, abs=1e-9)
    assert high == pytest.approx(0.9567281317072583, abs=1e-9)
    assert 0.20 < low < 0.25
    assert 0.94 < high < 0.97

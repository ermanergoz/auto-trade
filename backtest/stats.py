"""Vendored edge-validation statistics (pure functions, no engine coupling).

Closed-form Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR)
after Bailey & López de Prado, "The Deflated Sharpe Ratio" (SSRN 2460551).
The Euler–Mascheroni term in the expected-maximum-Sharpe approximation and the
sqrt(n_obs - 1) sample-size term were checked against the paper.

n_obs DEFINITION (load-bearing — consumed identically by 02-03 and 02-07):
    n_obs is the number of OUT-OF-SAMPLE OBSERVATIONS, i.e. the count of OOS
    trading days in the daily-return / equity-curve series the Sharpe was
    computed from. It is the SAMPLE SIZE that enters the sqrt(n_obs - 1) term.
    n_obs is NOT the number of trades. The >=30-TRADE statistical floor is a
    SEPARATE concept, handled by ``min_trade_gate``; do not conflate the two.
    Callers must pass len(OOS daily-return series), never len(trades).

This module is intentionally standalone so it does not contend with the
report.py / engine.py hubs and can be imported by both the walk-forward report
(02-03, display) and the extension sweep (02-07, gating). It adds no heavy
dependency beyond scipy.stats, which the methodology stack already prescribes.
"""

import math

import numpy as np
from scipy.stats import binomtest, norm, ttest_1samp

__all__ = [
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "per_trade_tstat",
    "min_trade_gate",
    "win_rate_binomial_ci",
]

# Euler–Mascheroni constant, used in the Bailey/López de Prado expected-maximum
# Sharpe approximation (SSRN 2460551, eq. for E[max]).
_EULER_MASCHERONI = 0.5772156649015329


def _psr_denominator(observed_sr: float, skew: float, kurtosis: float) -> float:
    """sqrt(1 - skew*SR + ((kurt - 1)/4) * SR^2) — the higher-moment adjustment."""
    return math.sqrt(
        1.0 - skew * observed_sr + ((kurtosis - 1.0) / 4.0) * observed_sr**2
    )


def probabilistic_sharpe_ratio(
    observed_sr: float,
    n_obs: int,
    sr_star: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """P(true SR > sr_star) given an observed Sharpe over ``n_obs`` observations.

    PSR(SR*) = Phi( ((SR - SR*) * sqrt(n_obs - 1))
                    / sqrt(1 - skew*SR + ((kurt - 1)/4) * SR^2) )

    ``n_obs`` is the OOS observation count (trading days in the return/equity
    series), NOT the trade count — it is the sample size in the sqrt(n_obs - 1)
    term. ``observed_sr``/``sr_star`` must be expressed on the same horizon as
    ``n_obs`` (e.g. per-day Sharpe with n_obs = number of days).
    """
    if n_obs < 2:
        raise ValueError("n_obs must be >= 2 (need at least two OOS observations)")
    denom = _psr_denominator(observed_sr, skew, kurtosis)
    z = (observed_sr - sr_star) * math.sqrt(n_obs - 1) / denom
    return float(norm.cdf(z))


def _expected_max_sharpe(n_trials: int, sr_trials_var: float) -> float:
    """Bailey/López de Prado expected maximum Sharpe across ``n_trials`` configs.

    E[max] ≈ sqrt(Var) * ( (1 - γ) * Φ⁻¹(1 - 1/N)
                           + γ      * Φ⁻¹(1 - 1/(N·e)) )

    Returns 0.0 for a single trial (no selection bias, no deflation).
    """
    if n_trials < 2:
        return 0.0
    sigma = math.sqrt(sr_trials_var)
    g = _EULER_MASCHERONI
    inv1 = norm.ppf(1.0 - 1.0 / n_trials)
    inv2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(sigma * ((1.0 - g) * inv1 + g * inv2))


def deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sr_trials_var: float | None = None,
) -> float:
    """Deflated Sharpe Ratio: P(true SR > 0) corrected for ``n_trials`` trials.

    When you try ``n_trials`` configurations and keep the best, the winner's
    Sharpe is inflated by selection bias. DSR deflates the observed Sharpe by an
    expected-maximum threshold ``sr_star = E[max Sharpe across n_trials]`` and
    returns PSR against that threshold (Bailey & López de Prado, SSRN 2460551).

    For fixed ``observed_sr`` / ``n_obs`` the result is monotonically
    non-increasing as ``n_trials`` grows. For fixed ``observed_sr`` /
    ``n_trials`` it increases with ``n_obs``.

    Parameters
    ----------
    observed_sr : float
        The best (selected) Sharpe ratio, on the same horizon as ``n_obs``.
    n_trials : int
        Number of configurations tried (the multiple-testing count, K).
    n_obs : int
        Number of OOS OBSERVATIONS (trading days) in the return series — the
        sample size in the sqrt(n_obs - 1) term. This is NOT the trade count;
        the >=30-trade floor is handled separately by ``min_trade_gate``.
    skew, kurtosis : float
        Higher moments of the return distribution (defaults: normal).
    sr_trials_var : float, optional
        Variance of the Sharpe ratios across the ``n_trials`` trials. When None,
        it defaults to the null-hypothesis variance of the Sharpe estimator over
        ``n_obs`` observations, i.e. denom^2 / (n_obs - 1).
    """
    if sr_trials_var is None:
        denom = _psr_denominator(observed_sr, skew, kurtosis)
        sr_trials_var = denom**2 / (n_obs - 1)
    sr_star = _expected_max_sharpe(n_trials, sr_trials_var)
    return probabilistic_sharpe_ratio(
        observed_sr, n_obs, sr_star=sr_star, skew=skew, kurtosis=kurtosis
    )

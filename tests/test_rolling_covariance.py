"""
test_rolling_covariance.py

A rolling (fixed-window) covariance. covariance_calculators dropped
RollingCovariance; SMACovariance computes the same window.
"""

import numpy as np

from covariance_calculators.estimators import SMACovariance


def test_rolling_mean_and_covariance_match_the_last_span_observations():
    span = 3
    observations = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
                    [0, 1, 1], [0, 1, 1], [0, 1, 1], [1, 1, 1], [0, 0, 0]]
    calc = SMACovariance(3, span=span)
    for n, obs in enumerate(observations, start=1):
        calc.add(obs)
        window = np.array(observations[max(0, n - span):n], dtype=float)
        np.testing.assert_allclose(calc.mean, window.mean(axis=0))
        if len(window) >= 2:
            np.testing.assert_allclose(calc.cov, np.cov(window, rowvar=False))

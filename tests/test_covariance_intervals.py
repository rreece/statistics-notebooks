"""
Tests for covariance confidence intervals.

These are simple end-to-end smoke tests. Comprehensive tests are in
the covariance_calculators package tests.
"""

import numpy as np

from covariance_calculators.intervals import (
    calc_covariance_intervals,
    calc_precision_intervals,
)


def test_covariance_intervals_bracket_estimate():
    """Confidence intervals should bracket the covariance estimate."""
    np.random.seed(42)

    data = np.random.randn(100, 2)
    cov, cov_lower, cov_upper = calc_covariance_intervals(
        data=data, confidence_level=0.95, method="normal"
    )

    assert np.all(cov_lower <= cov)
    assert np.all(cov_upper >= cov)


def test_precision_intervals_bracket_estimate():
    """Precision intervals should bracket the precision estimate."""
    np.random.seed(42)

    data = np.random.randn(100, 3)
    prec, prec_lower, prec_upper = calc_precision_intervals(
        data=data, confidence_level=0.95, method="invwishart"
    )

    assert np.all(prec_lower <= prec)
    assert np.all(prec_upper >= prec)

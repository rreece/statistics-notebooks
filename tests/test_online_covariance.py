"""
Tests for covariance estimators.

These are simple end-to-end smoke tests. Comprehensive tests are in
the covariance_calculators package tests.
"""

import numpy as np

from covariance_calculators.estimators import (
    OnlineCovariance,
    EMACovariance,
    SMACovariance,
)


def test_online_covariance_matches_numpy():
    """OnlineCovariance should match numpy's mean and covariance."""
    np.random.seed(42)

    oc = OnlineCovariance(order=3, frequency=1)
    data = np.random.randn(100, 3)

    for row in data:
        oc.add(row)

    assert np.allclose(oc.mean, np.mean(data, axis=0), rtol=1e-6)
    assert np.allclose(oc.cov, np.cov(data, rowvar=False), rtol=1e-6)


def test_sma_covariance_matches_numpy():
    """SMACovariance should match numpy on the last N samples."""
    np.random.seed(42)

    window = 20
    sma = SMACovariance(order=2, span=window, frequency=1)
    data = np.random.randn(100, 2)

    for row in data:
        sma.add(row)

    assert np.allclose(sma.mean, np.mean(data[-window:], axis=0), rtol=1e-6)
    assert np.allclose(sma.cov, np.cov(data[-window:], rowvar=False), rtol=1e-6)


def test_all_estimators_produce_valid_covariance():
    """All estimators should produce symmetric, positive semidefinite matrices."""
    np.random.seed(42)

    n_vars = 3
    estimators = [
        OnlineCovariance(order=n_vars, frequency=1),
        EMACovariance(order=n_vars, span=50, frequency=1),
        SMACovariance(order=n_vars, span=50, frequency=1),
    ]

    data = np.random.randn(100, n_vars)

    for row in data:
        for est in estimators:
            est.add(row)

    for est in estimators:
        cov = est.cov
        # Symmetric
        assert np.allclose(cov, cov.T, rtol=1e-6)
        # Positive semidefinite
        assert np.all(np.linalg.eigvalsh(cov) >= -1e-10)

"""
test_online_covariance.py
"""

import math
import numpy as np

from covariance_calculators.estimators import calc_sample_mean, calc_sample_covariance, OnlineCovariance


np.random.seed(42)
np.set_printoptions(precision=4, suppress=True)


def test_online_covariance():

    # Generate sample data
    n_samples = 1000
    n_features = 3
    true_mean = np.zeros(n_features)
    true_cov = np.array([[0.010,  0.005,  0.003],
                         [0.005,  0.020, -0.002],
                         [0.003, -0.002,  0.030]])
    data = np.random.multivariate_normal(mean=true_mean,
                                   cov=true_cov,
                                   size=n_samples)
    mean = calc_sample_mean(data)

    covariance = calc_sample_covariance(data)

    assert np.allclose(mean, true_mean, rtol=0, atol=1e-2)
    assert np.allclose(covariance, true_cov, rtol=0, atol=1e-3)

    online_calculator = OnlineCovariance(n_features)

    for row in data:
        online_calculator.add(row)

    assert np.allclose(online_calculator.mean, mean, rtol=0, atol=1e-4)
    assert np.allclose(online_calculator.cov, covariance, rtol=0, atol=1e-4)


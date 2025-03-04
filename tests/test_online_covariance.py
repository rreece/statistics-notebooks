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
    true_cov = np.array([[0.010,  0.005,  0.003],
                         [0.005,  0.020, -0.002],
                         [0.003, -0.002,  0.030]])
    data = np.random.multivariate_normal(mean=np.zeros(n_features),
                                   cov=true_cov,
                                   size=n_samples)
    mean = calc_sample_mean(data)

    covariance = calc_sample_covariance(data)

    ref_mean = np.array([0.0051, -0.0020, 0.0089])
    assert np.allclose(mean, ref_mean, rtol=0, atol=1e-4)

    ref_covariance = np.array([[0.0095,  0.0051,  0.0031],
                               [0.0051,  0.0202, -0.0005],
                               [0.0031, -0.0005,  0.0285]])
    assert np.allclose(covariance, ref_covariance, rtol=0, atol=1e-4)

    online_calculator = OnlineCovariance(n_features)

    for row in data:
        online_calculator.add(row)

    assert np.allclose(online_calculator.mean, ref_mean, rtol=0, atol=1e-4)
    assert np.allclose(online_calculator.cov, ref_covariance, rtol=0, atol=1e-4)


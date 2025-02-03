"""
statistics-notebooks pytests
"""

import numpy as np

import covariance_intervals as ci


np.random.seed(42)
np.set_printoptions(precision=4, suppress=True)


def test_normal_covariance_interval():

    # Generate sample data
    n_samples = 1000
    n_features = 3
    true_cov = np.array([[0.010,  0.005,  0.003],
                         [0.005,  0.020, -0.002],
                         [0.003, -0.002,  0.030]])
    data = np.random.multivariate_normal(mean=np.zeros(n_features),
                                   cov=true_cov,
                                   size=n_samples)

    # Calculate confidence interval
    confidence_level = 0.95
    method = "normal"
    covariance, covariance_lower, covariance_upper = ci.calc_covariance_intervals(data, confidence_level, method)

    ref_covariance = np.array([[0.0095,  0.0051,  0.0031],
                               [0.0051,  0.0202, -0.0005],
                               [0.0031, -0.0005,  0.0285]])
    assert np.allclose(covariance, ref_covariance, rtol=0, atol=1e-4)


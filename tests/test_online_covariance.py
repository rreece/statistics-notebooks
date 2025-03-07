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
    n_samples = 10000
    n_features = 3
    true_mean = np.array([1, 2, 3])
    true_cov = np.array([[0.010,  0.005,  0.003],
                         [0.005,  0.020, -0.004],
                         [0.003, -0.004,  0.040]])
    data = np.random.multivariate_normal(mean=true_mean,
                                   cov=true_cov,
                                   size=n_samples)

    # Calculate sample mean and covariance
    mean = calc_sample_mean(data)
    covariance = calc_sample_covariance(data)

    assert np.allclose(mean, true_mean, rtol=0, atol=1e-2)
    assert np.allclose(covariance, true_cov, rtol=0, atol=1e-3)

    # Calculate online mean and covariance
    online_calculator = OnlineCovariance(n_features)

    for row in data:
        online_calculator.add(row)

    assert np.allclose(online_calculator.mean, mean, rtol=0, atol=1e-4)
    assert np.allclose(online_calculator.cov, covariance, rtol=0, atol=1e-4)


def test_online_covariance_merge():
    # create two differently correllated datasets
    # (again, three dimensions)
#    data_part1 = create_correlated_dataset( \
#       500, (2.2, 4.4, 1.5), np.array([[0.2, 0.5, 0.7],[0.3, 0.2, 0.2],[0.5,0.3,0.1]]), (1, 5, 3))
#   data_part2 = create_correlated_dataset( \
#       1000, (5, 6, 2), np.array([[0.2, 0.5, 0.7],[0.3, 0.2, 0.2],[0.5,0.3,0.1]]), (1, 5, 3))

    # Generate sample data
    n_samples = 10000
    n_features = 3
    true_mean = np.array([1, 2, 3])
    true_cov = np.array([[0.2, 2.5, 2.1],
                         [0.3, 1. , 0.6],
                         [0.5, 1.5, 0.3]])
    data_part1 = np.random.multivariate_normal(
                        mean=np.array([2.2, 4.4, 1.5]),  # true_mean
                        cov=true_cov,
                        size=500)  # n_samples
    data_part2 = np.random.multivariate_normal(
                        mean=np.array([5, 6, 2]),  # true_mean
                        cov=true_cov,
                        size=1000)  # n_samples

    ocov_part1 = OnlineCovariance(3)
    ocov_part2 = OnlineCovariance(3)
    ocov_both = OnlineCovariance(3)
    
    # "grow" online-covariances for part 1 and 2 separately but also
    # put all observations into the OnlineCovariance object for both.
    
    for row in data_part1:
        ocov_part1.add(row)
        ocov_both.add(row)
        
    for row in data_part2:
        ocov_part2.add(row)
        ocov_both.add(row)
        
    ocov_merged = ocov_part1.merge(ocov_part2)
    
    assert ocov_both.count == ocov_merged.count, \
    """
    Count of ocov_both and ocov_merged should be the same.
    """
    assert np.isclose(ocov_both.mean, ocov_merged.mean).all(), \
    """
    Mean of ocov_both and ocov_merged should be the same.
    """
    assert np.isclose(ocov_both.cov, ocov_merged.cov).all(), \
    """
    Covarance-matrix of ocov_both and ocov_merged should be the same.
    """
    assert np.isclose(ocov_both.corr, ocov_merged.corr).all(), \
    """
    Pearson-Correlationcoefficient-matrix of ocov_both and ocov_merged should be the same.
    """


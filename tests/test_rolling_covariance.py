"""
test_rolling_covariance.py
"""

import numpy as np

from covariance_calculators.estimators import RollingCovariance


np.random.seed(42)
np.set_printoptions(precision=4, suppress=True)


def test_rolling_covariance():

    # Generate sample data
#    n_samples = 10000
#    n_samples = 5
    n_features = 3
#    true_mean = np.array([1, 2, 3])
#    true_cov = np.array([[0.010,  0.005,  0.003],
#                         [0.005,  0.020, -0.004],
#                         [0.003, -0.004,  0.040]])
#    data = np.random.multivariate_normal(mean=true_mean,
#                                   cov=true_cov,
#                                   size=n_samples)
#
#    # Calculate sample mean and covariance
#    mean = calc_sample_mean(data)
#    covariance = calc_sample_covariance(data)
#
#    # Check that sample statistics are close to the truth
#    assert np.allclose(mean, true_mean, rtol=0, atol=1e-2)
#    assert np.allclose(covariance, true_cov, rtol=0, atol=1e-3)

    # Calculate rolling mean and covariance
    roll_calc = RollingCovariance(n_features, span=3)
    print("")
    print("%02i  %s" % (0, roll_calc.mean))
    roll_calc.add([1, 1, 1])
    print("%02i  %s" % (1, roll_calc.mean))
    roll_calc.add([1, 1, 1])
    print("%02i  %s" % (2, roll_calc.mean))
    roll_calc.add([1, 1, 1])
    print("%02i  %s" % (3, roll_calc.mean))
    roll_calc.add([1, 1, 1])
    print("%02i  %s" % (4, roll_calc.mean))
    roll_calc.add([0, 1, 1])
    print("%02i  %s" % (5, roll_calc.mean))
    roll_calc.add([0, 1, 1])
    print("%02i  %s" % (6, roll_calc.mean))
    roll_calc.add([0, 1, 1])
    print("%02i  %s" % (7, roll_calc.mean))
    roll_calc.add([1, 1, 1])
    print("%02i  %s" % (8, roll_calc.mean))
    roll_calc.add([0, 0, 0])
    print("%02i  %s" % (8, roll_calc.mean))
    print("")

#    for row in data:
#        roll_calc.add(row)
#
#    # Check that rolling statistics are virtually identical to the sample statistics
#    assert np.allclose(roll_calc.mean, mean, rtol=0, atol=1e-5)
#    assert np.allclose(roll_calc.cov, covariance, rtol=0, atol=1e-5)


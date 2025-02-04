"""
covariance_calculators/estimators.py

TODO:

[ ] Sample mean and covariance
[ ] Rolling mean and covariance
[ ] Exponential moving mean and covariance
[ ] Online mean and covariance
[ ] Shrinkage estimators
"""


import numpy as np


def calc_sample_covariance(data, frequency=None):
    assert False, "Not implemented"


def calc_sample_covariance(data, frequency=None):
    covariance = np.cov(data, rowvar=False, bias=False)  # bias=False uses n-1

    # TODO: Use frequency

    return covariance


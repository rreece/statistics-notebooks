"""
covariance_calculators/test_intervals.py
"""

import math
import numpy as np

from covariance_calculators.estimators import calc_sample_covariance
from covariance_calculators.intervals import calc_covariance_intervals, calc_precision_intervals


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
    covariance, covariance_lower, covariance_upper = calc_covariance_intervals(data, confidence_level=confidence_level, method=method)

    ref_covariance = np.array([[0.0095,  0.0051,  0.0031],
                               [0.0051,  0.0202, -0.0005],
                               [0.0031, -0.0005,  0.0285]])
    assert np.allclose(covariance, ref_covariance, rtol=0, atol=1e-4)


def test_compare_methods():

    # Generate sample data
    n_samples = 1000
    n_features = 3
    true_cov = np.array([[0.010,  0.005,  0.003],
                         [0.005,  0.020, -0.002],
                         [0.003, -0.002,  0.030]])
    data = np.random.multivariate_normal(mean=np.zeros(n_features),
                                   cov=true_cov,
                                   size=n_samples)

    # Compare all methods
    confidence_level = 0.95
    compare_methods(data, confidence_level=confidence_level)


def compare_methods(
    data: np.ndarray,
    confidence_level: float = 0.95,
):
    """
    Compare normal approximation, Wishart, and bootstrap methods.
    """
    methods = ["normal", "wishart", "bootstrap"]

    # calculate the covariance once for all methods
    covariance1 = calc_sample_covariance(data)

    results = dict()
    for method in methods:
        results[method] = calc_covariance_intervals(data, covariance=covariance1, confidence_level=confidence_level, method=method)

    # Print results
    for method in methods:
        print(f"\n{method} method estimate:")
        covariance, covariance_lower, covariance_upper = results[method]
        print(covariance)
        print("lower:")
        print(covariance_lower)
        print("upper:")
        print(covariance_upper)

    return results


def test_coverage():
    import matplotlib.pyplot as plt
    import hepplot as hep

    # F(z) = Phi(z) = (1/2) * (1 + erf(z/sqrt(2)))
    # Phi(z) = (1 - alpha/2)   For two-sided
    # alpha = 2*(1 - Phi(z))
    #       = 2*(1 - 0.5*(1+math.erf(z/math.sqrt(2))))
    def _z_to_alpha(z):
        return 2*(1 - 0.5*(1+math.erf(z/math.sqrt(2))))

    alphas = [ _z_to_alpha(_z) for _z in [1, 2, 3, 4] ]
    confidence_levels = [ 1.0 - _a for _a in alphas ]
    print(alphas)
    print(confidence_levels)

    n_toys = 1000
#    n_toys = 10000

    # normal method experiments 
    normal_coverages = list()
    for cl in confidence_levels:
        coverage = run_coverage_test(confidence_level=cl, n_toys=n_toys, method="normal")
        avg_coverage = np.average(coverage)
        normal_coverages.append(avg_coverage)

    normal_coverage_alphas = [ 1.0 - _c for _c in normal_coverages ]
    print(normal_coverage_alphas)

    # HACK to speedup the test because Wishart MC takes a bit
    n_toys = 100

    # wishart method experiments 
    wishart_coverages = list()
    for cl in confidence_levels:
        coverage = run_coverage_test(confidence_level=cl, n_toys=n_toys, method="wishart")
        avg_coverage = np.average(coverage)
        wishart_coverages.append(avg_coverage)

    wishart_coverage_alphas = [ 1.0 - _c for _c in wishart_coverages ]
    print(wishart_coverage_alphas)

    # make coverage plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha = 1 - p_\mathrm{CL}$")
    ax.set_ylabel(r"$\alpha_\mathrm{coverage} = 1 - p_\mathrm{coverage}$")
    ax.plot(alphas, alphas, color="darkgray", label="Perfect calibration")
    ax.plot(alphas, normal_coverage_alphas, marker='o', color="#1f77b4", label="Asymptotic interval")
    ax.plot(alphas, wishart_coverage_alphas, marker='o', color="red", label="Wishart interval")
    legend = ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("coverage.pdf")
    plt.savefig("coverage.png")


def run_coverage_test(confidence_level=0.95, n_toys=200, method="normal"):
    n_samples = 1000
    n_features = 3
    true_cov = np.array([[0.010,  0.005,  0.003],
                         [0.005,  0.020, -0.002],
                         [0.003, -0.002,  0.030]])

    n_accept = np.zeros((n_features, n_features))

    for _ in range(n_toys):
        # Generate sample data
        data = np.random.multivariate_normal(mean=np.zeros(n_features),
                                       cov=true_cov,
                                       size=n_samples)

        # TODO: Use the same pseudo data with all methods

        # calculate the covariance
        covariance1 = calc_sample_covariance(data)

        # Calculate covariance and confidence intervals
        covariance, covariance_lower, covariance_upper = calc_covariance_intervals(data, covariance=covariance1, confidence_level=confidence_level, method=method)

        # Check coverage
        accepts = np.where( (covariance_lower < true_cov) & (true_cov < covariance_upper), 1, 0)
        n_accept += accepts

    coverage = n_accept / n_toys
    return coverage


def test_invwishart_precision_interval():

    # Generate sample data
    n_samples = 1000
    n_features = 3
    true_cov = np.array([[0.010,  0.005,  0.003],
                         [0.005,  0.020, -0.002],
                         [0.003, -0.002,  0.030]])
    true_precision = np.linalg.inv(true_cov)
    data = np.random.multivariate_normal(mean=np.zeros(n_features),
                                   cov=true_cov,
                                   size=n_samples)

    # Calculate confidence interval
    confidence_level = 0.95
    method = "invwishart"
    precision, precision_lower, precision_upper = calc_precision_intervals(data, confidence_level=confidence_level, method=method)

    print("DEBUG: true_precision =")
    print(true_precision)
    print("DEBUG: precision =")
    print(precision)
    print("DEBUG: precision_lower =")
    print(precision_lower)
    print("DEBUG: precision_upper =")
    print(precision_upper)

    ref_precision =       np.array([[117.096, -31.865, -12.001],
                                    [-31.865,  58.596,   4.764],
                                    [-12.001,   4.764,  37.930]])

    ref_precision_lower = np.array([[107.257, -37.515, -16.239],
                                    [-37.515,  53.671,   1.846],
                                    [-16.239,   1.846,  34.753]])

    ref_precision_upper = np.array([[127.775, -26.505,  -7.807],
                                    [-26.505,  63.783,   7.711], 
                                    [ -7.807,   7.711,  41.496]])

    assert np.allclose(precision, ref_precision, rtol=0, atol=1e-3)
    assert np.allclose(precision_lower, ref_precision_lower, rtol=0, atol=1e-3)
    assert np.allclose(precision_upper, ref_precision_upper, rtol=0, atol=1e-3)



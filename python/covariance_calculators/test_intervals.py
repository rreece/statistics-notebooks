"""
covariance_calculators/test_intervals.py
"""

import math
import numpy as np

from covariance_calculators.estimators import calc_sample_covariance
from covariance_calculators.intervals import calc_covariance_intervals


def test_compare_methods():
    np.random.seed(42)
    np.set_printoptions(precision=4, suppress=True)

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
    np.random.seed(42)
    np.set_printoptions(precision=4, suppress=True)

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


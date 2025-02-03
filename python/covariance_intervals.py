"""
Based on initial generated versions in conversation with
Claude AI: Claude 3.5 Sonnet
Feb 1, 2025
"""

import math
import numpy as np
from scipy import stats
from typing import Tuple, Optional


def calc_covariance_intervals(
    data: np.ndarray,
    confidence_level: float = 0.95,
    method: str = 'normal',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate confidence intervals for sample covariance matrix using unbiased estimators.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data matrix (n_samples, n_features)
    confidence_level : float
        Confidence level (default: 0.95)
    method : str
        Method to use ('normal', 'wishart', or 'bootstrap')
        
    Returns:
    --------
    cov_matrix : np.ndarray
        Sample covariance matrix
    ci_lower : np.ndarray
        Lower bounds of confidence intervals
    ci_upper : np.ndarray
        Upper bounds of confidence intervals
    """
    n_samples, n_features = data.shape
    # Using n-1 for unbiased estimation
    cov_matrix = np.cov(data, rowvar=False, bias=False)  # bias=False uses n-1
    alpha = 1.0 - confidence_level

    if method == 'normal':
        # Calculate standard errors using asymptotic formula with n-1 correction
        z_score = stats.norm.ppf(1 - alpha/2)
        se_matrix = np.zeros((n_features, n_features))
        
        # Standard error calculated from Wishart variance
        # Var(V_ij) = (V_ii V_jj + V_ij^2) / (n-1)
        # se_ij = sqrt( Var(V_ij) )
        for i in range(n_features):
            for j in range(n_features):
                se_matrix[i,j] = np.sqrt(
                    (cov_matrix[i,i] * cov_matrix[j,j] + 
                     cov_matrix[i,j]**2) / (n_samples - 1)
                )
        
        ci_lower = cov_matrix - z_score * se_matrix
        ci_upper = cov_matrix + z_score * se_matrix

    elif method == 'wishart':
        # Using Wishart distribution
        # Initialize Wishart distribution with scale matrix S/(df)
        # Scale matrix is S/(df) because wishart.rvs returns W/df where W ~ W_p(df, scale)
        dof = n_samples - 1
        wishart_dist = stats.wishart(df=dof, scale=cov_matrix/dof)

        # Generate samples to estimate quantiles
        n_samples_wishart = 10000
        wishart_samples = wishart_dist.rvs(n_samples_wishart)

        # Calculate element-wise quantiles (ppf)
        ci_lower = np.zeros((n_features, n_features))
        ci_upper = np.zeros((n_features, n_features))

        for i in range(n_features):
            for j in range(n_features):
                samples_ij = wishart_samples[:, i, j]
                ci_lower[i, j] = np.percentile(samples_ij, 100 * alpha/2)
                ci_upper[i, j] = np.percentile(samples_ij, 100 * (1 - alpha/2))
        
    elif method == 'bootstrap':
        # Bootstrap approach (np.cov already uses n-1 by default)
        n_bootstrap = 1000
        bootstrap_covs = np.zeros((n_bootstrap, n_features, n_features))
        
        for i in range(n_bootstrap):
            bootstrap_indices = np.random.choice(
                n_samples, size=n_samples, replace=True
            )
            bootstrap_sample = data[bootstrap_indices]
            bootstrap_covs[i] = np.cov(bootstrap_sample, rowvar=False, bias=False)
        
        # Calculate percentile intervals
        ci_lower = np.percentile(bootstrap_covs, 100 * alpha/2, axis=0)
        ci_upper = np.percentile(bootstrap_covs, 100 * (1 - alpha/2), axis=0)
    
    else:
        raise ValueError("Method must be either 'normal', 'wishart', or 'bootstrap'")
        
    return cov_matrix, ci_lower, ci_upper


def compare_methods(
    data: np.ndarray,
    confidence_level: float = 0.95,
):
    """
    Compare normal approximation, Wishart, and bootstrap methods.
    """
    methods = ["normal", "wishart", "bootstrap"]

    results = dict()
    for method in methods:
        results[method] = calc_covariance_intervals(data, confidence_level, method)

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

        # Calculate covariance and confidence intervals
        covariance, covariance_lower, covariance_upper = calc_covariance_intervals(data, confidence_level, method)

        # Check coverage
        accepts = np.where( (covariance_lower < true_cov) & (true_cov < covariance_upper), 1, 0)
        n_accept += accepts

    coverage = n_accept / n_toys
    return coverage


def main():
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


def main_coverage_test():

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


if __name__ == "__main__":
    np.random.seed(42)
    np.set_printoptions(precision=4, suppress=True)

    main()
#    main_coverage_test()


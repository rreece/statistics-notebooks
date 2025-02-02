"""
Based on initial generated versions in conversation with
Claude AI: Claude 3.5 Sonnet
Feb 1, 2025
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional


np.random.seed(42)
np.set_printoptions(precision=3, suppress=True)


def covariance_confidence_intervals(
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
    half_alpha = alpha/2.0

    assert np.isclose((1 + confidence_level) / 2, 1 - half_alpha, rtol=0, atol=1e-5)
    
    if method == 'normal':
        # Calculate standard errors using asymptotic formula with n-1 correction
        z_score = stats.norm.ppf(1 - half_alpha)
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

        # Calculate element-wise quantiles
        ci_lower = np.zeros((n_features, n_features))
        ci_upper = np.zeros((n_features, n_features))

        for i in range(n_features):
            for j in range(n_features):
                samples_ij = wishart_samples[:, i, j]
                ci_lower[i, j] = np.percentile(samples_ij, 100 * half_alpha)
                ci_upper[i, j] = np.percentile(samples_ij, 100 * (1 - half_alpha))
        
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
        ci_lower = np.percentile(bootstrap_covs, 100 * half_alpha, axis=0)
        ci_upper = np.percentile(bootstrap_covs, 100 * (1 - half_alpha), axis=0)
    
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
        results[method] = covariance_confidence_intervals(data, confidence_level, method)

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


def main():
    # Generate sample data
    n_samples = 1000
    n_features = 3
    true_cov = np.array([[1.0, 0.5, 0.3],
                         [0.5, 1.0, -0.2],
                         [0.3, -0.2, 1.0]])
    data = np.random.multivariate_normal(mean=np.zeros(n_features),
                                   cov=true_cov,
                                   size=n_samples)

    # Compare all methods
    confidence_level = 0.95
    compare_methods(data, confidence_level=confidence_level)


def main_coverage_test():
    n_samples = 1000
    n_features = 3
    true_cov = np.array([[1.0, 0.5, 0.3],
                         [0.5, 1.0, -0.2],
                         [0.3, -0.2, 1.0]])

    n_experiments = 200
    n_accept = np.zeros((n_features, n_features))

    for _ in range(n_experiments):
        # Generate sample data
        data = np.random.multivariate_normal(mean=np.zeros(n_features),
                                       cov=true_cov,
                                       size=n_samples)

        # Calculate covariance and confidence intervals
        confidence_level = 0.95
        method = "normal"
        covariance, covariance_lower, covariance_upper = covariance_confidence_intervals(data, confidence_level, method)

        # Check coverage
        accepts = np.where( (covariance_lower < true_cov) & (true_cov < covariance_upper), 1, 0)
        n_accept += accepts

    coverage = n_accept / n_experiments
    print("coverage =")
    print(coverage)


if __name__ == "__main__":
    main()
#    main_coverage_test()


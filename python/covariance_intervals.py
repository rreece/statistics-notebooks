"""
Generated in conversation with Claude AI: Claude 3.5 Sonnet
Jan 2025
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional

def covariance_confidence_intervals(
    data: np.ndarray,
    confidence_level: float = 0.95,
    method: str = 'asymptotic',
    n_bootstrap: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate confidence intervals for sample covariance matrix.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data matrix (n_samples, n_features)
    confidence_level : float
        Confidence level (default: 0.95)
    method : str
        Method to use ('asymptotic' or 'bootstrap')
    n_bootstrap : int
        Number of bootstrap samples if using bootstrap method
        
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
    cov_matrix = np.cov(data, rowvar=False)
    
    if method == 'asymptotic':
        # Calculate standard errors using asymptotic formula
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        se_matrix = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(n_features):
                se_matrix[i,j] = np.sqrt(
                    (cov_matrix[i,i] * cov_matrix[j,j] + 
                     cov_matrix[i,j]**2) / n_samples
                )
        
        ci_lower = cov_matrix - z_score * se_matrix
        ci_upper = cov_matrix + z_score * se_matrix
        
    elif method == 'bootstrap':
        # Bootstrap approach
        bootstrap_covs = np.zeros((n_bootstrap, n_features, n_features))
        
        for i in range(n_bootstrap):
            bootstrap_indices = np.random.choice(
                n_samples, size=n_samples, replace=True
            )
            bootstrap_sample = data[bootstrap_indices]
            bootstrap_covs[i] = np.cov(bootstrap_sample, rowvar=False)
        
        # Calculate percentile intervals
        alpha = (1 - confidence_level) / 2
        ci_lower = np.percentile(bootstrap_covs, 100 * alpha, axis=0)
        ci_upper = np.percentile(bootstrap_covs, 100 * (1 - alpha), axis=0)
    
    else:
        raise ValueError("Method must be either 'asymptotic' or 'bootstrap'")
        
    return cov_matrix, ci_lower, ci_upper

def format_results(
    cov_matrix: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    feature_names: Optional[list] = None
) -> None:
    """
    Print formatted results of covariance matrix with confidence intervals.
    """
    n_features = cov_matrix.shape[0]
    if feature_names is None:
        feature_names = [f"X{i+1}" for i in range(n_features)]
        
    print("Covariance Matrix with Confidence Intervals:")
    print("-" * 50)
    
    for i in range(n_features):
        for j in range(i + 1):
            print(f"{feature_names[i]} - {feature_names[j]}:")
            print(f"Covariance: {cov_matrix[i,j]:.4f}")
            print(f"95% CI: [{ci_lower[i,j]:.4f}, {ci_upper[i,j]:.4f}]")
            print()

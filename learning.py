"""
learning.py
Clean implementation of Bayesian and adaptive learning mechanisms for belief updating
"""

import numpy as np
from typing import Dict, List
from scipy.spatial.distance import mahalanobis


def create_path_vector(gdp_path: List[float], inf_path: List[float]) -> np.ndarray:
    """Create combined path vector [gdp1, gdp2, ..., inf1, inf2, ...]"""
    return np.concatenate([gdp_path, inf_path])


def safe_matrix_inverse(matrix: np.ndarray) -> np.ndarray:
    """Safely compute matrix inverse with regularization if needed"""
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        # Add small regularization to diagonal
        regularized = matrix + 1e-8 * np.eye(matrix.shape[0])
        return np.linalg.inv(regularized)


def calculate_raw_likelihoods(observed_gdp: List[float],
                             observed_inf: List[float],
                             scenarios: Dict[str, Dict],
                             covariance_matrix: np.ndarray) -> Dict[str, float]:
    """
    Calculate raw likelihoods P(data|scenario) using Mahalanobis distance

    Args:
        observed_gdp: Observed GDP growth so far
        observed_inf: Observed inflation so far
        scenarios: Scenario definitions with gdp_growth and inflation paths
        covariance_matrix: Historical covariance matrix

    Returns:
        {scenario_name: raw_likelihood}
    """

    n_periods = len(observed_gdp)
    observed_vector = create_path_vector(observed_gdp, observed_inf)

    # Extract covariance subset for observed periods
    # Structure: [GDP1, GDP2, GDP3, INF1, INF2, INF3]
    gdp_indices = list(range(n_periods))  # [0, 1, ...]
    inf_indices = list(range(3, 3 + n_periods))  # [3, 4, ...]
    all_indices = gdp_indices + inf_indices

    cov_subset = covariance_matrix[np.ix_(all_indices, all_indices)]
    cov_inv = safe_matrix_inverse(cov_subset)

    raw_likelihoods = {}

    for scenario_name, scenario_def in scenarios.items():
        # Get predicted path for same number of periods
        predicted_gdp = scenario_def['gdp_growth'][:n_periods]
        predicted_inf = scenario_def['inflation'][:n_periods]
        predicted_vector = create_path_vector(predicted_gdp, predicted_inf)

        # Mahalanobis distance squared
        d_squared = mahalanobis(observed_vector, predicted_vector, cov_inv) ** 2

        # Raw likelihood: exp(-d²/2)
        raw_likelihood = np.exp(-d_squared / 2.0)
        raw_likelihoods[scenario_name] = raw_likelihood

    return raw_likelihoods


def bayesian_update(prior_beliefs: Dict[str, float],
                   observed_gdp: List[float],
                   observed_inf: List[float],
                   scenarios: Dict[str, Dict],
                   covariance_matrix: np.ndarray) -> Dict[str, float]:
    """
    Bayesian belief updating using Mahalanobis distance likelihoods

    Args:
        prior_beliefs: Current belief probabilities
        observed_gdp: Observed GDP growth path
        observed_inf: Observed inflation path
        scenarios: Scenario definitions
        covariance_matrix: Historical covariance matrix

    Returns:
        Updated belief probabilities
    """

    # Get raw likelihoods
    raw_likelihoods = calculate_raw_likelihoods(
        observed_gdp, observed_inf, scenarios, covariance_matrix
    )

    # Evidence: P(data) = Σ P(data|scenario) × P(scenario)
    evidence = sum(raw_likelihoods[s] * prior_beliefs[s] for s in prior_beliefs.keys())

    if evidence <= 1e-15:
        # Return uniform distribution if evidence is too small
        n_scenarios = len(prior_beliefs)
        return {s: 1.0/n_scenarios for s in prior_beliefs.keys()}

    # Posterior: P(scenario|data) = P(data|scenario) × P(scenario) / P(data)
    posterior = {}
    for scenario in prior_beliefs.keys():
        numerator = raw_likelihoods[scenario] * prior_beliefs[scenario]
        posterior[scenario] = numerator / evidence

    # Ensure probabilities sum to 1
    total = sum(posterior.values())
    if total > 0:
        posterior = {s: p/total for s, p in posterior.items()}

    return posterior


def adaptive_update(prior_beliefs: Dict[str, float],
                   observed_gdp: List[float],
                   observed_inf: List[float],
                   scenarios: Dict[str, Dict],
                   covariance_matrix: np.ndarray,
                   learning_rate: float = 0.3) -> Dict[str, float]:
    """
    Adaptive belief updating with configurable learning rate

    Formula: P_new = P_old + λ × (P_bayesian - P_old)

    Args:
        prior_beliefs: Current belief probabilities
        observed_gdp: Observed GDP growth path
        observed_inf: Observed inflation path
        scenarios: Scenario definitions
        covariance_matrix: Historical covariance matrix
        learning_rate: Lambda parameter (0 < λ ≤ 1)

    Returns:
        Updated belief probabilities
    """

    # First get Bayesian posterior
    bayesian_posterior = bayesian_update(
        prior_beliefs, observed_gdp, observed_inf, scenarios, covariance_matrix
    )

    # Apply adaptive formula
    updated = {}
    for scenario in prior_beliefs.keys():
        old_prob = prior_beliefs[scenario]
        bayesian_prob = bayesian_posterior[scenario]

        # Adaptive update: old + λ(new - old)
        new_prob = old_prob + learning_rate * (bayesian_prob - old_prob)
        new_prob = max(0.0, min(1.0, new_prob))  # Ensure [0,1] bounds
        updated[scenario] = new_prob

    # Renormalize to ensure sum = 1
    total = sum(updated.values())
    if total > 0:
        updated = {s: p/total for s, p in updated.items()}
    else:
        # Fallback to uniform
        n_scenarios = len(updated)
        updated = {s: 1.0/n_scenarios for s in updated.keys()}

    return updated


def no_learning_update(prior_beliefs: Dict[str, float],
                      observed_gdp: List[float],
                      observed_inf: List[float],
                      scenarios: Dict[str, Dict],
                      covariance_matrix: np.ndarray) -> Dict[str, float]:
    """
    No learning: return unchanged beliefs

    Args:
        prior_beliefs: Current belief probabilities
        observed_gdp: Ignored
        observed_inf: Ignored
        scenarios: Ignored
        covariance_matrix: Ignored

    Returns:
        Unchanged belief probabilities
    """
    return prior_beliefs.copy()


def create_learning_function(learning_type: str, learning_rate: float = 0.3):
    """
    Factory function to create learning functions with preset parameters

    Args:
        learning_type: 'no_learning', 'bayesian', or 'adaptive'
        learning_rate: Lambda parameter for adaptive learning

    Returns:
        Learning function with signature: f(beliefs, gdp, inf, scenarios, cov_matrix)
    """

    if learning_type == 'no_learning':
        return no_learning_update
    elif learning_type == 'bayesian':
        return bayesian_update
    elif learning_type == 'adaptive':
        def adaptive_with_rate(beliefs, gdp, inf, scenarios, cov_matrix):
            return adaptive_update(beliefs, gdp, inf, scenarios, cov_matrix, learning_rate)
        return adaptive_with_rate
    else:
        raise ValueError(f"Unknown learning type: {learning_type}")

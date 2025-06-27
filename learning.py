"""
learning.py
Simple Bayesian belief updating
"""

import numpy as np
from typing import Dict, List


def update_beliefs(current_beliefs: Dict[str, float],
                  observed_gdp: List[float],
                  observed_inflation: List[float],
                  scenarios: Dict[str, Dict],
                  covariance_matrix: np.ndarray) -> Dict[str, float]:
    """
    Update scenario probabilities using Bayesian learning

    Args:
        current_beliefs: {scenario: probability}
        observed_gdp: [gdp1, gdp2, ...] observed so far
        observed_inflation: [inf1, inf2, ...] observed so far
        scenarios: {scenario: {'gdp_growth': [...], 'inflation': [...]}}
        covariance_matrix: Historical covariance matrix

    Returns:
        {scenario: updated_probability}
    """
    n_periods = len(observed_gdp)

    # Combine observed data
    observed = np.concatenate([observed_gdp, observed_inflation])

    # Get covariance subset for observed periods
    gdp_indices = list(range(n_periods))  # [0, 1, ...]
    inf_indices = list(range(n_periods, n_periods*2))  # [3, 4, ...]
    all_indices = gdp_indices + inf_indices

    cov_subset = covariance_matrix[np.ix_(all_indices, all_indices)]

    # Safe matrix inversion
    try:
        cov_inv = np.linalg.inv(cov_subset)
    except:
        cov_inv = np.linalg.pinv(cov_subset)

    # Calculate likelihoods for each scenario
    likelihoods = {}

    for scenario_name, scenario_def in scenarios.items():
        # Get predicted path for observed periods
        predicted_gdp = scenario_def['gdp_growth'][:n_periods]
        predicted_inf = scenario_def['inflation'][:n_periods]
        predicted = np.concatenate([predicted_gdp, predicted_inf])

        # Mahalanobis distance
        diff = observed - predicted
        distance_squared = diff.T @ cov_inv @ diff
        distance_squared = max(0, distance_squared)  # Ensure non-negative

        # Likelihood
        likelihoods[scenario_name] = np.exp(-distance_squared / 2.0)

    # Bayesian update: P(scenario|data) ∝ P(data|scenario) × P(scenario)
    posteriors = {}
    total = 0

    for scenario in current_beliefs.keys():
        posterior = likelihoods[scenario] * current_beliefs[scenario]
        posteriors[scenario] = posterior
        total += posterior

    # Normalize to probabilities
    if total > 0:
        for scenario in posteriors.keys():
            posteriors[scenario] /= total
    else:
        # Uniform fallback
        n_scenarios = len(current_beliefs)
        posteriors = {s: 1.0/n_scenarios for s in current_beliefs.keys()}

    return posteriors

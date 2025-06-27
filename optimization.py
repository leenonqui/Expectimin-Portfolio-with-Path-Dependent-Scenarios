"""
optimization.py
Clean implementation of expectimin portfolio optimization and sequential optimization with learning
"""

import pulp as lp
from typing import Dict, List, Callable
import numpy as np


def expectimin_optimize(probabilities: Dict[str, float],
                       scenario_returns: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Expectimin portfolio optimization: minimize expected losses

    Args:
        probabilities: {scenario_name: probability}
        scenario_returns: {scenario_name: {asset_name: return_%}}

    Returns:
        {asset_name: weight} - optimal portfolio weights
    """

    scenarios = list(probabilities.keys())
    assets = list(next(iter(scenario_returns.values())).keys())

    # Create LP problem
    prob = lp.LpProblem("Expectimin", lp.LpMinimize)

    # Decision variables
    weights = {asset: lp.LpVariable(f"w_{asset}", 0, 1) for asset in assets}
    losses = {scenario: lp.LpVariable(f"loss_{scenario}", 0) for scenario in scenarios}

    # Objective: minimize expected loss
    prob += lp.lpSum([probabilities[s] * losses[s] for s in scenarios])

    # Constraints
    prob += lp.lpSum(weights.values()) == 1  # weights sum to 1

    for scenario in scenarios:
        # Portfolio return for this scenario
        portfolio_return = lp.lpSum([
            weights[asset] * scenario_returns[scenario][asset] / 100.0
            for asset in assets
        ])

        # Loss = max(0, -portfolio_return)
        prob += losses[scenario] >= -portfolio_return

    # Solve
    prob.solve(lp.PULP_CBC_CMD(msg=0))

    if prob.status == lp.LpStatusOptimal:
        optimal_weights = {asset: weights[asset].varValue or 0.0 for asset in assets}

        # Normalize weights (handle numerical errors)
        total = sum(optimal_weights.values())
        if total > 0:
            optimal_weights = {asset: w/total for asset, w in optimal_weights.items()}
        else:
            # Fallback to equal weights
            optimal_weights = {asset: 1.0/len(assets) for asset in assets}

        return optimal_weights
    else:
        # Fallback to equal weights
        return {asset: 1.0/len(assets) for asset in assets}


def sequential_optimize(initial_probabilities: Dict[str, float],
                       scenario_forecasts: Dict[str, Dict[str, List[float]]],
                       evidence_path: List[float],  # [gdp1, gdp2, gdp3, inf1, inf2, inf3]
                       learning_function: Callable,
                       scenarios_dict: Dict[str, Dict],  # Add scenarios parameter
                       covariance_matrix: np.ndarray,
                       horizon: int = 3) -> List[Dict[str, float]]:
    """
    Sequential portfolio optimization with learning over multiple periods

    Args:
        initial_probabilities: Initial scenario probabilities
        scenario_forecasts: {scenario: {asset: [year1, year2, year3]}}
        evidence_path: Realized economic path [gdp1, gdp2, gdp3, inf1, inf2, inf3]
        learning_function: Function to update beliefs
        scenarios_dict: Scenario definitions for learning
        covariance_matrix: For Mahalanobis distance calculation
        horizon: Number of years

    Returns:
        List of optimal weights for each year [w0, w1, w2]
    """

    weights_sequence = []
    current_beliefs = initial_probabilities.copy()

    # Split evidence path into GDP and inflation
    gdp_path = evidence_path[:horizon]
    inf_path = evidence_path[horizon:]

    for year in range(horizon):
        # Get current year forecasts for all scenarios
        annual_scenario_returns = {}
        for scenario, forecasts in scenario_forecasts.items():
            annual_scenario_returns[scenario] = {
                asset: forecasts[asset][year] for asset in forecasts.keys()
            }

        # Optimize portfolio for current year
        optimal_weights = expectimin_optimize(current_beliefs, annual_scenario_returns)
        weights_sequence.append(optimal_weights)

        # Update beliefs for next year (except in last year)
        if year < horizon - 1:
            # Evidence observed so far
            observed_gdp = gdp_path[:year+1]
            observed_inf = inf_path[:year+1]

            # Update beliefs using learning function
            current_beliefs = learning_function(
                current_beliefs, observed_gdp, observed_inf, scenarios_dict, covariance_matrix
            )

    return weights_sequence


def calculate_portfolio_performance(weights_sequence: List[Dict[str, float]],
                                  actual_returns: Dict[str, List[float]]) -> Dict:
    """
    Calculate portfolio performance given weights and actual returns

    Args:
        weights_sequence: [w0, w1, w2] portfolio weights for each year
        actual_returns: {asset: [year1, year2, year3]} actual returns

    Returns:
        Performance metrics dictionary
    """

    portfolio_value = 1.0
    annual_returns = []

    for year, weights in enumerate(weights_sequence):
        # Calculate portfolio return for this year
        portfolio_return = sum(
            weights[asset] * actual_returns[asset][year] / 100.0
            for asset in weights.keys()
        )

        annual_returns.append(portfolio_return)
        portfolio_value *= (1 + portfolio_return)

    cumulative_return = portfolio_value - 1.0

    return {
        'cumulative_return': cumulative_return,
        'annual_returns': annual_returns,
        'final_value': portfolio_value
    }

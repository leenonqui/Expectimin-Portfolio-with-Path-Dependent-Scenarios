"""
optimization.py
Simple portfolio optimization with mean-variance utility constraint
"""

import numpy as np
import scipy.optimize as opt
from typing import Dict
from constants import REAL_RISK_FREE_RATES, RISK_AVERSION


def optimize_portfolio(probabilities: Dict[str, float],
                      scenario_returns: Dict[str, Dict[str, float]],
                      year: int,
                      risk_aversion: float = RISK_AVERSION) -> Dict[str, float]:
    """
    Find portfolio weights that minimize expected loss
    Subject to: E[return] ≥ ½ × A × Var[return] + risk_free_rate

    Args:
        probabilities: {scenario: probability}
        scenario_returns: {scenario: {asset: return_%}}
        year: Year index (0, 1, 2) for risk-free rate
        risk_aversion: Risk aversion parameter A

    Returns:
        {asset: weight}
    """
    # Setup
    scenarios = list(probabilities.keys())
    assets = list(next(iter(scenario_returns.values())).keys())
    n_assets = len(assets)

    # Convert to arrays
    probs = np.array([probabilities[s] for s in scenarios])
    returns_matrix = np.array([[scenario_returns[s][asset]/100.0 for asset in assets]
                              for s in scenarios])

    # Get risk-free rate
    rf_year = 2018 + year
    risk_free_rate = REAL_RISK_FREE_RATES.get(rf_year, 0.0)

    def objective_and_constraint(weights):
        """Calculate expected loss and utility"""
        weights = np.array(weights)

        # Portfolio returns for each scenario
        portfolio_returns = returns_matrix @ weights

        # Expected return and variance
        expected_return = probs @ portfolio_returns
        variance = probs @ ((portfolio_returns - expected_return) ** 2)

        # Expected loss (objective)
        losses = np.maximum(0, -portfolio_returns)
        expected_loss = probs @ losses

        # Utility constraint: E[r] - ½×A×Var[r] - rf ≥ 0
        utility = expected_return - 0.5 * risk_aversion * variance -risk_free_rate

        return expected_loss, utility

    def objective(weights):
        expected_loss, _ = objective_and_constraint(weights)
        return expected_loss

    def utility_constraint(weights):
        _, utility = objective_and_constraint(weights)
        return utility

    # Constraints and bounds
    constraints = [
        {'type': 'ineq', 'fun': utility_constraint},  # Utility ≥ 0
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
    ]
    bounds = [(0.0, 1.0) for _ in range(n_assets)]

    # Solve optimization
    x0 = np.ones(n_assets) / n_assets  # Equal weights start

    result = opt.minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )

    # Return results
    if result.success:
        weights_array = result.x / np.sum(result.x)  # Normalize
        return {assets[i]: weights_array[i] for i in range(n_assets)}
    else:
        # Fallback to equal weights
        return {asset: 1.0/n_assets for asset in assets}

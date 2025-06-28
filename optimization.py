"""
optimization.py
Simple Linear Programming formulation using PuLP
Minimize expected losses with cash constraint
"""

import pulp
from typing import Dict


def optimize_portfolio(probabilities: Dict[str, float],
                      scenario_returns: Dict[str, Dict[str, float]],
                      year: int
                    ) -> Dict[str, float]:
    """
    Linear Programming: Minimize expected losses

    Variables:
        w_asset = weight in each asset
        loss_scenario = loss in each scenario

    Minimize: sum(probability × loss)

    Subject to:
        sum(weights) = 1
        0 ≤ w_cash ≤ 0.10
        w_stocks ≥ 0.25
        w_bonds ≥ 0
        loss ≥ 0
        loss ≥ -portfolio_return  (captures max(0, -return))
    """

    scenarios = list(probabilities.keys())
    assets = list(next(iter(scenario_returns.values())).keys())

    # Create LP problem
    prob = pulp.LpProblem("Portfolio_Optimization", pulp.LpMinimize)

    # Variables: Portfolio weights
    weights = {}
    for asset in assets:
        if asset == 'Cash':
            weights[asset] = pulp.LpVariable(f"w_{asset}", lowBound=0, upBound=0.10)
        elif asset == 'Stocks':
            weights[asset] = pulp.LpVariable(f"w_{asset}", lowBound=0.25)
        else:
            weights[asset] = pulp.LpVariable(f"w_{asset}", lowBound=0)

    # Variables: Loss in each scenario
    losses = {}
    for scenario in scenarios:
        losses[scenario] = pulp.LpVariable(f"loss_{scenario}", lowBound=0)

    # Objective: Minimize expected loss
    prob += pulp.lpSum([probabilities[s] * losses[s] for s in scenarios])

    # Constraint: Weights sum to 1
    prob += pulp.lpSum([weights[asset] for asset in assets]) == 1

    # Constraint: Define losses for each scenario
    for scenario in scenarios:
        portfolio_return = pulp.lpSum([
            weights[asset] * scenario_returns[scenario][asset] / 100.0
            for asset in assets
        ])
        # loss ≥ -portfolio_return (captures max(0, -return))
        prob += losses[scenario] >= -portfolio_return

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Extract results
    if prob.status == pulp.LpStatusOptimal:
        result = {}
        for asset in assets:
            result[asset] = weights[asset].varValue

        # Normalize (ensure sum = 1)
        total = sum(result.values())
        for asset in assets:
            result[asset] /= total

        return result

    else:
        # Fallback to simple allocation
        return {'Cash': 0.10, 'Stocks': 0.45, 'Bonds': 0.45}

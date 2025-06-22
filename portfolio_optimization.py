"""
Portfolio Optimization Module for Bachelor Thesis
Implements Expectimin portfolio optimization by minimizing expected cumulative loss
Uses scipy to handle non-convex cumulative returns with weight drift
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from constants import ASSET_CLASSES, HORIZON


class ExpectiminOptimizer:
    """
    Expectimin Portfolio Optimizer using exact cumulative loss minimization.

    Handles:
    - Non-convex cumulative returns using scipy optimization
    - Weight drift over time (no rebalancing assumption)
    - Exact expected loss calculation
    """

    def __init__(self, asset_classes: List[str] = None):
        self.asset_classes = asset_classes or ASSET_CLASSES
        self.n_assets = len(self.asset_classes)

    def calculate_cumulative_return_with_drift(self, weights, scenario_returns):
        """
        Calculate cumulative return accounting for weight drift.

        Without rebalancing:
        - Year 1: Portfolio return = w₀ · r₁, new weights = w₀(1+r₁) / Σw₀(1+r₁)
        - Year 2: Portfolio return = w₁ · r₂, new weights = w₁(1+r₂) / Σw₁(1+r₂)
        - Year 3: Portfolio return = w₂ · r₃

        Args:
            weights: Initial portfolio weights [w₀_cash, w₀_stocks, w₀_bonds]
            scenario_returns: Asset returns for each year [[r₁], [r₂], [r₃]]

        Returns:
            Cumulative return over the entire horizon (percentage)
        """
        current_weights = np.array(weights)
        portfolio_value = 1.0

        for year_returns in scenario_returns:
            # Convert percentage returns to decimal
            year_returns_decimal = np.array(year_returns) / 100.0

            # Calculate portfolio return for this year
            portfolio_return = np.sum(current_weights * year_returns_decimal)

            # Update portfolio value
            portfolio_value *= (1 + portfolio_return)

            # Update weights due to drift (for next year)
            new_asset_values = current_weights * (1 + year_returns_decimal)
            current_weights = new_asset_values / np.sum(new_asset_values)

        # Return cumulative return as percentage
        return (portfolio_value - 1) * 100

    def optimize_expectimin_cumulative_loss(self,
                                          scenario_forecasts: Dict[str, Dict[str, List[float]]],
                                          probabilities: Dict[str, float],
                                          min_return: Optional[float] = None) -> Dict:
        """
        Minimize expected cumulative loss using scipy optimization.

        Objective: minimize E[Loss] = Σ(p_s × max(0, -Cumulative_Return_s))

        Args:
            scenario_forecasts: {scenario_name: {asset_name: [year1, year2, year3]}}
            probabilities: {scenario_name: probability}
            min_return: Optional minimum expected cumulative return (decimal)

        Returns:
            Optimization results dictionary
        """
        print(f"EXPECTIMIN CUMULATIVE LOSS OPTIMIZATION")
        print(f"Method: Minimize expected loss subject to minimum return constraint")
        print("="*70)

        scenarios = list(scenario_forecasts.keys())
        n_scenarios = len(scenarios)

        # Prepare scenario data
        scenario_data = []
        for scenario_name in scenarios:
            asset_returns = scenario_forecasts[scenario_name]
            # Convert to year-by-year format: [[year1_returns], [year2_returns], [year3_returns]]
            year_returns = []
            for year in range(HORIZON):
                year_returns.append([asset_returns[asset][year] for asset in self.asset_classes])
            scenario_data.append(year_returns)

        prob_array = np.array([probabilities[scenario] for scenario in scenarios])

        def objective(weights):
            """Calculate expected cumulative loss"""
            total_expected_loss = 0.0

            for i, scenario_returns in enumerate(scenario_data):
                # Calculate cumulative return with weight drift
                cum_return = self.calculate_cumulative_return_with_drift(weights, scenario_returns)

                # Loss = max(0, -cumulative_return_percent) converted to decimal
                loss_percent = max(0, -cum_return)
                loss_decimal = loss_percent / 100  # Convert percentage to decimal

                # Add probability-weighted loss
                total_expected_loss += prob_array[i] * loss_decimal

            return total_expected_loss  # Return as decimal (e.g., 0.05 for 5%)

        def constraint_sum_to_one(weights):
            """Weights must sum to 1"""
            return np.sum(weights) - 1.0

        def constraint_min_return(weights):
            """Expected return must exceed minimum required return"""
            if min_return is None:
                return 1.0  # Always satisfied

            expected_return = 0.0
            for i, scenario_returns in enumerate(scenario_data):
                cum_return = self.calculate_cumulative_return_with_drift(weights, scenario_returns)
                expected_return += prob_array[i] * cum_return

            expected_return_decimal = expected_return / 100  # Convert to decimal
            constraint_value = expected_return_decimal - min_return
            return constraint_value  # Should be ≥ 0 for constraint satisfaction

        # Set up constraints
        constraints = [{'type': 'eq', 'fun': constraint_sum_to_one}]

        if min_return is not None:
            constraints.append({'type': 'ineq', 'fun': constraint_min_return})
            print(f"Added constraint: E[Return] ≥ {min_return:.2%}")

        # Bounds: weights between 0 and 1
        bounds = [(0, 1) for _ in range(self.n_assets)]

        # Initial guess: equal weights
        x0 = np.ones(self.n_assets) / self.n_assets

        print(f"Solving optimization...")
        print(f"Variables: {self.n_assets} portfolio weights")
        print(f"Scenarios: {n_scenarios}")
        print(f"Constraints: {len(constraints)}")

        # Solve optimization
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'ftol': 1e-9, 'disp': False, 'maxiter': 1000}
            )

            if result.success:
                optimal_weights = result.x
                weights_dict = dict(zip(self.asset_classes, optimal_weights))
                expected_loss = result.fun

                print(f"✓ Optimization successful!")
                print(f"Expected Loss: {expected_loss:.4f} ({expected_loss*100:.2f}%)")
                print(f"Optimal weights: {', '.join(f'{asset}={weight:.3f}' for asset, weight in weights_dict.items())}")

                # Calculate comprehensive results
                portfolio_metrics = self._calculate_portfolio_metrics(
                    scenario_forecasts, probabilities, weights_dict
                )

                # Verify constraint satisfaction
                if min_return is not None:
                    actual_return = portfolio_metrics['expected_cumulative_return']
                    constraint_check = actual_return - min_return
                    print(f"Return check: {actual_return:.4f} vs required {min_return:.4f}")
                    if constraint_check < -1e-6:
                        print(f"⚠️  WARNING: Return constraint violated!")
                    elif abs(constraint_check) < 1e-3:
                        print(f"✓ Return constraint is binding")
                    else:
                        print(f"✓ Return constraint satisfied")

                return {
                    'success': True,
                    'weights': weights_dict,
                    'expected_cumulative_loss': expected_loss,
                    'solver_used': 'scipy.optimize.minimize',
                    'optimization_status': 'optimal',
                    **portfolio_metrics
                }

            else:
                print(f"✗ Optimization failed: {result.message}")
                return {
                    'success': False,
                    'message': f"Scipy optimization failed: {result.message}"
                }

        except Exception as e:
            print(f"✗ Optimization error: {str(e)}")
            return {
                'success': False,
                'message': f"Optimization error: {str(e)}"
            }

    def _calculate_portfolio_metrics(self,
                                   scenario_forecasts: Dict[str, Dict[str, List[float]]],
                                   probabilities: Dict[str, float],
                                   weights: Dict[str, float]) -> Dict:
        """
        Calculate comprehensive portfolio performance metrics.
        """
        scenario_cumulative_returns = {}
        scenario_losses = {}

        # Convert weights to array
        weights_array = np.array([weights[asset] for asset in self.asset_classes])

        for scenario_name, asset_returns in scenario_forecasts.items():
            # Prepare year-by-year returns
            year_returns = []
            for year in range(HORIZON):
                year_returns.append([asset_returns[asset][year] for asset in self.asset_classes])

            # Calculate cumulative return with weight drift
            cum_return = self.calculate_cumulative_return_with_drift(weights_array, year_returns)
            scenario_cumulative_returns[scenario_name] = cum_return / 100  # Convert to decimal

            # Calculate loss
            loss = max(0, -cum_return)
            scenario_losses[scenario_name] = loss / 100  # Convert to decimal

        # Calculate probability-weighted metrics
        expected_cumulative = sum(probabilities[scenario] * cum_return
                                for scenario, cum_return in scenario_cumulative_returns.items())

        expected_loss = sum(probabilities[scenario] * loss
                          for scenario, loss in scenario_losses.items())

        # Risk metrics
        worst_case = min(scenario_cumulative_returns.values())
        best_case = max(scenario_cumulative_returns.values())

        # Loss analysis
        loss_scenarios = [s for s, r in scenario_cumulative_returns.items() if r < 0]
        prob_of_loss = sum(probabilities[scenario] for scenario in loss_scenarios)

        # Standard deviation
        variance = sum(probabilities[scenario] * (cum_return - expected_cumulative)**2
                      for scenario, cum_return in scenario_cumulative_returns.items())
        std_dev = np.sqrt(variance)

        return {
            'expected_cumulative_return': expected_cumulative,
            'expected_cumulative_loss': expected_loss,  # Override with exact calculation
            'scenario_cumulative_returns': scenario_cumulative_returns,
            'scenario_losses': scenario_losses,
            'worst_case_cumulative': worst_case,
            'best_case_cumulative': best_case,
            'cumulative_std_dev': std_dev,
            'probability_of_loss': prob_of_loss,
            'loss_scenarios': loss_scenarios,
            'number_of_loss_scenarios': len(loss_scenarios)
        }

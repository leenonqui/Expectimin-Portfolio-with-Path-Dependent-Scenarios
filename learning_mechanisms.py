"""
learning_mechanisms.py
Corrected belief updating strategies for portfolio choice game theory analysis

Strategies:
1. No Learning: Fixed beliefs throughout horizon
2. Bayesian: Proper Bayesian updating using raw Mahalanobis distance likelihoods
3. Adaptive: Partial learning with configurable lambda parameter
"""

import numpy as np
import pulp as lp
from typing import Dict, List
from utils import calculate_mahalanobis_distance, create_path_vector, safe_matrix_inverse
from constants import SCENARIOS


class BeliefUpdater:
    """Handles different belief updating mechanisms with corrected implementation"""

    def __init__(self, learning_type: str, learning_rate: float = 0.3):
        """
        Args:
            learning_type: "no_learning", "bayesian", "adaptive"
            learning_rate: Lambda parameter for adaptive learning (0 < λ ≤ 1)
                          λ = 0.1 (slow learning), λ = 0.5 (moderate), λ = 1.0 (full Bayesian)
        """
        self.learning_type = learning_type
        self.learning_rate = learning_rate

    def update_beliefs(self, prior_beliefs: Dict[str, float],
                      observed_path: List[float],
                      observed_inflation: List[float],
                      covariance_matrix: np.ndarray) -> Dict[str, float]:
        """
        Update beliefs based on observed GDP growth and inflation

        Args:
            prior_beliefs: Current belief probabilities
            observed_path: Observed GDP growth path so far
            observed_inflation: Observed inflation path so far
            covariance_matrix: Historical covariance matrix for Mahalanobis distance

        Returns:
            Updated belief probabilities
        """
        if self.learning_type == "no_learning":
            return prior_beliefs.copy()

        # Get raw likelihoods (not normalized)
        raw_likelihoods = self._calculate_raw_likelihoods(
            observed_path, observed_inflation, covariance_matrix
        )

        # Apply specific learning mechanism
        if self.learning_type == "bayesian":
            return self._bayesian_update(prior_beliefs, raw_likelihoods)
        elif self.learning_type == "adaptive":
            return self._adaptive_update(prior_beliefs, raw_likelihoods)
        else:
            return prior_beliefs.copy()

    def _calculate_raw_likelihoods(self, observed_gdp: List[float],
                                  observed_inflation: List[float],
                                  covariance_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate RAW likelihoods P(data|scenario) using Mahalanobis distance"""

        n_periods = len(observed_gdp)
        observed_vector = create_path_vector(observed_gdp, observed_inflation)

        # Use subset of covariance matrix for available periods
        cov_subset = covariance_matrix[:2*n_periods, :2*n_periods]
        cov_inv = safe_matrix_inverse(cov_subset)

        raw_likelihoods = {}
        for scenario_name, scenario_def in SCENARIOS.items():
            # Get predicted path for same number of periods
            predicted_gdp = scenario_def.gdp_growth[:n_periods]
            predicted_inflation = scenario_def.inflation[:n_periods]
            predicted_vector = create_path_vector(predicted_gdp, predicted_inflation)

            # Mahalanobis distance squared
            d_squared = calculate_mahalanobis_distance(
                observed_vector, predicted_vector, cov_inv
            )

            # RAW likelihood: exp(-d²/2) - NOT NORMALIZED YET
            raw_likelihood = np.exp(-d_squared / 2.0)
            raw_likelihoods[scenario_name] = raw_likelihood

        return raw_likelihoods

    def _bayesian_update(self, prior_beliefs: Dict[str, float],
                        raw_likelihoods: Dict[str, float]) -> Dict[str, float]:
        """
        PROPER Bayesian updating using RAW likelihoods
        P(scenario|data) = P(data|scenario) × P(scenario) / P(data)
        """

        # Evidence: P(data) = Σ P(data|scenario) × P(scenario)
        evidence = sum(raw_likelihoods[s] * prior_beliefs[s] for s in prior_beliefs.keys())

        # Posterior: P(scenario|data) = P(data|scenario) × P(scenario) / P(data)
        posterior = {}
        for scenario in prior_beliefs.keys():
            if evidence > 0:
                posterior[scenario] = (raw_likelihoods[scenario] * prior_beliefs[scenario]) / evidence
            else:
                posterior[scenario] = prior_beliefs[scenario]

        return posterior

    def _adaptive_update(self, prior_beliefs: Dict[str, float],
                        raw_likelihoods: Dict[str, float]) -> Dict[str, float]:
        """
        Adaptive learning using proper Bayesian posteriors
        P_new = P_old + λ × (P_bayesian - P_old)
        """

        # First get Bayesian posterior
        bayesian_posterior = self._bayesian_update(prior_beliefs, raw_likelihoods)

        # Then apply adaptive formula
        updated = {}
        for scenario in prior_beliefs.keys():
            old_prob = prior_beliefs[scenario]
            bayesian_prob = bayesian_posterior[scenario]
            updated[scenario] = old_prob + self.learning_rate * (bayesian_prob - old_prob)

        return updated


class PortfolioOptimizer:
    """Expectimin portfolio optimizer using Linear Programming"""

    def __init__(self, asset_classes: List[str]):
        self.asset_classes = asset_classes

    def optimize_expectimin(self, beliefs: Dict[str, float],
                           annual_forecasts: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        PROPER Expectimin optimization using Linear Programming

        PROBLEM: minimize E[Loss] = Σ P(scenario) × max(0, -portfolio_return_scenario)

        CHALLENGE: max(0, x) is non-linear, can't optimize directly

        SOLUTION: Use auxiliary "loss" variables with constraints:
        - loss_s ≥ -portfolio_return_s  (captures the max function)
        - loss_s ≥ 0                    (losses are non-negative)
        - minimize Σ P(s) × loss_s      (linear objective!)

        RESULT: Solver automatically sets loss_s = max(0, -portfolio_return_s)

        NUMERICAL EXAMPLE:
        Scenario 1: portfolio_return = +8%  → solver sets loss_1 = 0% (no loss)
        Scenario 2: portfolio_return = -3%  → solver sets loss_2 = 3% (actual loss)
        Scenario 3: portfolio_return = +2%  → solver sets loss_3 = 0% (no loss)

        If P(S1)=0.5, P(S2)=0.3, P(S3)=0.2, then:
        Expected Loss = 0.5×0% + 0.3×3% + 0.2×0% = 0.9%

        WHY IT WORKS:
        The LP solver minimizes the loss variables subject to constraints.
        Since loss_s ≥ -portfolio_return_s and loss_s ≥ 0, the solver chooses:
        - loss_s = 0 when portfolio_return_s > 0 (constraint allows it, minimization prefers it)
        - loss_s = -portfolio_return_s when portfolio_return_s < 0 (constraint forces it)
        This exactly implements max(0, -portfolio_return_s)!
        """

        scenarios = list(beliefs.keys())

        # Create LP problem
        prob = lp.LpProblem("Expectimin_Annual", lp.LpMinimize)

        # Decision variables: portfolio weights
        weights = {asset: lp.LpVariable(f"w_{asset}", 0, 1) for asset in self.asset_classes}

        # Auxiliary variables: losses for each scenario (≥ 0)
        losses = {scenario: lp.LpVariable(f"loss_{scenario}", 0) for scenario in scenarios}

        # OBJECTIVE: minimize expected loss (now linear!)
        prob += lp.lpSum([beliefs[s] * losses[s] for s in scenarios])

        # CONSTRAINT 1: weights sum to 1
        prob += lp.lpSum(weights.values()) == 1

        # CONSTRAINT 2: loss definition for each scenario
        # Force: loss_s ≥ max(0, -portfolio_return_s)
        for scenario in scenarios:
            portfolio_return = lp.lpSum([
                weights[asset] * annual_forecasts[scenario][asset] / 100.0
                for asset in self.asset_classes
            ])
            # This constraint + minimization objective ensures loss_s = max(0, -portfolio_return_s)
            prob += losses[scenario] >= -portfolio_return

        # SOLVE using CBC (Coin-OR Branch and Cut) solver
        prob.solve(lp.PULP_CBC_CMD(msg=0))

        if prob.status == lp.LpStatusOptimal:
            optimal_weights = {asset: weights[asset].varValue for asset in self.asset_classes}

            # Optional: Print solution details
            expected_loss = sum(beliefs[s] * losses[s].varValue for s in scenarios)
            print(f"  Expectimin LP solved: Expected Loss = {expected_loss:.4f}")

            return optimal_weights
        else:
            print(f"  LP solver failed with status: {lp.LpStatus[prob.status]}")
            # Fallback to equal weights
            n_assets = len(self.asset_classes)
            return {asset: 1.0/n_assets for asset in self.asset_classes}

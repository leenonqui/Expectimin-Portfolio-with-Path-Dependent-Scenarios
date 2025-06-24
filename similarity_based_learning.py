"""
similarity_based_learning.py
GDP Similarity-Based Learning for Portfolio Choice
Implements Bayesian and Partial learning using actual observed GDP vs predicted GDP

Key Innovation:
- Uses similarity between observed and predicted GDP paths
- Proper Bayesian updating: P(scenario|gdp) = P(gdp|scenario) × P(scenario) / P(gdp)
- Partial learning with normalized likelihoods
- Sequential belief updating as more GDP data becomes available
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import copy

from constants import SCENARIOS, ASSET_CLASSES, HORIZON


@dataclass
class ObservedEconomicData:
    """Container for actual observed economic outcomes"""
    years: List[int]
    gdp_growth: List[float]
    inflation: List[float]
    asset_returns: Dict[str, List[float]]  # {asset: [year1, year2, year3]}


class SimilarityBasedLearner:
    """
    Implements similarity-based learning using observed GDP vs predicted GDP

    Two learning mechanisms:
    1. Bayesian: P(scenario|gdp) = P(gdp|scenario) × P(scenario) / P(gdp)
    2. Partial: New_P = λ × normalized_likelihood + (1-λ) × Old_P
    """

    def __init__(self, learning_type: str = "bayesian", learning_rate: float = 0.3,
                 similarity_measure: str = "euclidean", bandwidth: float = 2.0):
        """
        Args:
            learning_type: "bayesian", "partial", "no_learning", or "perfect"
            learning_rate: λ for partial learning (0 < λ < 1)
            similarity_measure: "euclidean" or "mahalanobis"
            bandwidth: σ parameter for likelihood calculation
        """
        self.learning_type = learning_type
        self.learning_rate = learning_rate
        self.similarity_measure = similarity_measure
        self.bandwidth = bandwidth

    def calculate_gdp_likelihood(self, observed_gdp_path: List[float],
                                predicted_gdp_path: List[float]) -> float:
        """
        Calculate P(gdp_witnessed | scenario) using similarity

        Args:
            observed_gdp_path: Actual GDP growth observed [2.9%, 2.2%, ...]
            predicted_gdp_path: Scenario's predicted GDP growth [1.3%, 0.4%, ...]

        Returns:
            Likelihood P(observed | predicted scenario)
        """

        # Ensure equal length (use available data)
        min_length = min(len(observed_gdp_path), len(predicted_gdp_path))
        obs_path = observed_gdp_path[:min_length]
        pred_path = predicted_gdp_path[:min_length]

        if len(obs_path) == 0:
            return 1.0  # No data to compare

        # Calculate distance between paths
        if self.similarity_measure == "euclidean":
            distance = self._euclidean_distance(obs_path, pred_path)
        elif self.similarity_measure == "mahalanobis":
            distance = self._mahalanobis_distance(obs_path, pred_path)
        else:
            distance = self._euclidean_distance(obs_path, pred_path)

        # Convert distance to likelihood using Gaussian kernel
        # Higher similarity (lower distance) → Higher likelihood
        likelihood = np.exp(-distance**2 / (2 * self.bandwidth**2))

        return likelihood

    def _euclidean_distance(self, path1: List[float], path2: List[float]) -> float:
        """Calculate Euclidean distance between GDP paths"""
        return np.sqrt(sum((x1 - x2)**2 for x1, x2 in zip(path1, path2)))

    def _mahalanobis_distance(self, path1: List[float], path2: List[float]) -> float:
        """
        Calculate Mahalanobis distance (simplified version)
        Uses inverse of sample variance as precision matrix
        """
        diff = np.array(path1) - np.array(path2)

        # Simple precision matrix (inverse variance) - can be improved
        if len(diff) > 1:
            precision = 1.0 / (np.var(diff) + 1e-6)  # Add small constant for stability
        else:
            precision = 1.0

        # Mahalanobis distance squared
        mahal_dist_squared = np.sum(diff**2 * precision)
        return np.sqrt(mahal_dist_squared)

    def update_beliefs_with_gdp(self, prior_beliefs: Dict[str, float],
                               observed_gdp_path: List[float],
                               scenario_forecasts: Dict[str, Dict[str, List[float]]]) -> Dict[str, float]:
        """
        Update beliefs based on observed GDP using similarity-based learning

        Args:
            prior_beliefs: P(scenario) before update
            observed_gdp_path: Actual GDP growth observed
            scenario_forecasts: Predicted GDP paths for each scenario

        Returns:
            Updated beliefs P(scenario | observed_gdp)
        """

        if self.learning_type == "no_learning":
            return prior_beliefs.copy()

        # Extract predicted GDP paths from forecasts
        predicted_gdp_paths = {}
        for scenario_name, forecasts in scenario_forecasts.items():
            # GDP growth is embedded in the scenarios - extract from scenario definition
            scenario_def = SCENARIOS[scenario_name]
            predicted_gdp_paths[scenario_name] = scenario_def.gdp_growth

        # Calculate likelihoods P(gdp_witnessed | scenario)
        likelihoods = {}
        for scenario_name, predicted_path in predicted_gdp_paths.items():
            likelihood = self.calculate_gdp_likelihood(observed_gdp_path, predicted_path)
            likelihoods[scenario_name] = likelihood

        # Normalize likelihoods to sum to 1
        total_likelihood = sum(likelihoods.values())
        if total_likelihood > 0:
            normalized_likelihoods = {s: l/total_likelihood for s, l in likelihoods.items()}
        else:
            # Fallback: equal likelihoods
            normalized_likelihoods = {s: 1.0/len(likelihoods) for s in likelihoods.keys()}

        # Apply learning mechanism
        if self.learning_type == "bayesian":
            return self._bayesian_update(prior_beliefs, normalized_likelihoods)
        elif self.learning_type == "partial":
            return self._partial_update(prior_beliefs, normalized_likelihoods)
        elif self.learning_type == "perfect":
            return self._perfect_update(normalized_likelihoods)
        else:
            return prior_beliefs.copy()

    def _bayesian_update(self, prior_beliefs: Dict[str, float],
                        normalized_likelihoods: Dict[str, float]) -> Dict[str, float]:
        """
        Proper Bayesian updating: P(scenario|gdp) = P(gdp|scenario) × P(scenario) / P(gdp)
        """

        # Calculate evidence: P(gdp) = Σ P(gdp|scenario) × P(scenario)
        evidence = sum(normalized_likelihoods[scenario] * prior_beliefs[scenario]
                      for scenario in prior_beliefs.keys())

        # Calculate posterior: P(scenario|gdp)
        posterior_beliefs = {}
        for scenario in prior_beliefs.keys():
            if evidence > 0:
                posterior = (normalized_likelihoods[scenario] * prior_beliefs[scenario]) / evidence
            else:
                posterior = prior_beliefs[scenario]  # Fallback
            posterior_beliefs[scenario] = posterior

        # Verify normalization (should sum to 1)
        total = sum(posterior_beliefs.values())
        if abs(total - 1.0) > 1e-6:
            print(f"⚠️  Warning: Posterior probabilities sum to {total:.6f}, renormalizing")
            posterior_beliefs = {s: p/total for s, p in posterior_beliefs.items()}

        return posterior_beliefs

    def _partial_update(self, prior_beliefs: Dict[str, float],
                       normalized_likelihoods: Dict[str, float]) -> Dict[str, float]:
        """
        Partial learning: New_P = λ × normalized_likelihood + (1-λ) × Old_P
        """

        updated_beliefs = {}
        for scenario in prior_beliefs.keys():
            updated_beliefs[scenario] = (self.learning_rate * normalized_likelihoods[scenario] +
                                       (1 - self.learning_rate) * prior_beliefs[scenario])

        # Verify normalization (should automatically sum to 1 due to convex combination)
        total = sum(updated_beliefs.values())
        if abs(total - 1.0) > 1e-6:
            print(f"⚠️  Warning: Partial learning probabilities sum to {total:.6f}, renormalizing")
            updated_beliefs = {s: p/total for s, p in updated_beliefs.items()}

        return updated_beliefs

    def _perfect_update(self, normalized_likelihoods: Dict[str, float]) -> Dict[str, float]:
        """
        Perfect learning: Complete confidence in most likely scenario
        """

        # Find scenario with highest likelihood
        best_scenario = max(normalized_likelihoods.items(), key=lambda x: x[1])[0]

        # Assign all probability to best scenario
        perfect_beliefs = {scenario: 1.0 if scenario == best_scenario else 0.0
                          for scenario in normalized_likelihoods.keys()}

        return perfect_beliefs


class SequentialPortfolioOptimizer:
    """
    Optimizes portfolio sequentially using similarity-based learning
    Updates beliefs year by year as GDP data becomes available
    """

    def __init__(self, asset_classes: List[str] = None):
        self.asset_classes = asset_classes or ASSET_CLASSES

    def optimize_with_sequential_learning(self,
                                        initial_beliefs: Dict[str, float],
                                        scenario_forecasts: Dict[str, Dict[str, List[float]]],
                                        observed_data: ObservedEconomicData,
                                        learning_mechanism: SimilarityBasedLearner) -> Dict:
        """
        Run sequential portfolio optimization with GDP-based learning

        Process:
        1. Start with initial beliefs
        2. Choose Year 1 portfolio
        3. Observe Year 1 GDP → Update beliefs
        4. Choose Year 2 portfolio
        5. Observe Year 2 GDP → Update beliefs
        6. Choose Year 3 portfolio
        7. Calculate final performance
        """

        print(f"📊 Sequential Portfolio Optimization")
        print(f"Learning mechanism: {learning_mechanism.learning_type}")
        print(f"Observed GDP path: {observed_data.gdp_growth}")

        # Track evolution
        belief_evolution = [initial_beliefs.copy()]
        portfolio_evolution = []
        period_results = []
        cumulative_return = 0.0

        current_beliefs = initial_beliefs.copy()

        for period in range(HORIZON):
            print(f"\n--- Period {period + 1} (Year {observed_data.years[period]}) ---")

            # Optimize portfolio for this period given current beliefs
            portfolio = self._optimize_single_period(current_beliefs, scenario_forecasts, period)
            portfolio_evolution.append(portfolio)

            print(f"Portfolio allocation: {self._format_portfolio(portfolio)}")

            # Calculate realized return for this period
            if period < len(observed_data.asset_returns[ASSET_CLASSES[0]]):
                realized_return = sum(
                    portfolio[asset] * observed_data.asset_returns[asset][period] / 100.0
                    for asset in self.asset_classes
                )
                cumulative_return += realized_return

                print(f"Realized return: {realized_return*100:.2f}%")

                period_results.append({
                    'period': period,
                    'year': observed_data.years[period],
                    'portfolio': portfolio.copy(),
                    'realized_return': realized_return,
                    'beliefs_before': current_beliefs.copy()
                })

                # Update beliefs based on observed GDP (for next period)
                if period < HORIZON - 1:  # Don't update after last period
                    observed_gdp_so_far = observed_data.gdp_growth[:period+1]

                    print(f"Observed GDP so far: {observed_gdp_so_far}")
                    print(f"Beliefs before update: {self._format_beliefs(current_beliefs)}")

                    current_beliefs = learning_mechanism.update_beliefs_with_gdp(
                        current_beliefs, observed_gdp_so_far, scenario_forecasts
                    )

                    print(f"Beliefs after update: {self._format_beliefs(current_beliefs)}")
                    belief_evolution.append(current_beliefs.copy())

        print(f"\n📈 Final Results:")
        print(f"Cumulative return: {cumulative_return*100:.2f}%")
        print(f"Final beliefs: {self._format_beliefs(current_beliefs)}")

        return {
            'learning_type': learning_mechanism.learning_type,
            'belief_evolution': belief_evolution,
            'portfolio_evolution': portfolio_evolution,
            'period_results': period_results,
            'cumulative_return': cumulative_return,
            'final_beliefs': current_beliefs
        }

    def _optimize_single_period(self, beliefs: Dict[str, float],
                               scenario_forecasts: Dict[str, Dict[str, List[float]]],
                               period: int) -> Dict[str, float]:
        """Optimize portfolio for single period using current beliefs"""

        # Simple optimization: minimize expected loss
        # This is a simplified version - can be replaced with full LP optimization

        # Calculate expected returns for each asset
        expected_returns = {}
        for asset in self.asset_classes:
            expected_return = sum(
                beliefs[scenario] * scenario_forecasts[scenario][asset][period] / 100.0
                for scenario in beliefs.keys()
            )
            expected_returns[asset] = expected_return

        # Simple allocation based on expected returns (can be improved)
        # Higher expected return → Higher allocation
        total_expected = sum(max(0, ret) for ret in expected_returns.values())

        if total_expected > 0:
            portfolio = {asset: max(0, expected_returns[asset]) / total_expected
                        for asset in self.asset_classes}
        else:
            # Equal allocation if all expected returns are negative
            portfolio = {asset: 1.0/len(self.asset_classes) for asset in self.asset_classes}

        # Ensure weights sum to 1
        total_weight = sum(portfolio.values())
        if total_weight > 0:
            portfolio = {asset: weight/total_weight for asset, weight in portfolio.items()}

        return portfolio

    def _format_portfolio(self, portfolio: Dict[str, float]) -> str:
        """Format portfolio for display"""
        return ", ".join(f"{asset}: {weight*100:.1f}%" for asset, weight in portfolio.items())

    def _format_beliefs(self, beliefs: Dict[str, float]) -> str:
        """Format beliefs for display"""
        sorted_beliefs = sorted(beliefs.items(), key=lambda x: x[1], reverse=True)
        return ", ".join(f"{scenario}: {prob*100:.1f}%" for scenario, prob in sorted_beliefs[:3])


def test_similarity_learning_framework():
    """Test the similarity-based learning framework with example data"""

    print("🧪 TESTING SIMILARITY-BASED LEARNING FRAMEWORK")
    print("="*60)

    # Example initial beliefs (from 2017 training)
    initial_beliefs = {
        "Baseline V": 0.236,
        "Shallow V": 0.253,
        "U-Shaped": 0.270,
        "W-Shaped": 0.089,
        "Depression": 0.026,
        "Stagflation": 0.127
    }

    # Example scenario forecasts (simplified)
    scenario_forecasts = {
        "Baseline V": {"Cash": [0.2, -1.8, -2.5], "Stocks": [13.0, 6.8, -4.4], "Bonds": [1.8, -2.6, 2.5]},
        "Shallow V": {"Cash": [0.3, -1.7, -3.1], "Stocks": [15.5, 6.9, -0.6], "Bonds": [2.5, -2.0, 1.0]},
        "U-Shaped": {"Cash": [1.3, 0.4, -1.5], "Stocks": [-7.1, 11.3, 5.1], "Bonds": [4.0, 0.4, 3.1]},
        "W-Shaped": {"Cash": [1.7, 3.7, 4.0], "Stocks": [16.7, -3.8, -5.0], "Bonds": [-0.2, -0.4, 11.6]},
        "Depression": {"Cash": [3.4, 9.2, 10.1], "Stocks": [-23.1, -29.6, 1.3], "Bonds": [-1.3, 4.6, 17.7]},
        "Stagflation": {"Cash": [0.0, -3.3, -5.2], "Stocks": [3.2, 2.2, -1.0], "Bonds": [1.1, -6.2, -4.4]}
    }

    # Example observed data (2018-2020)
    observed_data = ObservedEconomicData(
        years=[2018, 2019, 2020],
        gdp_growth=[2.9, 2.2, -3.4],  # Actual GDP growth
        inflation=[2.4, 1.8, 1.2],    # Actual inflation
        asset_returns={
            "Cash": [2.4, 2.3, 0.4],
            "Stocks": [-4.4, 31.5, 18.4],
            "Bonds": [0.0, 8.7, 7.5]
        }
    )

    print(f"Initial beliefs: {initial_beliefs}")
    print(f"Observed GDP path: {observed_data.gdp_growth}")

    # Test different learning mechanisms
    learning_mechanisms = [
        SimilarityBasedLearner("no_learning"),
        SimilarityBasedLearner("bayesian", bandwidth=2.0),
        SimilarityBasedLearner("partial", learning_rate=0.3),
        SimilarityBasedLearner("perfect")
    ]

    optimizer = SequentialPortfolioOptimizer()
    results = {}

    for learner in learning_mechanisms:
        print(f"\n{'='*60}")
        print(f"TESTING: {learner.learning_type.upper()}")
        print(f"{'='*60}")

        result = optimizer.optimize_with_sequential_learning(
            initial_beliefs, scenario_forecasts, observed_data, learner
        )

        results[learner.learning_type] = result

    # Compare results
    print(f"\n{'COMPARISON SUMMARY':=^60}")
    print(f"{'Mechanism':<12} {'Cumulative Return':<18} {'Final Top Belief'}")
    print(f"{'-'*50}")

    for mechanism, result in results.items():
        cum_ret = result['cumulative_return'] * 100
        top_belief = max(result['final_beliefs'].items(), key=lambda x: x[1])
        print(f"{mechanism:<12} {cum_ret:>16.2f}% {top_belief[0]}: {top_belief[1]*100:.1f}%")

    return results


if __name__ == "__main__":
    test_results = test_similarity_learning_framework()

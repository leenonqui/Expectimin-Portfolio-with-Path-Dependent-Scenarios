"""
dynamic_game.py
Game Theory Framework for Portfolio Choice
Based on Rasmusen: Games and Information (2006)

Implements:
- Extensive form games with perfect recall
- Backward induction algorithm
- Subgame perfect equilibrium
- Learning mechanism comparison within single extensive form game
- Portfolio evolution tracking
- Belief trajectory analysis
"""

import numpy as np
import pulp as lp
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from constants import SCENARIOS, ASSET_CLASSES, HORIZON


@dataclass(frozen=True)
class InformationSet:
    """Rasmusen Chapter 2: Information sets for extensive form games"""
    time: int                          # Period (0, 1, 2)
    observable_history: Tuple          # History of (scenario, return) observations

    def __str__(self):
        return f"I_{self.time}({len(self.observable_history)} obs)"


@dataclass
class BeliefState:
    """Complete belief state including regime probabilities"""
    scenario_beliefs: Dict[str, float]      # P(scenario) direct beliefs
    regime_beliefs: Dict[str, float]        # P(regime) hidden state beliefs
    confidence: float                       # Confidence in current beliefs [0,1]

    def copy(self):
        return BeliefState(
            scenario_beliefs=self.scenario_beliefs.copy(),
            regime_beliefs=self.regime_beliefs.copy(),
            confidence=self.confidence
        )


class EconomicRegimes:
    """
    Hidden economic regime model for Bayesian learning
    Maps scenarios to underlying persistent regimes
    """

    def __init__(self):
        # Define economic regimes
        self.regimes = ["Recession", "Normal", "Expansion"]

        # Regime transition matrix P(regime_t+1 | regime_t)
        self.transition_matrix = np.array([
            #  Rec   Norm  Exp
            [0.7,  0.25, 0.05],  # From Recession
            [0.1,  0.80, 0.10],  # From Normal
            [0.05, 0.25, 0.70]   # From Expansion
        ])

        # Scenario emission probabilities P(scenario | regime)
        self.emission_matrix = {
            "Recession": {
                "Depression": 0.30, "Stagflation": 0.25, "W-Shaped": 0.20,
                "U-Shaped": 0.15, "Baseline V": 0.08, "Shallow V": 0.02
            },
            "Normal": {
                "Baseline V": 0.35, "U-Shaped": 0.30, "Shallow V": 0.15,
                "W-Shaped": 0.10, "Stagflation": 0.08, "Depression": 0.02
            },
            "Expansion": {
                "Shallow V": 0.40, "Baseline V": 0.25, "U-Shaped": 0.20,
                "W-Shaped": 0.10, "Stagflation": 0.03, "Depression": 0.02
            }
        }

        # Prior regime distribution
        self.prior_regime_beliefs = {"Recession": 0.15, "Normal": 0.70, "Expansion": 0.15}

    def get_likelihood(self, scenario: str, regime: str) -> float:
        """P(scenario | regime) - emission probability"""
        return self.emission_matrix[regime].get(scenario, 0.01)  # Small non-zero for robustness

    def get_transition_prob(self, regime_from: str, regime_to: str) -> float:
        """P(regime_t+1 | regime_t) - transition probability"""
        from_idx = self.regimes.index(regime_from)
        to_idx = self.regimes.index(regime_to)
        return self.transition_matrix[from_idx, to_idx]


class BayesianLearner:
    """
    Implements proper Bayesian belief updating using Bayes' theorem
    Rasmusen Chapter 8: Incomplete Information
    """

    def __init__(self, learning_type: str = "bayesian"):
        self.learning_type = learning_type
        self.regimes = EconomicRegimes()

    def update_beliefs(self, prior_beliefs: BeliefState,
                      observed_scenario: str) -> BeliefState:
        """
        Update beliefs using different learning mechanisms
        """

        if self.learning_type == "no_learning":
            return self._no_learning_update(prior_beliefs, observed_scenario)
        elif self.learning_type == "bayesian":
            return self._bayesian_update(prior_beliefs, observed_scenario)
        elif self.learning_type == "adaptive":
            return self._adaptive_update(prior_beliefs, observed_scenario)
        elif self.learning_type == "perfect":
            return self._perfect_learning_update(prior_beliefs, observed_scenario)
        else:
            return prior_beliefs.copy()

    def _no_learning_update(self, prior_beliefs: BeliefState,
                           observed_scenario: str) -> BeliefState:
        """No learning - beliefs never change (independence assumption)"""
        return prior_beliefs.copy()

    def _bayesian_update(self, prior_beliefs: BeliefState,
                        observed_scenario: str) -> BeliefState:
        """
        Proper Bayesian updating using Bayes' theorem
        P(regime | scenario) = P(scenario | regime) * P(regime) / P(scenario)
        """

        # Update regime beliefs using Bayes' theorem
        posterior_regime = {}
        evidence = 0.0

        # Calculate evidence: P(scenario) = Σ P(scenario | regime) * P(regime)
        for regime in self.regimes.regimes:
            likelihood = self.regimes.get_likelihood(observed_scenario, regime)
            prior_regime_prob = prior_beliefs.regime_beliefs[regime]
            evidence += likelihood * prior_regime_prob

        # Calculate posterior: P(regime | scenario)
        for regime in self.regimes.regimes:
            likelihood = self.regimes.get_likelihood(observed_scenario, regime)
            prior_regime_prob = prior_beliefs.regime_beliefs[regime]

            if evidence > 0:
                posterior_regime[regime] = (likelihood * prior_regime_prob) / evidence
            else:
                posterior_regime[regime] = prior_regime_prob

        # Update scenario beliefs based on regime beliefs
        scenario_beliefs = {}
        for scenario in SCENARIOS.keys():
            prob = 0.0
            for regime in self.regimes.regimes:
                emission_prob = self.regimes.get_likelihood(scenario, regime)
                regime_prob = posterior_regime[regime]
                prob += emission_prob * regime_prob
            scenario_beliefs[scenario] = prob

        # Normalize scenario beliefs
        total_scenario_prob = sum(scenario_beliefs.values())
        if total_scenario_prob > 0:
            scenario_beliefs = {k: v/total_scenario_prob for k, v in scenario_beliefs.items()}

        # Calculate confidence (entropy-based)
        regime_entropy = -sum(p * np.log2(p + 1e-10) for p in posterior_regime.values())
        max_entropy = np.log2(len(self.regimes.regimes))
        confidence = 1.0 - (regime_entropy / max_entropy) if max_entropy > 0 else 0.5

        return BeliefState(
            scenario_beliefs=scenario_beliefs,
            regime_beliefs=posterior_regime,
            confidence=confidence
        )

    def _adaptive_update(self, prior_beliefs: BeliefState,
                        observed_scenario: str, learning_rate: float = 0.3) -> BeliefState:
        """
        Adaptive learning with learning rate (behavioral approach)
        P_new(scenario) = λ * indicator + (1-λ) * P_old(scenario)
        """

        scenario_beliefs = {}
        for scenario in SCENARIOS.keys():
            indicator = 1.0 if scenario == observed_scenario else 0.0
            old_prob = prior_beliefs.scenario_beliefs[scenario]
            new_prob = learning_rate * indicator + (1 - learning_rate) * old_prob
            scenario_beliefs[scenario] = new_prob

        # Normalize
        total = sum(scenario_beliefs.values())
        if total > 0:
            scenario_beliefs = {k: v/total for k, v in scenario_beliefs.items()}

        # Update regime beliefs (simplified)
        regime_beliefs = prior_beliefs.regime_beliefs.copy()

        # Simple confidence update
        confidence = min(0.9, prior_beliefs.confidence + 0.1)

        return BeliefState(
            scenario_beliefs=scenario_beliefs,
            regime_beliefs=regime_beliefs,
            confidence=confidence
        )

    def _perfect_learning_update(self, prior_beliefs: BeliefState,
                                observed_scenario: str) -> BeliefState:
        """Perfect learning - full confidence in observed scenario continuing"""

        scenario_beliefs = {scenario: 1.0 if scenario == observed_scenario else 0.0
                           for scenario in SCENARIOS.keys()}

        # Update regime beliefs to reflect perfect confidence
        regime_beliefs = {}
        max_likelihood = 0
        best_regime = "Normal"

        for regime in self.regimes.regimes:
            likelihood = self.regimes.get_likelihood(observed_scenario, regime)
            if likelihood > max_likelihood:
                max_likelihood = likelihood
                best_regime = regime

        regime_beliefs = {regime: 1.0 if regime == best_regime else 0.0
                         for regime in self.regimes.regimes}

        return BeliefState(
            scenario_beliefs=scenario_beliefs,
            regime_beliefs=regime_beliefs,
            confidence=1.0
        )


class SinglePeriodOptimizer:
    """Solves single-period expectimin problem using LP"""

    def __init__(self, asset_classes: List[str]):
        self.asset_classes = asset_classes

    def solve_expectimin(self, beliefs: BeliefState,
                        period_returns: Dict[str, Dict[str, float]],
                        min_return: Optional[float] = None) -> Dict:
        """Solve single-period expectimin LP"""

        prob = lp.LpProblem("Expectimin_SinglePeriod", lp.LpMinimize)

        scenarios = list(beliefs.scenario_beliefs.keys())

        # Decision variables
        weights = {asset: lp.LpVariable(f"w_{asset}", 0, 1)
                  for asset in self.asset_classes}
        losses = {scenario: lp.LpVariable(f"loss_{scenario}", 0)
                 for scenario in scenarios}

        # Objective: minimize expected loss
        prob += lp.lpSum([beliefs.scenario_beliefs[s] * losses[s] for s in scenarios])

        # Constraint: weights sum to 1
        prob += lp.lpSum(weights.values()) == 1

        # Loss definition constraints
        for scenario in scenarios:
            portfolio_return = lp.lpSum([
                weights[asset] * period_returns[scenario][asset] / 100.0
                for asset in self.asset_classes
            ])
            prob += losses[scenario] >= -portfolio_return

        # Optional minimum return constraint
        if min_return is not None:
            expected_return = lp.lpSum([
                beliefs.scenario_beliefs[scenario] * lp.lpSum([
                    weights[asset] * period_returns[scenario][asset] / 100.0
                    for asset in self.asset_classes
                ])
                for scenario in scenarios
            ])
            prob += expected_return >= min_return

        # Solve
        prob.solve(lp.PULP_CBC_CMD(msg=0))

        if prob.status == lp.LpStatusOptimal:
            optimal_weights = {asset: weights[asset].varValue
                             for asset in self.asset_classes}
            expected_loss = sum(beliefs.scenario_beliefs[s] * losses[s].varValue
                              for s in scenarios)

            expected_return = sum(
                beliefs.scenario_beliefs[scenario] * sum(
                    optimal_weights[asset] * period_returns[scenario][asset] / 100.0
                    for asset in self.asset_classes
                ) for scenario in scenarios
            )

            return {
                'success': True,
                'weights': optimal_weights,
                'expected_loss': expected_loss,
                'expected_return': expected_return,
                'status': 'optimal'
            }
        else:
            return {
                'success': False,
                'status': lp.LpStatus[prob.status]
            }


class LearningMechanismAnalyzer:
    """
    Main class for analyzing different learning mechanisms
    Compares learning approaches within the same extensive form game
    """

    def __init__(self, scenario_forecasts: Dict[str, Dict[str, List[float]]],
                 initial_probabilities: Dict[str, float]):
        self.scenario_forecasts = scenario_forecasts
        self.initial_probabilities = initial_probabilities
        self.optimizer = SinglePeriodOptimizer(ASSET_CLASSES)
        self.regimes = EconomicRegimes()

    def analyze_learning_mechanisms(self,
                                  learning_types: List[str] = None,
                                  scenario_path: List[str] = None) -> Dict:
        """
        Compare different learning mechanisms along a specific scenario path

        Args:
            learning_types: List of learning mechanisms to compare
            scenario_path: Specific sequence of scenarios to analyze
        """

        if learning_types is None:
            learning_types = ["no_learning", "bayesian", "adaptive", "perfect"]

        if scenario_path is None:
            # Use most likely scenario path
            most_likely = max(self.initial_probabilities.items(), key=lambda x: x[1])[0]
            scenario_path = [most_likely, most_likely, most_likely]

        print("🧠 LEARNING MECHANISM ANALYSIS")
        print("Extensive Form Game with Perfect Recall")
        print("="*60)
        print(f"Scenario path: {' → '.join(scenario_path)}")
        print(f"Learning mechanisms: {learning_types}")

        results = {}

        for learning_type in learning_types:
            print(f"\n📊 Analyzing: {learning_type.upper()}")
            result = self._analyze_single_learning_mechanism(learning_type, scenario_path)
            results[learning_type] = result

        # Compare results
        comparison = self._compare_learning_mechanisms(results)

        return {
            'learning_results': results,
            'comparison': comparison,
            'scenario_path': scenario_path,
            'parameters': {
                'learning_types': learning_types,
                'initial_probabilities': self.initial_probabilities
            }
        }

    def _analyze_single_learning_mechanism(self, learning_type: str,
                                         scenario_path: List[str]) -> Dict:
        """Analyze single learning mechanism along scenario path"""

        learner = BayesianLearner(learning_type)

        # Initialize beliefs
        initial_beliefs = BeliefState(
            scenario_beliefs=self.initial_probabilities.copy(),
            regime_beliefs=self.regimes.prior_regime_beliefs.copy(),
            confidence=0.5
        )

        # Track evolution
        belief_evolution = [initial_beliefs]
        portfolio_evolution = []
        period_results = []
        cumulative_return = 0.0
        cumulative_loss = 0.0

        current_beliefs = initial_beliefs

        for period in range(HORIZON):
            # Extract period returns
            period_returns = {}
            for scenario in self.scenario_forecasts:
                period_returns[scenario] = {}
                for asset in ASSET_CLASSES:
                    period_returns[scenario][asset] = self.scenario_forecasts[scenario][asset][period]

            # Optimize portfolio for this period
            optimization_result = self.optimizer.solve_expectimin(
                beliefs=current_beliefs,
                period_returns=period_returns
            )

            if optimization_result['success']:
                portfolio = optimization_result['weights']
                portfolio_evolution.append(portfolio)

                # Calculate realized return for this period
                realized_scenario = scenario_path[period]
                realized_return = sum(
                    portfolio[asset] * period_returns[realized_scenario][asset] / 100.0
                    for asset in ASSET_CLASSES
                )

                # Update cumulative metrics
                cumulative_return += realized_return
                period_loss = max(0, -realized_return)
                cumulative_loss += period_loss

                period_results.append({
                    'period': period,
                    'beliefs': current_beliefs.scenario_beliefs.copy(),
                    'regime_beliefs': current_beliefs.regime_beliefs.copy(),
                    'confidence': current_beliefs.confidence,
                    'portfolio': portfolio.copy(),
                    'realized_scenario': realized_scenario,
                    'realized_return': realized_return,
                    'period_loss': period_loss
                })

                # Update beliefs based on observed scenario
                if period < HORIZON - 1:  # Don't update after last period
                    current_beliefs = learner.update_beliefs(current_beliefs, realized_scenario)
                    belief_evolution.append(current_beliefs)

            else:
                print(f"  ⚠️  Optimization failed for period {period}")
                break

        print(f"  ✓ Final cumulative return: {cumulative_return*100:.2f}%")
        print(f"  ✓ Final cumulative loss: {cumulative_loss*100:.2f}%")

        return {
            'learning_type': learning_type,
            'belief_evolution': belief_evolution,
            'portfolio_evolution': portfolio_evolution,
            'period_results': period_results,
            'cumulative_return': cumulative_return,
            'cumulative_loss': cumulative_loss,
            'final_beliefs': current_beliefs
        }

    def _compare_learning_mechanisms(self, results: Dict[str, Dict]) -> Dict:
        """Compare performance across learning mechanisms"""

        print(f"\n📈 LEARNING MECHANISM COMPARISON")
        print(f"{'Type':<12} {'Cum Return':<12} {'Cum Loss':<10} {'Final Confidence':<16}")
        print(f"{'-'*55}")

        comparison_data = {}
        best_mechanism = None
        best_performance = float('-inf')

        for learning_type, result in results.items():
            cum_return = result['cumulative_return'] * 100
            cum_loss = result['cumulative_loss'] * 100
            confidence = result['final_beliefs'].confidence

            print(f"{learning_type:<12} {cum_return:>10.2f}% {cum_loss:>8.2f}% {confidence:>14.2f}")

            comparison_data[learning_type] = {
                'cumulative_return_pct': cum_return,
                'cumulative_loss_pct': cum_loss,
                'final_confidence': confidence,
                'performance_score': cum_return - cum_loss  # Simple performance metric
            }

            if comparison_data[learning_type]['performance_score'] > best_performance:
                best_performance = comparison_data[learning_type]['performance_score']
                best_mechanism = learning_type

        print(f"{'-'*55}")
        print(f"Best mechanism: {best_mechanism}")

        return {
            'detailed_comparison': comparison_data,
            'best_mechanism': best_mechanism,
            'best_performance': best_performance
        }

    def analyze_belief_sensitivity(self, learning_type: str = "bayesian",
                                 scenario_paths: List[List[str]] = None) -> Dict:
        """
        Analyze sensitivity of beliefs and portfolios to different scenario sequences
        """

        if scenario_paths is None:
            # Generate representative scenario paths
            scenarios = list(self.initial_probabilities.keys())
            scenario_paths = [
                [scenarios[0], scenarios[0], scenarios[0]],  # Persistent
                [scenarios[0], scenarios[1], scenarios[2]],  # Mixed
                [scenarios[-1], scenarios[-1], scenarios[-1]]  # Different persistent
            ]

        print(f"\n🔍 BELIEF SENSITIVITY ANALYSIS")
        print(f"Learning mechanism: {learning_type}")
        print("="*50)

        sensitivity_results = {}

        for i, path in enumerate(scenario_paths):
            print(f"\nPath {i+1}: {' → '.join(path)}")
            result = self._analyze_single_learning_mechanism(learning_type, path)
            sensitivity_results[f"path_{i+1}"] = {
                'scenario_path': path,
                'result': result
            }

        return sensitivity_results

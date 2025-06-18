import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Optional
from ..models.portfolio import OptimizationResult
from ..config import RISK_AVERSION_PROFILES, ASSET_CLASSES

class ExpectiminimaxOptimizer:
    """
    Expectiminimax Portfolio Optimizer for Path-Dependent Scenarios

    Implements expectiminimax criterion:
    - Maximizes expected utility across scenarios
    - Uses mean-variance utility function with risk aversion parameter
    - Supports multiple risk aversion profiles
    """

    def __init__(self,
                 scenario_probabilities: Dict[str, float],
                 asset_returns: Dict[str, Dict[str, List[float]]],
                 min_weight: float = 0.0,
                 max_weight: float = 1.0):

        self.scenario_probabilities = scenario_probabilities
        self.asset_returns = asset_returns
        self.min_weight = min_weight
        self.max_weight = max_weight

        # Validate inputs
        self._validate_inputs()

        # Setup optimization parameters
        self.n_assets = len(ASSET_CLASSES)
        self.n_years = 3  # 3-year scenarios

    def optimize_single_profile(self, risk_aversion: float, utility_type: str = "crra") -> OptimizationResult:
        """Optimize portfolio for single risk aversion profile"""

        initial_weights = np.ones(self.n_assets) / self.n_assets
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(self.min_weight, self.max_weight) for _ in range(self.n_assets)]

        # Objective function: maximize expected utility
        def objective(weights):
            return -self._expectiminimax_value(weights, risk_aversion, utility_type)

        result = minimize(
            objective, initial_weights, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'ftol': 1e-12, 'disp': False}
        )

        optimal_weights = result.x
        optimal_value = -result.fun

        # Calculate portfolio metrics
        expected_return, expected_volatility = self._calculate_portfolio_metrics(optimal_weights)

        return OptimizationResult(
            risk_profile="Custom",
            risk_aversion=risk_aversion,
            optimal_weights=dict(zip(ASSET_CLASSES, optimal_weights)),
            expectiminimax_value=optimal_value,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            optimization_success=result.success
        )

    def optimize_all_profiles(self) -> Dict[str, OptimizationResult]:
        """Optimize portfolios for all predefined risk aversion profiles"""

        results = {}

        for profile in RISK_AVERSION_PROFILES:
            name = profile["name"]
            risk_aversion = profile["risk_aversion"]

            result = self.optimize_single_profile(risk_aversion)
            result.risk_profile = name
            results[name] = result

        return results

    def _expectiminimax_value(self, weights: np.ndarray, risk_aversion: float, utility_type: str = "crra") -> float:
        """Calculate expectiminimax value as probability-weighted sum of scenario utilities"""

        expected_utility = 0.0

        # Calculate utility for each scenario and weight by probability
        for scenario_name, probability in self.scenario_probabilities.items():
            if scenario_name in self.asset_returns:
                # Get portfolio returns path for this scenario
                portfolio_returns = self._portfolio_returns_path(weights, scenario_name)

                # Calculate utility for this specific scenario
                scenario_utility = self._calculate_scenario_utility(
                    portfolio_returns,
                    risk_aversion,
                    utility_type
                )

                # Weight by scenario probability
                expected_utility += probability * scenario_utility

        return expected_utility

    def _calculate_scenario_utility(self, portfolio_returns: np.ndarray, risk_aversion: float, utility_type: str) -> float:
        """Calculate utility for a single scenario"""

        if utility_type == "crra":
            return self._crra_utility(portfolio_returns, risk_aversion)
        elif utility_type == "mean_variance":
            return self._mean_variance_utility(portfolio_returns, risk_aversion)
        else:
            raise ValueError(f"Unknown utility type: {utility_type}")

    def _crra_utility(self, portfolio_returns: np.ndarray, gamma: float) -> float:
        """

        U = (W_final)^(1-γ) / (1-γ)  for γ ≠ 1
        U = ln(W_final)              for γ = 1

        Where W_final = initial_wealth * (1+R1) * (1+R2) * (1+R3)
        """

        # Convert percentage returns to decimals
        returns_decimal = portfolio_returns / 100

        # Calculate final wealth (assuming initial wealth = 1)
        final_wealth = 1.0
        for r in returns_decimal:
            final_wealth *= (1 + r)

        # Handle negative wealth (should never happen with reasonable portfolios)
        if final_wealth <= 0:
            return -np.inf

        # Calculate CRRA utility
        if abs(gamma - 1.0) < 1e-8:  # γ ≈ 1 (log utility)
            utility = np.log(final_wealth)
        else:  # γ ≠ 1
            utility = (final_wealth**(1 - gamma)) / (1 - gamma)

        return utility

    def _mean_variance_utility(self, portfolio_returns: np.ndarray, gamma: float) -> float:
        """Mean-variance utility function"""

        if gamma == 0.0:
            # Risk neutral: sum of returns
            return np.sum(portfolio_returns)

        mean_return = np.mean(portfolio_returns)
        variance = np.var(portfolio_returns)

        # Scale to match the time horizon (3 years)
        return mean_return - (gamma / 2) * variance

    def _portfolio_returns_path(self, weights: np.ndarray, scenario_name: str) -> np.ndarray:
        """Calculate portfolio returns path for given scenario and weights"""

        scenario_returns = self.asset_returns[scenario_name]
        portfolio_returns = []

        for year in range(self.n_years):
            year_return = sum(
                weights[i] * scenario_returns[ASSET_CLASSES[i]][year]
                for i in range(self.n_assets)
            )
            portfolio_returns.append(year_return)

        return np.array(portfolio_returns)

    def _calculate_portfolio_metrics(self, weights: np.ndarray) -> tuple:
        """Calculate expected return and volatility with proper mathematical units"""

        # Calculate cumulative returns
        cumulative_returns_decimal = []
        probabilities = []

        for scenario_name, probability in self.scenario_probabilities.items():
            if scenario_name in self.asset_returns:
                portfolio_returns = self._portfolio_returns_path(weights, scenario_name)

                # Calculate 3-year cumulative return
                cumulative_decimal = np.prod([1 + r/100 for r in portfolio_returns]) - 1

                cumulative_returns_decimal.append(cumulative_decimal)
                probabilities.append(probability)

        cumulative_returns_decimal = np.array(cumulative_returns_decimal)
        probabilities = np.array(probabilities)

        # Expected cumulative return
        expected_return_decimal = np.sum(probabilities * cumulative_returns_decimal)

        # Variance
        variance_decimal = np.sum(probabilities * (cumulative_returns_decimal - expected_return_decimal)**2)

        # Standard deviation (volatility) in decimal units
        volatility_decimal = np.sqrt(variance_decimal)

        # Convert to percentages for output
        expected_return_pct = expected_return_decimal * 100
        volatility_pct = volatility_decimal * 100

        return expected_return_pct, volatility_pct

    def get_detailed_scenario_analysis(self, weights: np.ndarray) -> Dict:
        """Get detailed analysis showing cumulative returns by scenario"""

        analysis = {
            'scenario_cumulative_returns': {},
            'scenario_annual_returns': {},
            'probability_weighted_stats': {}
        }

        cumulative_returns = []
        probabilities = []

        for scenario_name, probability in self.scenario_probabilities.items():
            if scenario_name in self.asset_returns:
                portfolio_returns = self._portfolio_returns_path(weights, scenario_name)

                # Annual returns
                analysis['scenario_annual_returns'][scenario_name] = portfolio_returns.tolist()

                # Cumulative return
                cumulative = (np.prod([1 + r/100 for r in portfolio_returns]) - 1) * 100
                analysis['scenario_cumulative_returns'][scenario_name] = cumulative

                cumulative_returns.append(cumulative)
                probabilities.append(probability)

        # Summary statistics
        cumulative_returns = np.array(cumulative_returns)
        probabilities = np.array(probabilities)

        analysis['probability_weighted_stats'] = {
            'expected_cumulative_return': np.sum(probabilities * cumulative_returns),
            'volatility': np.sqrt(np.sum(probabilities * (cumulative_returns - np.sum(probabilities * cumulative_returns))**2)),
            'min_cumulative': np.min(cumulative_returns),
            'max_cumulative': np.max(cumulative_returns)
        }

        return analysis

    def _validate_inputs(self):
        """Validate input data consistency"""

        # Check that all scenarios have asset return data
        for scenario in self.scenario_probabilities.keys():
            if scenario not in self.asset_returns:
                raise ValueError(f"Missing asset returns for scenario: {scenario}")

        # Check that all asset returns have correct structure
        for scenario_name, returns in self.asset_returns.items():
            for asset in ASSET_CLASSES:
                if asset not in returns:
                    raise ValueError(f"Missing {asset} returns for scenario {scenario_name}")
                if len(returns[asset]) != 3:
                    raise ValueError(f"Expected 3 years of returns for {asset} in {scenario_name}")

        # Check that probabilities sum to approximately 1
        total_prob = sum(self.scenario_probabilities.values())
        if abs(total_prob - 1.0) > 1e-6:
            raise ValueError(f"Scenario probabilities sum to {total_prob}, not 1.0")

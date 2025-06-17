import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Optional
from ..models.portfolio import OptimizationResult
from ..config import RISK_AVERSION_PROFILES, ASSET_CLASSES, LIQUIDITY_PREFERENCE_PROFILES, COMBINED_PROFILES

class ExpectiminimaxOptimizer:
    """
    Enhanced Expectiminimax Portfolio Optimizer with Liquidity Preferences

    Implements expectiminimax criterion with liquidity constraints:
    - Maximizes expected utility across scenarios
    - Uses mean-variance utility function with risk aversion parameter
    - Supports minimum cash allocation (liquidity preference)
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

    def optimize_single_profile(self,
                               risk_aversion: float,
                               min_cash_pct: float = 0.0) -> OptimizationResult:
        """
        Optimize portfolio for single risk aversion profile with liquidity preference

        Args:
            risk_aversion: Risk aversion parameter (λ)
            min_cash_pct: Minimum cash allocation as percentage (0.0-0.10)
        """

        # Initial weights (equal allocation, respecting cash minimum)
        if min_cash_pct > 0:
            remaining_weight = 1.0 - min_cash_pct
            initial_weights = np.array([min_cash_pct, remaining_weight/2, remaining_weight/2])
        else:
            initial_weights = np.ones(self.n_assets) / self.n_assets

        # Constraints: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

        # Enhanced bounds with liquidity preference
        bounds = self._create_bounds_with_liquidity(min_cash_pct)

        # Objective function (minimize negative expectiminimax value)
        def objective(weights):
            return -self._expectiminimax_value(weights, risk_aversion)

        # Optimize
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
            min_cash_pct=min_cash_pct,
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

            result = self.optimize_single_profile(risk_aversion, min_cash_pct=0.0)
            result.risk_profile = name
            results[name] = result

        return results

    def optimize_liquidity_profiles(self) -> Dict[str, OptimizationResult]:
        """Optimize portfolios for all liquidity preference profiles (with moderate risk aversion)"""

        results = {}
        moderate_risk_aversion = 1.0  # Use moderate risk aversion as baseline

        for profile in LIQUIDITY_PREFERENCE_PROFILES:
            name = profile["name"]
            min_cash_pct = profile["min_cash_pct"]

            result = self.optimize_single_profile(moderate_risk_aversion, min_cash_pct)
            result.risk_profile = name
            results[name] = result

        return results

    def optimize_combined_profiles(self) -> Dict[str, OptimizationResult]:
        """Optimize portfolios for combined risk-liquidity profiles"""

        results = {}

        for profile in COMBINED_PROFILES:
            name = profile["name"]
            risk_aversion = profile["risk_aversion"]
            min_cash_pct = profile["min_cash_pct"]

            result = self.optimize_single_profile(risk_aversion, min_cash_pct)
            result.risk_profile = name
            results[name] = result

        return results

    def optimize_liquidity_sensitivity(self,
                                     risk_aversion: float = 1.0,
                                     liquidity_range: List[float] = None) -> Dict[str, OptimizationResult]:
        """
        Analyze sensitivity to liquidity preferences for given risk aversion

        Args:
            risk_aversion: Fixed risk aversion parameter
            liquidity_range: List of minimum cash percentages to test
        """

        if liquidity_range is None:
            liquidity_range = [0.0, 0.02, 0.05, 0.08, 0.10]  # 0% to 10%

        results = {}

        for min_cash_pct in liquidity_range:
            name = f"Liquidity_{min_cash_pct*100:.0f}%"

            result = self.optimize_single_profile(risk_aversion, min_cash_pct)
            result.risk_profile = name
            results[name] = result

        return results

    def _create_bounds_with_liquidity(self, min_cash_pct: float) -> List[tuple]:
        """Create bounds with liquidity preference constraint"""

        bounds = []

        for i, asset in enumerate(ASSET_CLASSES):
            if asset == "Cash":
                # Cash has minimum allocation based on liquidity preference
                bounds.append((min_cash_pct, self.max_weight))
            else:
                # Other assets have standard bounds
                bounds.append((self.min_weight, self.max_weight))

        return bounds

    def _expectiminimax_value(self, weights: np.ndarray, risk_aversion: float) -> float:
        """Calculate expectiminimax value for given weights and risk aversion"""

        expected_utility = 0.0

        for scenario_name, probability in self.scenario_probabilities.items():
            if scenario_name in self.asset_returns:
                scenario_utility = self._scenario_utility(weights, scenario_name, risk_aversion)
                expected_utility += probability * scenario_utility

        return expected_utility

    def _scenario_utility(self, weights: np.ndarray, scenario_name: str, risk_aversion: float) -> float:
        """Calculate utility for specific scenario"""

        # Get portfolio returns for this scenario
        portfolio_returns = self._portfolio_returns_path(weights, scenario_name)

        # Calculate mean-variance utility
        mean_return = np.mean(portfolio_returns)

        if risk_aversion == 0.0:
            # Risk neutral case
            return mean_return
        else:
            # Risk averse case
            variance = np.var(portfolio_returns)
            return mean_return - (risk_aversion / 2) * variance

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
        """Calculate expected return and volatility across all scenarios"""

        expected_return = 0.0
        expected_variance = 0.0

        for scenario_name, probability in self.scenario_probabilities.items():
            if scenario_name in self.asset_returns:
                portfolio_returns = self._portfolio_returns_path(weights, scenario_name)

                scenario_mean = np.mean(portfolio_returns)
                scenario_variance = np.var(portfolio_returns)

                expected_return += probability * scenario_mean
                expected_variance += probability * scenario_variance

        expected_volatility = np.sqrt(expected_variance)

        return expected_return, expected_volatility

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

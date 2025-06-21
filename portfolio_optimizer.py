"""
Simplified Portfolio Optimization Module
Implements Section 3.2 of the thesis methodology
"""

import numpy as np
from scipy.optimize import linprog
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from constants import (
    ASSET_CLASSES, HORIZON, DEFAULT_CONFIDENCE_LEVEL,
    MIN_WEIGHT, MAX_WEIGHT, OPTIMIZATION_TOLERANCE
)
from utils import (
    simple_to_log_return, log_to_simple_return,
    calculate_cumulative_return, portfolio_performance_summary
)

@dataclass
class OptimizationResult:
    """Results from CVaR optimization"""
    weights: Dict[str, float]  # Asset weights w_0
    cvar: float               # Optimal CVaR value
    var: float                # VaR (eta) value
    expected_return: float    # Expected cumulative return
    success: bool            # Optimization success flag
    
class CVaROptimizer:
    """
    Implements Expectiminimax portfolio optimization using CVaR
    Following Section 3.2 of the thesis
    """
    
    def __init__(self, 
                 probabilities: Dict[str, float],
                 returns: Dict[str, Dict[str, List[float]]],
                 horizon: int = HORIZON,
                 confidence_level: float = DEFAULT_CONFIDENCE_LEVEL):
        """
        Args:
            probabilities: P_s for each scenario
            returns: Forecasted returns {scenario: {asset: [y1, y2, y3]}}
            horizon: Investment horizon T
            confidence_level: Confidence level for CVaR (e.g., 0.95)
        """
        self.probabilities = probabilities
        self.returns = returns
        self.horizon = horizon
        self.alpha = 1 - confidence_level  # For CVaR calculation
        
        # Extract scenario and asset names
        self.scenarios = list(probabilities.keys())
        self.assets = ASSET_CLASSES  # Use from constants
        
        # Convert to log returns (Section 3.2.3)
        self.log_returns = self._convert_to_log_returns()
        
    def _convert_to_log_returns(self) -> Dict[str, Dict[str, List[float]]]:
        """Convert simple returns to log returns: r = ln(1 + R)"""
        log_returns = {}
        
        for scenario, asset_returns in self.returns.items():
            log_returns[scenario] = {}
            for asset, yearly_returns in asset_returns.items():
                # Convert each year's simple return to log return
                log_returns[scenario][asset] = [
                    simple_to_log_return(R) for R in yearly_returns
                ]
        
        return log_returns
    
    def optimize(self, min_return: Optional[float] = None) -> OptimizationResult:
        """
        Minimize CVaR using linear programming
        Implements the optimization problem from Section 3.2.4
        
        Args:
            min_return: Minimum acceptable cumulative return (e.g., risk-free rate)
            
        Returns:
            OptimizationResult with optimal portfolio
        """
        n_assets = len(self.assets)
        n_scenarios = len(self.scenarios)
        
        # Decision variables: [w_0 (n_assets), eta (1), z_s (n_scenarios)]
        n_vars = n_assets + 1 + n_scenarios
        
        # Objective: minimize eta + 1/((1-alpha)*N_S) * sum(P_s * z_s)
        c = np.zeros(n_vars)
        c[n_assets] = 1  # Coefficient for eta
        
        # Coefficients for z_s
        for i, scenario in enumerate(self.scenarios):
            c[n_assets + 1 + i] = self.probabilities[scenario] / ((1 - self.alpha) * n_scenarios)
        
        # Inequality constraints: -r_cumul(w,s) - eta <= z_s
        A_ub = []
        b_ub = []
        
        for i, scenario in enumerate(self.scenarios):
            # Calculate cumulative log return coefficients for this scenario
            return_coeffs = self._get_scenario_return_coefficients(scenario)
            
            # Constraint: -r_cumul(w,s) - eta - z_s <= 0
            constraint = np.zeros(n_vars)
            constraint[:n_assets] = -return_coeffs  # -r_cumul(w,s)
            constraint[n_assets] = -1  # -eta
            constraint[n_assets + 1 + i] = -1  # -z_s
            
            A_ub.append(constraint)
            b_ub.append(0)
        
        # Equality constraint: sum(w_i) = 1
        A_eq = [np.zeros(n_vars)]
        A_eq[0][:n_assets] = 1
        b_eq = [1]
        
        # If minimum return specified, add constraint
        if min_return is not None:
            # Convert cumulative simple return to log return
            min_log_return = simple_to_log_return(min_return)
            
            # Expected return constraint: sum(P_s * r_cumul(w,s)) >= min_log_return
            # Rewrite as: -sum(P_s * r_cumul(w,s)) <= -min_log_return
            constraint = np.zeros(n_vars)
            
            for scenario in self.scenarios:
                return_coeffs = self._get_scenario_return_coefficients(scenario)
                prob = self.probabilities[scenario]
                constraint[:n_assets] -= prob * return_coeffs
            
            A_ub.append(constraint)
            b_ub.append(-min_log_return)
        
        # Variable bounds
        bounds = []
        # w_i >= 0 (no short selling)
        for i in range(n_assets):
            bounds.append((MIN_WEIGHT, MAX_WEIGHT))
        # eta unbounded
        bounds.append((None, None))
        # z_s >= 0
        for i in range(n_scenarios):
            bounds.append((0, None))
        
        # Solve linear program
        result = linprog(
            c=c,
            A_ub=A_ub if A_ub else None,
            b_ub=b_ub if b_ub else None,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs',
            options={'presolve': True, 'disp': False}
        )
        
        if result.success:
            # Extract results
            weights = result.x[:n_assets]
            eta = result.x[n_assets]
            z_values = result.x[n_assets + 1:]
            
            # Calculate expected return
            expected_log_return = self._calculate_expected_return(weights)
            expected_simple_return = log_to_simple_return(expected_log_return) * 100
            
            # Convert log CVaR to simple terms
            cvar_log = result.fun
            cvar_simple = -log_to_simple_return(-cvar_log) * 100  # Negative because CVaR is a loss
            
            return OptimizationResult(
                weights={asset: w for asset, w in zip(self.assets, weights)},
                cvar=cvar_simple,
                var=-log_to_simple_return(-eta) * 100,  # Convert to simple return
                expected_return=expected_simple_return,
                success=True
            )
        else:
            # Return equal weights if optimization fails
            equal_weight = 1.0 / n_assets
            weights = {asset: equal_weight for asset in self.assets}
            
            return OptimizationResult(
                weights=weights,
                cvar=np.nan,
                var=np.nan,
                expected_return=np.nan,
                success=False
            )
    
    def _get_scenario_return_coefficients(self, scenario: str) -> np.ndarray:
        """
        Get coefficients for cumulative log return calculation
        r_cumul(w,s) = sum_t (w' * r_t,s)
        """
        coeffs = np.zeros(len(self.assets))
        
        for i, asset in enumerate(self.assets):
            # Sum log returns across all years for this asset
            cumul_log_return = sum(self.log_returns[scenario][asset])
            coeffs[i] = cumul_log_return
        
        return coeffs
    
    def _calculate_expected_return(self, weights: np.ndarray) -> float:
        """Calculate expected cumulative log return across scenarios"""
        expected_return = 0
        
        for scenario in self.scenarios:
            prob = self.probabilities[scenario]
            return_coeffs = self._get_scenario_return_coefficients(scenario)
            scenario_return = weights @ return_coeffs
            expected_return += prob * scenario_return
        
        return expected_return
    
    def analyze_portfolio(self, weights: Dict[str, float]) -> Dict[str, any]:
        """
        Analyze portfolio performance across scenarios
        
        Args:
            weights: Portfolio weights to analyze
            
        Returns:
            Dict with scenario returns and risk metrics
        """
        weights_array = np.array([weights[asset] for asset in self.assets])
        
        # Calculate scenario returns
        scenario_returns = {}
        for scenario in self.scenarios:
            # Calculate cumulative simple return for this scenario
            annual_simple_returns = []
            
            for year in range(self.horizon):
                # Calculate portfolio return for this year
                portfolio_return = 0
                for i, asset in enumerate(self.assets):
                    portfolio_return += weights_array[i] * self.returns[scenario][asset][year]
                annual_simple_returns.append(portfolio_return)
            
            # Calculate cumulative return
            cumul_return = calculate_cumulative_return(annual_simple_returns, as_log=False)
            scenario_returns[scenario] = cumul_return
        
        # Use utility function for comprehensive metrics
        metrics = portfolio_performance_summary(
            scenario_returns, 
            self.probabilities,
            confidence_level=(1 - self.alpha)
        )
        
        # Add scenario-specific returns to metrics
        metrics['scenario_returns'] = scenario_returns
        
        return metrics
    
    def efficient_frontier(self, n_points: int = 20) -> List[Tuple[float, float, Dict[str, float]]]:
        """
        Generate efficient frontier by varying minimum return constraint
        
        Args:
            n_points: Number of points on the frontier
            
        Returns:
            List of (expected_return, cvar, weights) tuples
        """
        frontier = []
        
        # First, find unconstrained solution (minimum CVaR)
        unconstrained = self.optimize()
        if not unconstrained.success:
            return frontier
        
        min_return = unconstrained.expected_return
        
        # Find maximum possible return (usually all stocks)
        max_weights = {asset: 0.0 for asset in self.assets}
        max_weights['Stocks'] = 1.0  # Assume stocks have highest expected return
        max_metrics = self.analyze_portfolio(max_weights)
        max_return = max_metrics['expected_return']
        
        # Generate points along the frontier
        for target_return in np.linspace(min_return, max_return, n_points):
            result = self.optimize(min_return=target_return)
            if result.success:
                frontier.append((
                    result.expected_return,
                    result.cvar,
                    result.weights
                ))
        
        return frontier

"""
Helper functions for liquidity preference analysis
"""

from typing import Dict, List, Tuple
import numpy as np
from ..models.portfolio import OptimizationResult

class LiquidityAnalyzer:
    """Helper class for analyzing liquidity preference impacts"""

    @staticmethod
    def calculate_liquidity_cost(results: Dict[str, OptimizationResult],
                                base_liquidity: float = 0.0) -> Dict[str, Dict[str, float]]:
        """
        Calculate the cost of liquidity preferences relative to a baseline

        Args:
            results: Dictionary of optimization results with different liquidity levels
            base_liquidity: Baseline liquidity level for comparison

        Returns:
            Dictionary with cost metrics for each liquidity level
        """

        # Find baseline result
        baseline_key = None
        for key, result in results.items():
            if abs(result.min_cash_pct - base_liquidity) < 1e-6:
                baseline_key = key
                break

        if baseline_key is None:
            raise ValueError(f"No result found with liquidity level {base_liquidity}")

        baseline = results[baseline_key]
        costs = {}

        for key, result in results.items():
            if key == baseline_key:
                costs[key] = {
                    'utility_cost': 0.0,
                    'return_cost': 0.0,
                    'volatility_change': 0.0,
                    'efficiency_change': 0.0
                }
            else:
                utility_cost = baseline.expectiminimax_value - result.expectiminimax_value
                return_cost = baseline.expected_return - result.expected_return
                volatility_change = result.expected_volatility - baseline.expected_volatility

                # Calculate efficiency (return per unit of risk)
                baseline_efficiency = baseline.expected_return / baseline.expected_volatility
                result_efficiency = result.expected_return / result.expected_volatility
                efficiency_change = result_efficiency - baseline_efficiency

                costs[key] = {
                    'utility_cost': utility_cost,
                    'return_cost': return_cost,
                    'volatility_change': volatility_change,
                    'efficiency_change': efficiency_change
                }

        return costs

    @staticmethod
    def find_optimal_liquidity(results: Dict[str, OptimizationResult],
                              liquidity_penalty: float = 0.0) -> Tuple[str, OptimizationResult]:
        """
        Find optimal liquidity level considering utility and liquidity penalty

        Args:
            results: Dictionary of optimization results
            liquidity_penalty: Penalty for holding cash (utility units per % cash)

        Returns:
            Tuple of (profile_name, optimal_result)
        """

        best_adjusted_utility = float('-inf')
        best_profile = None
        best_result = None

        for profile_name, result in results.items():
            if not result.optimization_success:
                continue

            # Calculate adjusted utility (penalize excess liquidity)
            cash_weight = result.optimal_weights.get('Cash', 0.0)
            adjusted_utility = result.expectiminimax_value - (liquidity_penalty * cash_weight)

            if adjusted_utility > best_adjusted_utility:
                best_adjusted_utility = adjusted_utility
                best_profile = profile_name
                best_result = result

        return best_profile, best_result

    @staticmethod
    def liquidity_sensitivity_metrics(results: Dict[str, OptimizationResult]) -> Dict[str, float]:
        """
        Calculate sensitivity metrics for liquidity preferences

        Returns:
            Dictionary with sensitivity measures
        """

        # Extract data for analysis
        liquidity_levels = []
        utilities = []
        returns = []
        volatilities = []
        cash_weights = []

        for result in results.values():
            if result.optimization_success:
                liquidity_levels.append(result.min_cash_pct or 0.0)
                utilities.append(result.expectiminimax_value)
                returns.append(result.expected_return)
                volatilities.append(result.expected_volatility)
                cash_weights.append(result.optimal_weights.get('Cash', 0.0))

        if len(liquidity_levels) < 2:
            return {}

        # Convert to numpy arrays for calculations
        liquidity_array = np.array(liquidity_levels)
        utility_array = np.array(utilities)
        return_array = np.array(returns)
        vol_array = np.array(volatilities)
        cash_array = np.array(cash_weights)

        # Calculate sensitivities (approximate derivatives)
        metrics = {}

        if len(liquidity_array) > 1:
            # Utility sensitivity to liquidity constraints
            utility_sensitivity = np.polyfit(liquidity_array, utility_array, 1)[0]
            metrics['utility_sensitivity'] = utility_sensitivity

            # Return sensitivity
            return_sensitivity = np.polyfit(liquidity_array, return_array, 1)[0]
            metrics['return_sensitivity'] = return_sensitivity

            # Volatility sensitivity
            vol_sensitivity = np.polyfit(liquidity_array, vol_array, 1)[0]
            metrics['volatility_sensitivity'] = vol_sensitivity

            # Cash allocation sensitivity (how much extra cash beyond minimum)
            excess_cash = cash_array - liquidity_array
            cash_sensitivity = np.polyfit(liquidity_array, excess_cash, 1)[0]
            metrics['excess_cash_sensitivity'] = cash_sensitivity

            # Overall efficiency impact
            efficiency_array = return_array / vol_array
            efficiency_sensitivity = np.polyfit(liquidity_array, efficiency_array, 1)[0]
            metrics['efficiency_sensitivity'] = efficiency_sensitivity

        return metrics

    @staticmethod
    def generate_liquidity_report(results: Dict[str, OptimizationResult],
                                 title: str = "Liquidity Preference Analysis") -> str:
        """Generate formatted report for liquidity analysis"""

        report = []
        report.append("=" * 80)
        report.append(title.upper())
        report.append("=" * 80)

        # Summary table
        report.append("\nPORTFOLIO ALLOCATIONS BY LIQUIDITY PREFERENCE:")
        report.append("-" * 80)
        report.append(f"{'Profile':<25} {'Min Cash':<10} {'Actual Cash':<12} {'Stocks':<8} {'Bonds':<8} {'Utility':<10}")
        report.append("-" * 80)

        for profile_name, result in results.items():
            if result.optimization_success:
                report.append(
                    f"{profile_name:<25} "
                    f"{(result.min_cash_pct or 0.0):<10.1%} "
                    f"{result.optimal_weights['Cash']:<12.1%} "
                    f"{result.optimal_weights['Stocks']:<8.1%} "
                    f"{result.optimal_weights['Bonds']:<8.1%} "
                    f"{result.expectiminimax_value:<10.4f}"
                )

        # Cost analysis
        try:
            costs = LiquidityAnalyzer.calculate_liquidity_cost(results)
            report.append("\nLIQUIDITY COST ANALYSIS (relative to 0% minimum cash):")
            report.append("-" * 80)
            report.append(f"{'Profile':<25} {'Utility Cost':<12} {'Return Cost':<12} {'Vol Change':<12}")
            report.append("-" * 80)

            for profile_name, cost_metrics in costs.items():
                if cost_metrics['utility_cost'] != 0.0:  # Skip baseline
                    report.append(
                        f"{profile_name:<25} "
                        f"{cost_metrics['utility_cost']:<12.4f} "
                        f"{cost_metrics['return_cost']:<12.2f}% "
                        f"{cost_metrics['volatility_change']:<12.2f}%"
                    )
        except ValueError:
            report.append("\nLIQUIDITY COST ANALYSIS: Unable to calculate (no 0% baseline)")

        # Sensitivity metrics
        sensitivity = LiquidityAnalyzer.liquidity_sensitivity_metrics(results)
        if sensitivity:
            report.append("\nSENSITIVITY METRICS:")
            report.append("-" * 40)
            for metric, value in sensitivity.items():
                metric_name = metric.replace('_', ' ').title()
                report.append(f"{metric_name:<30}: {value:>8.4f}")

        report.append("\n" + "=" * 80)

        return "\n".join(report)

"""
portfolio_analysis.py
Corrected game theory portfolio analysis comparing preset vs optimized portfolios
with different learning mechanisms
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

from constants import SCENARIOS, ASSET_CLASSES, HORIZON
from scenario_analysis import ScenarioAnalyzer
from learning_mechanisms import BeliefUpdater, PortfolioOptimizer


@dataclass
class PortfolioSpec:
    """Specification for preset portfolios"""
    name: str
    weights: Dict[str, float]


@dataclass
class PerformanceMetrics:
    """Container for portfolio performance metrics"""
    cumulative_returns: Dict[str, float]  # By scenario
    prob_weighted_return: float
    worst_case_return: float
    max_drawdowns: Dict[str, float]  # By scenario
    prob_weighted_drawdown: float
    worst_case_drawdown: float
    within_horizon_losses: Dict[str, float]  # By scenario
    prob_weighted_within_loss: float
    worst_case_within_loss: float
    worst_annual_losses: Dict[str, float]  # By scenario
    annual_loss_counts: Dict[str, int]  # By scenario
    scenario_std: float
    annual_std: float


class GameTheoryPortfolioAnalyzer:
    """Main analyzer for game theory portfolio selection with learning mechanisms"""

    def __init__(self, data_path: str, anchor_year: int = 2019):
        self.data_path = data_path
        self.anchor_year = anchor_year

        # Initialize scenario analyzer
        self.scenario_analyzer = ScenarioAnalyzer(data_path, HORIZON)

        # Get initial probabilities and forecasts
        self.initial_probabilities = self.scenario_analyzer.estimate_probabilities(anchor_year)
        self.scenario_forecasts = self.scenario_analyzer.forecast_returns(anchor_year)

        # Get covariance matrix for learning
        self.covariance_matrix = self.scenario_analyzer._calculate_covariance_matrix(anchor_year, 0)

        # Define preset portfolios
        self.preset_portfolios = {
            "Conservative": PortfolioSpec("Conservative", {"Stocks": 0.4, "Bonds": 0.5, "Cash": 0.1}),
            "Moderate": PortfolioSpec("Moderate", {"Stocks": 0.6, "Bonds": 0.35, "Cash": 0.05}),
            "Aggressive": PortfolioSpec("Aggressive", {"Stocks": 0.8, "Bonds": 0.2, "Cash": 0.0})
        }

        # Define learning mechanisms with configurable lambda
        self.learning_mechanisms = {
            "No_Learning": BeliefUpdater("no_learning"),
            "Bayesian": BeliefUpdater("bayesian"),
            "Adaptive_Slow": BeliefUpdater("adaptive", learning_rate=0.1),      # λ = 0.1 (slow learning)
            "Adaptive_Moderate": BeliefUpdater("adaptive", learning_rate=0.3),  # λ = 0.3 (moderate)
            "Adaptive_Fast": BeliefUpdater("adaptive", learning_rate=0.7),      # λ = 0.7 (fast learning)
            "Adaptive_Full": BeliefUpdater("adaptive", learning_rate=1.0)       # λ = 1.0 (equivalent to Bayesian)
        }

        self.optimizer = PortfolioOptimizer(ASSET_CLASSES)

    def run_complete_analysis(self, observed_path: Optional[List[float]] = None,
                            observed_inflation: Optional[List[float]] = None) -> Dict:
        """
        Run complete game theory portfolio analysis

        Args:
            observed_path: Actual observed GDP growth [2018, 2019, 2020] (optional)
            observed_inflation: Actual observed inflation [2018, 2019, 2020] (optional)

        Returns:
            Complete analysis results
        """

        print("🎯 GAME THEORY PORTFOLIO ANALYSIS")
        print("Expectimin Loss Portfolio with Learning Mechanisms")
        print("="*80)

        results = {
            'initial_probabilities': self.initial_probabilities,
            'preset_portfolios': {},
            'optimized_portfolios': {},
            'detailed_tables': {}
        }

        # If no observed path provided, use most likely scenario path
        if observed_path is None:
            most_likely_scenario = max(self.initial_probabilities.items(), key=lambda x: x[1])[0]
            scenario_def = SCENARIOS[most_likely_scenario]
            observed_path = scenario_def.gdp_growth
            observed_inflation = scenario_def.inflation
            print(f"Using most likely scenario path for analysis: {most_likely_scenario}")

        print(f"Observed GDP path: {observed_path}")
        print(f"Observed inflation path: {observed_inflation}")

        # 1. Analyze preset portfolios (no rebalancing, weight drift)
        print(f"\n{'PRESET PORTFOLIOS ANALYSIS':=^80}")
        for portfolio_name, portfolio_spec in self.preset_portfolios.items():
            print(f"\nAnalyzing {portfolio_name} Portfolio:")
            print(f"Initial weights: {self._format_weights(portfolio_spec.weights)}")

            metrics = self._analyze_preset_portfolio(portfolio_spec)
            results['preset_portfolios'][portfolio_name] = {
                'specification': portfolio_spec,
                'metrics': metrics
            }

            print(f"Probability-weighted return: {metrics.prob_weighted_return*100:.2f}%")
            print(f"Worst case return: {metrics.worst_case_return*100:.2f}%")

        # 2. Analyze optimized portfolios (annual rebalancing with learning)
        print(f"\n{'OPTIMIZED PORTFOLIOS ANALYSIS':=^80}")
        for mechanism_name, belief_updater in self.learning_mechanisms.items():
            print(f"\nAnalyzing {mechanism_name} Learning Mechanism:")

            metrics = self._analyze_optimized_portfolio(
                belief_updater, observed_path, observed_inflation
            )
            results['optimized_portfolios'][mechanism_name] = {
                'learning_mechanism': belief_updater,
                'metrics': metrics
            }

            print(f"Probability-weighted return: {metrics.prob_weighted_return*100:.2f}%")
            print(f"Worst case return: {metrics.worst_case_return*100:.2f}%")

        # 3. Generate detailed tables
        print(f"\n{'GENERATING DETAILED TABLES':=^80}")
        results['detailed_tables'] = self._generate_all_tables(results)

        return results

    def _analyze_preset_portfolio(self, portfolio_spec: PortfolioSpec) -> PerformanceMetrics:
        """Analyze preset portfolio with weight drift (no rebalancing)"""

        scenario_metrics = {}

        for scenario_name, asset_forecasts in self.scenario_forecasts.items():
            # Calculate performance with weight drift
            metrics = self._calculate_preset_performance(portfolio_spec.weights, asset_forecasts)
            scenario_metrics[scenario_name] = metrics

        # Aggregate across scenarios
        return self._aggregate_scenario_metrics(scenario_metrics, self.initial_probabilities)

    def _analyze_optimized_portfolio(self, belief_updater: BeliefUpdater,
                                   observed_path: List[float],
                                   observed_inflation: List[float]) -> PerformanceMetrics:
        """Analyze optimized portfolio with annual rebalancing and learning"""

        scenario_metrics = {}

        for scenario_name, asset_forecasts in self.scenario_forecasts.items():
            # Simulate portfolio evolution for this scenario
            metrics = self._simulate_optimized_performance(
                belief_updater, asset_forecasts, scenario_name, observed_path, observed_inflation
            )
            scenario_metrics[scenario_name] = metrics

        # Aggregate across scenarios
        return self._aggregate_scenario_metrics(scenario_metrics, self.initial_probabilities)

    def _calculate_preset_performance(self, initial_weights: Dict[str, float],
                                    asset_forecasts: Dict[str, List[float]]) -> Dict:
        """Calculate performance metrics for preset portfolio with weight drift"""

        # Convert to arrays for easier calculation
        weights = np.array([initial_weights[asset] for asset in ASSET_CLASSES])
        returns_matrix = np.array([asset_forecasts[asset] for asset in ASSET_CLASSES]).T / 100.0

        # Track portfolio evolution with weight drift
        portfolio_values = [1.0]
        annual_returns = []

        current_weights = weights.copy()

        for year in range(HORIZON):
            year_returns = returns_matrix[year]

            # Calculate portfolio return
            portfolio_return = np.sum(current_weights * year_returns)
            annual_returns.append(portfolio_return)

            # Update portfolio value
            new_value = portfolio_values[-1] * (1 + portfolio_return)
            portfolio_values.append(new_value)

            # Update weights due to drift (no rebalancing)
            asset_multipliers = 1 + year_returns
            new_asset_values = current_weights * asset_multipliers
            current_weights = new_asset_values / np.sum(new_asset_values)

        # Calculate metrics
        cumulative_return = portfolio_values[-1] - 1.0

        # Maximum drawdown
        peak = 1.0
        max_drawdown = 0.0
        for value in portfolio_values[1:]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        # Within-horizon loss (decline from initial value)
        within_horizon_loss = max(0, 1.0 - min(portfolio_values[1:]))

        # Annual loss metrics
        annual_losses = [max(0, -ret) for ret in annual_returns]
        worst_annual_loss = max(annual_losses) if annual_losses else 0.0
        num_annual_losses = sum(1 for loss in annual_losses if loss > 0)

        return {
            'cumulative_return': cumulative_return,
            'annual_returns': annual_returns,
            'portfolio_values': portfolio_values,
            'max_drawdown': max_drawdown,
            'within_horizon_loss': within_horizon_loss,
            'worst_annual_loss': worst_annual_loss,
            'num_annual_losses': num_annual_losses
        }

    def _simulate_optimized_performance(self, belief_updater: BeliefUpdater,
                                      asset_forecasts: Dict[str, List[float]],
                                      true_scenario: str,
                                      observed_path: List[float],
                                      observed_inflation: List[float]) -> Dict:
        """Simulate optimized portfolio performance with annual rebalancing"""

        # Track evolution
        portfolio_values = [1.0]
        annual_returns = []
        current_beliefs = self.initial_probabilities.copy()

        for year in range(HORIZON):
            # Get this year's forecasts for all scenarios
            annual_forecasts = {}
            for scenario_name, forecasts in self.scenario_forecasts.items():
                annual_forecasts[scenario_name] = {
                    asset: forecasts[asset][year] for asset in ASSET_CLASSES
                }

            # Optimize portfolio for this year using corrected expectimin LP
            optimal_weights = self.optimizer.optimize_expectimin(current_beliefs, annual_forecasts)

            # Calculate actual return using true scenario
            actual_return = sum(
                optimal_weights[asset] * asset_forecasts[asset][year] / 100.0
                for asset in ASSET_CLASSES
            )
            annual_returns.append(actual_return)

            # Update portfolio value
            new_value = portfolio_values[-1] * (1 + actual_return)
            portfolio_values.append(new_value)

            # Update beliefs for next year (if not last year)
            if year < HORIZON - 1:
                observed_so_far = observed_path[:year+1]
                observed_inf_so_far = observed_inflation[:year+1]
                current_beliefs = belief_updater.update_beliefs(
                    current_beliefs, observed_so_far, observed_inf_so_far, self.covariance_matrix
                )

        # Calculate same metrics as preset portfolio
        cumulative_return = portfolio_values[-1] - 1.0

        # Maximum drawdown
        peak = 1.0
        max_drawdown = 0.0
        for value in portfolio_values[1:]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        # Within-horizon loss
        within_horizon_loss = max(0, 1.0 - min(portfolio_values[1:]))

        # Annual loss metrics
        annual_losses = [max(0, -ret) for ret in annual_returns]
        worst_annual_loss = max(annual_losses) if annual_losses else 0.0
        num_annual_losses = sum(1 for loss in annual_losses if loss > 0)

        return {
            'cumulative_return': cumulative_return,
            'annual_returns': annual_returns,
            'portfolio_values': portfolio_values,
            'max_drawdown': max_drawdown,
            'within_horizon_loss': within_horizon_loss,
            'worst_annual_loss': worst_annual_loss,
            'num_annual_losses': num_annual_losses
        }

    def _aggregate_scenario_metrics(self, scenario_metrics: Dict[str, Dict],
                                  probabilities: Dict[str, float]) -> PerformanceMetrics:
        """Aggregate metrics across scenarios using probabilities"""

        # Extract cumulative returns
        cumulative_returns = {s: metrics['cumulative_return']
                            for s, metrics in scenario_metrics.items()}

        # Probability-weighted metrics
        prob_weighted_return = sum(probabilities[s] * cumulative_returns[s]
                                 for s in cumulative_returns.keys())
        worst_case_return = min(cumulative_returns.values())

        # Maximum drawdowns
        max_drawdowns = {s: metrics['max_drawdown']
                        for s, metrics in scenario_metrics.items()}
        prob_weighted_drawdown = sum(probabilities[s] * max_drawdowns[s]
                                   for s in max_drawdowns.keys())
        worst_case_drawdown = max(max_drawdowns.values())

        # Within-horizon losses
        within_losses = {s: metrics['within_horizon_loss']
                        for s, metrics in scenario_metrics.items()}
        prob_weighted_within_loss = sum(probabilities[s] * within_losses[s]
                                      for s in within_losses.keys())
        worst_case_within_loss = max(within_losses.values())

        # Annual loss metrics
        worst_annual_losses = {s: metrics['worst_annual_loss']
                             for s, metrics in scenario_metrics.items()}
        annual_loss_counts = {s: metrics['num_annual_losses']
                            for s, metrics in scenario_metrics.items()}

        # Standard deviations
        scenario_std = np.std(list(cumulative_returns.values()))

        # Annual std (across all scenarios and years)
        all_annual_returns = []
        for metrics in scenario_metrics.values():
            all_annual_returns.extend(metrics['annual_returns'])
        annual_std = np.std(all_annual_returns)

        return PerformanceMetrics(
            cumulative_returns=cumulative_returns,
            prob_weighted_return=prob_weighted_return,
            worst_case_return=worst_case_return,
            max_drawdowns=max_drawdowns,
            prob_weighted_drawdown=prob_weighted_drawdown,
            worst_case_drawdown=worst_case_drawdown,
            within_horizon_losses=within_losses,
            prob_weighted_within_loss=prob_weighted_within_loss,
            worst_case_within_loss=worst_case_within_loss,
            worst_annual_losses=worst_annual_losses,
            annual_loss_counts=annual_loss_counts,
            scenario_std=scenario_std,
            annual_std=annual_std
        )

    def _generate_all_tables(self, results: Dict) -> Dict:
        """Generate all detailed tables as specified"""

        tables = {}

        # Table 1: Preset Portfolios Analysis
        tables['preset_portfolios_table'] = self._generate_preset_portfolio_table(results)

        # Table 2: Portfolio spread across scenarios and std across years
        tables['portfolio_spread_table'] = self._generate_portfolio_spread_table(results)

        # Table 3: Comprehensive comparison table
        tables['comprehensive_comparison_table'] = self._generate_comprehensive_comparison_table(results)

        # Print summary of tables
        print(f"✅ Table 1: Preset portfolios analysis ({len(tables['preset_portfolios_table'])} rows)")
        print(f"✅ Table 2: Portfolio spread analysis ({len(tables['portfolio_spread_table'])} rows)")
        print(f"✅ Table 3: Comprehensive comparison ({len(tables['comprehensive_comparison_table'])} rows)")

        return tables

    def _generate_preset_portfolio_table(self, results: dict) -> pd.DataFrame:
        """
        Table 1: Preset Portfolios Analysis
        For each scenario: cumulative returns, max drawdown, within-horizon loss,
        worst annual loss, num annual losses
        + Probability weighted average + Worst case
        """

        table_data = []

        for portfolio_name, data in results['preset_portfolios'].items():
            metrics = data['metrics']

            # Add row for each scenario
            for scenario in SCENARIOS.keys():
                table_data.append({
                    'Portfolio': portfolio_name,
                    'Scenario': scenario,
                    'Probability_%': results['initial_probabilities'][scenario] * 100,
                    'Cumulative_Return_%': metrics.cumulative_returns[scenario] * 100,
                    'Max_Drawdown_%': metrics.max_drawdowns[scenario] * 100,
                    'Within_Horizon_Loss_%': metrics.within_horizon_losses[scenario] * 100,
                    'Worst_Annual_Loss_%': metrics.worst_annual_losses[scenario] * 100,
                    'Num_Annual_Losses': metrics.annual_loss_counts[scenario]
                })

            # Add probability-weighted average row
            table_data.append({
                'Portfolio': portfolio_name,
                'Scenario': 'Prob_Weighted_Avg',
                'Probability_%': 100.0,
                'Cumulative_Return_%': metrics.prob_weighted_return * 100,
                'Max_Drawdown_%': metrics.prob_weighted_drawdown * 100,
                'Within_Horizon_Loss_%': metrics.prob_weighted_within_loss * 100,
                'Worst_Annual_Loss_%': sum(results['initial_probabilities'][s] *
                                         metrics.worst_annual_losses[s] for s in SCENARIOS.keys()) * 100,
                'Num_Annual_Losses': sum(results['initial_probabilities'][s] *
                                       metrics.annual_loss_counts[s] for s in SCENARIOS.keys())
            })

            # Add worst case row
            table_data.append({
                'Portfolio': portfolio_name,
                'Scenario': 'Worst_Case',
                'Probability_%': 100.0,
                'Cumulative_Return_%': metrics.worst_case_return * 100,
                'Max_Drawdown_%': metrics.worst_case_drawdown * 100,
                'Within_Horizon_Loss_%': metrics.worst_case_within_loss * 100,
                'Worst_Annual_Loss_%': max(metrics.worst_annual_losses.values()) * 100,
                'Num_Annual_Losses': max(metrics.annual_loss_counts.values())
            })

        return pd.DataFrame(table_data)

    def _generate_portfolio_spread_table(self, results: dict) -> pd.DataFrame:
        """
        Table 2: Portfolio spread across scenarios and std across years
        """

        spread_data = []

        # Preset portfolios
        for portfolio_name, data in results['preset_portfolios'].items():
            metrics = data['metrics']
            spread_data.append({
                'Portfolio': portfolio_name,
                'Type': 'Preset',
                'Learning_Mechanism': 'N/A',
                'Lambda': 'N/A',
                'Scenario_Std_%': metrics.scenario_std * 100,
                'Annual_Std_%': metrics.annual_std * 100,
                'Range_Scenarios_%': (max(metrics.cumulative_returns.values()) -
                                    min(metrics.cumulative_returns.values())) * 100
            })

        # Optimized portfolios
        for portfolio_name, data in results['optimized_portfolios'].items():
            metrics = data['metrics']

            # Extract lambda value for adaptive learning
            if "Adaptive" in portfolio_name:
                if "Slow" in portfolio_name:
                    lambda_str = "0.1"
                elif "Moderate" in portfolio_name:
                    lambda_str = "0.3"
                elif "Fast" in portfolio_name:
                    lambda_str = "0.7"
                elif "Full" in portfolio_name:
                    lambda_str = "1.0"
                else:
                    lambda_str = "0.3"
            else:
                lambda_str = "N/A"

            spread_data.append({
                'Portfolio': f"{portfolio_name}_Optimized",
                'Type': 'Optimized',
                'Learning_Mechanism': portfolio_name,
                'Lambda': lambda_str,
                'Scenario_Std_%': metrics.scenario_std * 100,
                'Annual_Std_%': metrics.annual_std * 100,
                'Range_Scenarios_%': (max(metrics.cumulative_returns.values()) -
                                    min(metrics.cumulative_returns.values())) * 100
            })

        return pd.DataFrame(spread_data)

    def _generate_comprehensive_comparison_table(self, results: dict) -> pd.DataFrame:
        """
        Table 3: Big comparison table
        Preset (not actively managed) vs Optimized (actively managed)
        Expected return, realized return, expected loss, risk measure
        """

        comparison_data = []

        # Preset portfolios
        for portfolio_name, data in results['preset_portfolios'].items():
            metrics = data['metrics']
            comparison_data.append({
                'Portfolio': portfolio_name,
                'Management': 'Passive_(Weight_Drift)',
                'Expected_Return_%': metrics.prob_weighted_return * 100,
                'Realized_Return_%': metrics.prob_weighted_return * 100,  # Same for preset
                'Expected_Loss_%': sum(results['initial_probabilities'][s] *
                                     max(0, -metrics.cumulative_returns[s]) for s in SCENARIOS.keys()) * 100,
                'Worst_Case_Return_%': metrics.worst_case_return * 100,
                'Scenario_Risk_Std_%': metrics.scenario_std * 100,
                'Max_Drawdown_Risk_%': metrics.prob_weighted_drawdown * 100
            })

        # Optimized portfolios
        for portfolio_name, data in results['optimized_portfolios'].items():
            metrics = data['metrics']
            comparison_data.append({
                'Portfolio': f"{portfolio_name}_Learning",
                'Management': 'Active_(Annual_Rebalancing)',
                'Expected_Return_%': metrics.prob_weighted_return * 100,
                'Realized_Return_%': metrics.prob_weighted_return * 100,  # Calculated with rebalancing
                'Expected_Loss_%': sum(results['initial_probabilities'][s] *
                                     max(0, -metrics.cumulative_returns[s]) for s in SCENARIOS.keys()) * 100,
                'Worst_Case_Return_%': metrics.worst_case_return * 100,
                'Scenario_Risk_Std_%': metrics.scenario_std * 100,
                'Max_Drawdown_Risk_%': metrics.prob_weighted_drawdown * 100
            })

        return pd.DataFrame(comparison_data)

    def _format_weights(self, weights: Dict[str, float]) -> str:
        """Format portfolio weights for display"""
        return ", ".join(f"{asset}: {weight*100:.0f}%" for asset, weight in weights.items())

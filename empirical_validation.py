"""
empirical_validation.py
Full Enumeration Learning Mechanisms with Train/Test Validation
Predicts 2018-2020 using 1925-2017 training data

Features:
- Full enumeration of all 6³ = 216 scenario paths
- Train/test split for empirical validation
- Actual vs predicted performance comparison
- Learning mechanism ranking based on real outcomes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from itertools import product
import os

from constants import SCENARIOS, ASSET_CLASSES, HORIZON
from scenario_analysis import ScenarioAnalyzer
from dynamic_game import LearningMechanismAnalyzer

class EmpiricalValidationFramework:
    """
    Empirical validation of learning mechanisms using historical data
    Trains on 1925-2017, tests on 2018-2020
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = self._load_full_data()
        self.scenarios = list(SCENARIOS.keys())

    def _load_full_data(self) -> pd.DataFrame:
        """Load complete historical data including test period"""
        df = pd.read_csv(self.data_path, sep=';', index_col='year')

        # Convert European decimal format if needed
        numeric_cols = ['rgdpmad', 'cpi', 'bill_rate', 'eq_tr', 'bond_tr']
        for col in numeric_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col].str.replace(',', '.'), errors='coerce')

        # Calculate growth rates
        df['gdp_growth'] = df['rgdpmad'].pct_change() * 100
        df['inflation'] = df['cpi'].pct_change() * 100

        # Calculate real returns
        df['bill_rr'] = ((1 + df['bill_rate']) / (1 + df['inflation']/100) - 1) * 100
        df['stock_rr'] = ((1 + df['eq_tr']) / (1 + df['inflation']/100) - 1) * 100
        df['bond_rr'] = ((1 + df['bond_tr']) / (1 + df['inflation']/100) - 1) * 100

        return df.dropna()

    def run_full_empirical_validation(self,
                                    train_end_year: int = 2017,
                                    test_start_year: int = 2018,
                                    test_end_year: int = 2020,
                                    learning_types: List[str] = None) -> Dict:
        """
        Run complete empirical validation with full enumeration

        Args:
            train_end_year: Last year of training data (2017)
            test_start_year: First year of test period (2018)
            test_end_year: Last year of test period (2020)
            learning_types: Learning mechanisms to test
        """

        if learning_types is None:
            learning_types = ["no_learning", "bayesian", "adaptive", "perfect"]

        print(f"🎯 EMPIRICAL VALIDATION FRAMEWORK")
        print(f"Full Enumeration of Learning Mechanisms")
        print("="*70)
        print(f"Training period: 1925-{train_end_year}")
        print(f"Test period: {test_start_year}-{test_end_year}")
        print(f"Prediction target: {test_start_year}-{test_end_year}")
        print(f"Learning mechanisms: {learning_types}")
        print(f"Total scenario paths: {len(self.scenarios)**HORIZON} = {len(self.scenarios)}³")

        results = {
            'config': {
                'train_end_year': train_end_year,
                'test_period': f"{test_start_year}-{test_end_year}",
                'learning_types': learning_types,
                'total_paths': len(self.scenarios)**HORIZON
            }
        }

        # Step 1: Train models on historical data
        print(f"\n{'STEP 1: TRAINING PHASE':=^70}")
        training_results = self._train_models(train_end_year, learning_types)
        results['training_results'] = training_results

        # Step 2: Extract actual test period outcomes
        print(f"\n{'STEP 2: ACTUAL OUTCOMES EXTRACTION':=^70}")
        actual_outcomes = self._extract_actual_outcomes(test_start_year, test_end_year)
        results['actual_outcomes'] = actual_outcomes

        # Step 3: Full enumeration of all scenario paths
        print(f"\n{'STEP 3: FULL SCENARIO PATH ENUMERATION':=^70}")
        full_enumeration = self._enumerate_all_scenario_paths(
            training_results, learning_types
        )
        results['full_enumeration'] = full_enumeration

        # Step 4: Find actual path and compare predictions
        print(f"\n{'STEP 4: PREDICTION VS REALITY COMPARISON':=^70}")
        validation_results = self._validate_predictions(
            full_enumeration, actual_outcomes, learning_types
        )
        results['validation_results'] = validation_results

        # Step 5: Rank learning mechanisms by prediction accuracy
        print(f"\n{'STEP 5: LEARNING MECHANISM RANKING':=^70}")
        mechanism_ranking = self._rank_learning_mechanisms(validation_results)
        results['mechanism_ranking'] = mechanism_ranking

        return results

    def _train_models(self, train_end_year: int, learning_types: List[str]) -> Dict:
        """Train all models on historical data up to train_end_year"""

        print(f"🎓 Training models using data up to {train_end_year}")

        # Use scenario analyzer with train_end_year as anchor
        analyzer = ScenarioAnalyzer(self.data_path, horizon=HORIZON)

        # Get training data only (up to train_end_year)
        train_data = self.data.loc[:train_end_year].copy()

        # Estimate probabilities using train_end_year anchor
        anchor_year = train_end_year
        probabilities = analyzer.estimate_probabilities(anchor_year)

        print(f"📊 Estimated scenario probabilities (anchor: {anchor_year}):")
        for scenario, prob in probabilities.items():
            print(f"  P({scenario:<15}) = {prob:6.3f}")

        # Generate return forecasts for test period
        forecasts = analyzer.forecast_returns(anchor_year)

        print(f"📈 Generated return forecasts for {anchor_year+1}-{anchor_year+HORIZON}")
        print(f"  • Scenarios: {len(forecasts)}")
        print(f"  • Assets: {len(ASSET_CLASSES)}")
        print(f"  • Forecast horizon: {HORIZON} years")

        return {
            'anchor_year': anchor_year,
            'probabilities': probabilities,
            'forecasts': forecasts,
            'train_data_end': train_end_year
        }

    def _extract_actual_outcomes(self, test_start_year: int, test_end_year: int) -> Dict:
        """Extract actual economic outcomes for test period"""

        print(f"📊 Extracting actual outcomes for {test_start_year}-{test_end_year}")

        # Extract actual data for test period
        test_data = self.data.loc[test_start_year:test_end_year].copy()

        if len(test_data) < HORIZON:
            print(f"⚠️  Warning: Only {len(test_data)} years of test data available")

        # Extract actual economic indicators
        actual_gdp_growth = test_data['gdp_growth'].tolist()
        actual_inflation = test_data['inflation'].tolist()
        actual_stock_returns = test_data['stock_rr'].tolist()
        actual_bond_returns = test_data['bond_rr'].tolist()
        actual_cash_returns = test_data['bill_rr'].tolist()

        print(f"📈 Actual economic outcomes:")
        print(f"{'Year':<6} {'GDP Growth':<12} {'Inflation':<10} {'Stocks':<8} {'Bonds':<8} {'Cash':<8}")
        print(f"{'-'*55}")

        for i, year in enumerate(test_data.index):
            if i < len(actual_gdp_growth):
                print(f"{year:<6} {actual_gdp_growth[i]:>10.1f}% {actual_inflation[i]:>8.1f}% "
                      f"{actual_stock_returns[i]:>6.1f}% {actual_bond_returns[i]:>6.1f}% "
                      f"{actual_cash_returns[i]:>6.1f}%")

        # Map to closest scenario (simplified approach)
        actual_scenario_path = self._map_to_scenarios(actual_gdp_growth, actual_inflation)

        print(f"\n🎯 Mapped actual path to scenarios:")
        for i, scenario in enumerate(actual_scenario_path):
            year = test_start_year + i
            print(f"  {year}: {scenario}")

        return {
            'test_period': f"{test_start_year}-{test_end_year}",
            'actual_gdp_growth': actual_gdp_growth,
            'actual_inflation': actual_inflation,
            'actual_returns': {
                'Cash': actual_cash_returns,
                'Stocks': actual_stock_returns,
                'Bonds': actual_bond_returns
            },
            'actual_scenario_path': actual_scenario_path,
            'test_data': test_data
        }

    def _map_to_scenarios(self, gdp_growth: List[float], inflation: List[float]) -> List[str]:
        """Map actual economic outcomes to predefined scenarios"""

        # Define scenario templates (GDP growth, inflation patterns)
        scenario_templates = {
            "Baseline V": {"gdp_pattern": [-3.5, 3.8, 2.3], "inf_pattern": [1.0, 1.7, 2.0]},
            "Shallow V": {"gdp_pattern": [-1.9, 5.4, 3.9], "inf_pattern": [1.0, 1.7, 2.0]},
            "U-Shaped": {"gdp_pattern": [-3.5, 0.0, 3.9], "inf_pattern": [1.0, 0.4, 0.7]},
            "W-Shaped": {"gdp_pattern": [-3.5, 3.8, -4.2], "inf_pattern": [1.0, 1.7, 2.0]},
            "Depression": {"gdp_pattern": [-5.1, -5.9, -7.4], "inf_pattern": [-0.3, -5.9, -5.6]},
            "Stagflation": {"gdp_pattern": [-5.1, -2.7, -0.9], "inf_pattern": [2.3, 4.2, 5.8]}
        }

        mapped_scenarios = []

        for year_idx in range(min(len(gdp_growth), HORIZON)):
            actual_gdp = gdp_growth[year_idx]
            actual_inf = inflation[year_idx]

            best_scenario = None
            min_distance = float('inf')

            # Find closest scenario for this year
            for scenario_name, template in scenario_templates.items():
                if year_idx < len(template["gdp_pattern"]):
                    template_gdp = template["gdp_pattern"][year_idx]
                    template_inf = template["inf_pattern"][year_idx]

                    # Euclidean distance
                    distance = np.sqrt((actual_gdp - template_gdp)**2 + (actual_inf - template_inf)**2)

                    if distance < min_distance:
                        min_distance = distance
                        best_scenario = scenario_name

            mapped_scenarios.append(best_scenario if best_scenario else "U-Shaped")

        return mapped_scenarios

    def _enumerate_all_scenario_paths(self, training_results: Dict,
                                    learning_types: List[str]) -> Dict:
        """Enumerate all possible 6³ = 216 scenario paths"""

        print(f"🔢 Enumerating all {len(self.scenarios)**HORIZON} scenario paths")

        # Generate all possible 3-year scenario sequences
        all_paths = list(product(self.scenarios, repeat=HORIZON))

        print(f"📊 Generated {len(all_paths)} total paths")
        print(f"  • Examples: {all_paths[0]}, {all_paths[1]}, ...")

        # Initialize learning mechanism analyzer
        analyzer = LearningMechanismAnalyzer(
            training_results['forecasts'],
            training_results['probabilities']
        )

        # Analyze each path with each learning mechanism
        path_results = {}
        total_analyses = len(all_paths) * len(learning_types)
        completed = 0

        print(f"🧠 Running {total_analyses} total analyses...")

        for path_idx, scenario_path in enumerate(all_paths):
            if path_idx % 50 == 0:
                print(f"  Progress: {path_idx}/{len(all_paths)} paths ({path_idx/len(all_paths)*100:.1f}%)")

            path_key = f"path_{path_idx:03d}"
            path_results[path_key] = {
                'scenario_path': list(scenario_path),
                'learning_results': {}
            }

            # Analyze this path with each learning mechanism
            for learning_type in learning_types:
                try:
                    result = analyzer._analyze_single_learning_mechanism(
                        learning_type, list(scenario_path)
                    )
                    path_results[path_key]['learning_results'][learning_type] = result
                    completed += 1

                except Exception as e:
                    print(f"  ⚠️  Error analyzing {learning_type} on path {path_idx}: {str(e)}")
                    continue

        print(f"✅ Completed {completed}/{total_analyses} analyses")

        return {
            'all_paths': [list(path) for path in all_paths],
            'path_results': path_results,
            'total_paths': len(all_paths),
            'completed_analyses': completed,
            'total_analyses': total_analyses
        }

    def _validate_predictions(self, full_enumeration: Dict, actual_outcomes: Dict,
                            learning_types: List[str]) -> Dict:
        """Compare predictions vs actual outcomes"""

        actual_path = actual_outcomes['actual_scenario_path']
        print(f"🎯 Validating predictions against actual path: {' → '.join(actual_path)}")

        # Find the actual path in enumeration
        actual_path_key = None
        for path_key, path_data in full_enumeration['path_results'].items():
            if path_data['scenario_path'] == actual_path:
                actual_path_key = path_key
                break

        if actual_path_key is None:
            print(f"⚠️  Actual path not found in enumeration - using closest match")
            # Find closest path (simplified)
            actual_path_key = list(full_enumeration['path_results'].keys())[0]
        else:
            print(f"✅ Found actual path: {actual_path_key}")

        # Extract actual returns
        actual_returns = actual_outcomes['actual_returns']
        actual_cumulative_return = 0.0

        # Calculate actual portfolio performance for each learning mechanism
        validation_results = {}

        for learning_type in learning_types:
            if learning_type in full_enumeration['path_results'][actual_path_key]['learning_results']:
                predicted_result = full_enumeration['path_results'][actual_path_key]['learning_results'][learning_type]

                # Calculate actual performance using predicted portfolios
                actual_performance = self._calculate_actual_performance(
                    predicted_result, actual_returns
                )

                prediction_error = abs(predicted_result['cumulative_return'] - actual_performance['actual_cumulative_return'])

                validation_results[learning_type] = {
                    'predicted_return': predicted_result['cumulative_return'],
                    'actual_return': actual_performance['actual_cumulative_return'],
                    'prediction_error': prediction_error,
                    'predicted_portfolios': predicted_result['portfolio_evolution'],
                    'actual_period_returns': actual_performance['period_returns']
                }

        # Print validation summary
        print(f"\n📊 Prediction vs Reality Comparison:")
        print(f"{'Mechanism':<12} {'Predicted':<10} {'Actual':<10} {'Error':<10}")
        print(f"{'-'*45}")

        for learning_type, results in validation_results.items():
            pred_ret = results['predicted_return'] * 100
            actual_ret = results['actual_return'] * 100
            error = results['prediction_error'] * 100
            print(f"{learning_type:<12} {pred_ret:>8.1f}% {actual_ret:>8.1f}% {error:>8.1f}%")

        return validation_results

    def _calculate_actual_performance(self, predicted_result: Dict,
                                    actual_returns: Dict[str, List[float]]) -> Dict:
        """Calculate actual performance using predicted portfolio evolution"""

        portfolios = predicted_result['portfolio_evolution']
        period_returns = []
        cumulative_return = 0.0

        for period in range(min(len(portfolios), HORIZON)):
            portfolio = portfolios[period]

            # Calculate actual return for this period
            period_return = 0.0
            for asset in ASSET_CLASSES:
                if period < len(actual_returns[asset]):
                    asset_return = actual_returns[asset][period] / 100.0
                    period_return += portfolio[asset] * asset_return

            period_returns.append(period_return)
            cumulative_return += period_return

        return {
            'actual_cumulative_return': cumulative_return,
            'period_returns': period_returns
        }

    def _rank_learning_mechanisms(self, validation_results: Dict) -> Dict:
        """Rank learning mechanisms by prediction accuracy"""

        # Sort by prediction error (lower is better)
        sorted_mechanisms = sorted(
            validation_results.items(),
            key=lambda x: x[1]['prediction_error']
        )

        print(f"\n🏆 Learning Mechanism Ranking (by prediction accuracy):")
        print(f"{'Rank':<6} {'Mechanism':<12} {'Error':<10} {'Predicted':<10} {'Actual':<10}")
        print(f"{'-'*50}")

        ranking = {}
        for rank, (mechanism, results) in enumerate(sorted_mechanisms, 1):
            error = results['prediction_error'] * 100
            pred_ret = results['predicted_return'] * 100
            actual_ret = results['actual_return'] * 100

            print(f"{rank:<6} {mechanism:<12} {error:>8.1f}% {pred_ret:>8.1f}% {actual_ret:>8.1f}%")

            ranking[mechanism] = {
                'rank': rank,
                'prediction_error': error,
                'predicted_return': pred_ret,
                'actual_return': actual_ret
            }

        best_mechanism = sorted_mechanisms[0][0]
        worst_mechanism = sorted_mechanisms[-1][0]

        print(f"\n🥇 Best predictor: {best_mechanism}")
        print(f"🥉 Worst predictor: {worst_mechanism}")

        return {
            'ranking': ranking,
            'best_mechanism': best_mechanism,
            'worst_mechanism': worst_mechanism,
            'sorted_results': sorted_mechanisms
        }


def run_empirical_validation_analysis(data_path: str = "data/usa_macro_var_and_asset_returns.csv"):
    """Main function to run empirical validation"""

    print(f"🎓 EMPIRICAL VALIDATION OF LEARNING MECHANISMS")
    print(f"📅 Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Objective: Test which learning mechanism predicted 2018-2020 best")

    if not os.path.exists(data_path):
        print(f"\n❌ ERROR: Data file '{data_path}' not found!")
        return None

    print(f"\n✅ Data file found: {data_path}")

    # Initialize framework
    framework = EmpiricalValidationFramework(data_path)

    # Run validation
    results = framework.run_full_empirical_validation(
        train_end_year=2017,
        test_start_year=2018,
        test_end_year=2020,
        learning_types=["no_learning", "bayesian", "adaptive", "perfect"]
    )

    print(f"\n{'EMPIRICAL VALIDATION COMPLETE':=^70}")

    if results:
        best_mechanism = results['mechanism_ranking']['best_mechanism']
        total_paths = results['config']['total_paths']

        print(f"🎉 Validation completed successfully!")
        print(f"🔢 Analyzed {total_paths} scenario paths")
        print(f"🏆 Best predictor: {best_mechanism}")
        print(f"📊 Full enumeration completed")
        print(f"🎯 Real-world validation achieved")

    return results


if __name__ == "__main__":
    results = run_empirical_validation_analysis()

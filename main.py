"""
main.py
Clean main analysis script for game theory portfolio selection
"""

from typing import Dict, List

from optimization import sequential_optimize, calculate_portfolio_performance
from learning import create_learning_function
from scenario_analysis import ScenarioAnalyzer
from constants import SCENARIOS


def display_scenario_paths():
    """Display scenario economic paths"""
    print("SCENARIO ECONOMIC PATHS")
    print("=" * 80)
    print(f"{'Scenario':<15} {'GDP Year 1':<10} {'GDP Year 2':<10} {'GDP Year 3':<10} {'INF Year 1':<10} {'INF Year 2':<10} {'INF Year 3':<10}")
    print("-" * 80)

    for name, scenario in SCENARIOS.items():
        gdp_str = [f"{g:+.1f}%" for g in scenario.gdp_growth]
        inf_str = [f"{i:+.1f}%" for i in scenario.inflation]
        print(f"{name:<15} {gdp_str[0]:<10} {gdp_str[1]:<10} {gdp_str[2]:<10} {inf_str[0]:<10} {inf_str[1]:<10} {inf_str[2]:<10}")


def display_anchor_and_test_paths(anchor_year: int, test_path: List[float]):
    """Display anchor year and 2018-2020 test path"""
    print(f"\nANCHOR YEAR & TEST PERIOD")
    print("=" * 50)

    print(f"Anchor Year: {anchor_year}")

    test_gdp = test_path[:3]
    test_inf = test_path[3:]

    print(f"\n2018-2020 Realized Economic Path:")
    print(f"{'Year':<6} {'GDP Growth':<12} {'Inflation':<10}")
    print("-" * 30)
    print(f"{'2018':<6} {test_gdp[0]:+.1f}%{'':<7} {test_inf[0]:+.1f}%")
    print(f"{'2019':<6} {test_gdp[1]:+.1f}%{'':<7} {test_inf[1]:+.1f}%")
    print(f"{'2020':<6} {test_gdp[2]:+.1f}%{'':<7} {test_inf[2]:+.1f}%")


def display_scenario_probabilities(probabilities: Dict[str, float]):
    """Display scenario probabilities"""
    print(f"\nSCENARIO PROBABILITIES")
    print("=" * 30)

    for scenario, prob in probabilities.items():
        print(f"{scenario:<15}: {prob:6.1%}")


def display_scenario_forecasts(forecasts: Dict[str, Dict[str, List[float]]]):
    """Display scenario asset return forecasts"""
    print(f"\nSCENARIO ASSET RETURN FORECASTS")
    print("=" * 80)

    for scenario, asset_forecasts in forecasts.items():
        print(f"\n{scenario}:")
        for asset, returns in asset_forecasts.items():
            returns_str = [f"{r:+.1f}%" for r in returns]
            print(f"  {asset:<8}: {returns_str[0]:<8} {returns_str[1]:<8} {returns_str[2]:<8}")


def display_portfolio_weights(weights_results: Dict[str, List[Dict[str, float]]]):
    """Display portfolio weights for different learning mechanisms"""
    print(f"\nPORTFOLIO WEIGHTS BY LEARNING MECHANISM")
    print("=" * 60)

    for mechanism, weights_sequence in weights_results.items():
        print(f"\n{mechanism}:")
        print(f"{'Year':<6} {'Cash':<8} {'Stocks':<8} {'Bonds':<8}")
        print("-" * 30)

        for year, weights in enumerate(weights_sequence):
            cash = weights.get('Cash', 0)
            stocks = weights.get('Stocks', 0)
            bonds = weights.get('Bonds', 0)
            print(f"{year+1:<6} {cash:<8.3f} {stocks:<8.3f} {bonds:<8.3f}")


def display_performance_analysis(performance_results: Dict[str, Dict]):
    """Display performance analysis for different learning mechanisms"""
    print(f"\nPERFORMANCE ANALYSIS (2018-2020)")
    print("=" * 60)

    print(f"{'Mechanism':<20} {'Cumulative Return':<18} {'Final Value':<12}")
    print("-" * 50)

    for mechanism, performance in performance_results.items():
        cum_return = performance['cumulative_return']
        final_value = performance['final_value']
        print(f"{mechanism:<20} {cum_return*100:<18.2f}% {final_value:<12.3f}")

    print(f"\nANNUAL RETURNS BY MECHANISM:")
    print(f"{'Mechanism':<20} {'2018':<8} {'2019':<8} {'2020':<8}")
    print("-" * 45)

    for mechanism, performance in performance_results.items():
        annual = performance['annual_returns']
        print(f"{mechanism:<20} {annual[0]*100:<8.2f}% {annual[1]*100:<8.2f}% {annual[2]*100:<8.2f}%")


def main():
    """Main analysis function"""

    print("GAME THEORY PORTFOLIO SELECTION - CLEAN ANALYSIS")
    print("=" * 80)

    # Initialize analyzer
    data_path = "data/usa_macro_var_and_asset_returns.csv"
    anchor_year = 2017

    try:
        analyzer = ScenarioAnalyzer(data_path)

        # Get scenario probabilities and forecasts
        probabilities = analyzer.estimate_probabilities(anchor_year)
        forecasts = analyzer.forecast_returns(anchor_year)

        # 2018-2020 actual path for out-of-sample test
        test_path = [2.9, 2.2, -3.4, 2.4, 1.8, 1.2]  # [gdp1, gdp2, gdp3, inf1, inf2, inf3]

        # Actual 2018-2020 asset returns (for performance analysis)
        actual_returns = {
            'Cash': [2.4, 2.3, 0.6],      # Approximate 3-month Treasury rates
            'Stocks': [-4.4, 31.5, 18.4], # S&P 500 total returns
            'Bonds': [0.9, 8.7, 7.5]      # 10-year Treasury bond returns
        }

        # Display information
        display_scenario_paths()
        display_anchor_and_test_paths(anchor_year, test_path)
        display_scenario_probabilities(probabilities)
        display_scenario_forecasts(forecasts)

        # Convert scenarios to format needed by learning functions
        scenarios_dict = {}
        for name, scenario in SCENARIOS.items():
            scenarios_dict[name] = {
                'gdp_growth': scenario.gdp_growth,
                'inflation': scenario.inflation
            }

        # Learning mechanisms to test
        learning_mechanisms = {
            'No Learning': create_learning_function('no_learning'),
            'Adaptive (λ=0.3)': create_learning_function('adaptive', 0.3),
            'Adaptive (λ=0.5)': create_learning_function('adaptive', 0.5),
            'Adaptive (λ=0.7)': create_learning_function('adaptive', 0.7),
            'Bayesian': create_learning_function('bayesian')
        }

        # Get covariance matrix for learning
        covariance_matrix = analyzer._calculate_covariance_matrix(anchor_year, 0)

        # Calculate portfolio weights for each learning mechanism
        weights_results = {}

        for mechanism_name, learning_function in learning_mechanisms.items():
            weights_sequence = sequential_optimize(
                initial_probabilities=probabilities,
                scenario_forecasts=forecasts,
                evidence_path=test_path,
                learning_function=learning_function,
                scenarios_dict=scenarios_dict,
                covariance_matrix=covariance_matrix,
                horizon=3
            )
            weights_results[mechanism_name] = weights_sequence

        # Display portfolio weights
        display_portfolio_weights(weights_results)

        # Calculate performance using actual 2018-2020 returns
        performance_results = {}

        for mechanism_name, weights_sequence in weights_results.items():
            performance = calculate_portfolio_performance(weights_sequence, actual_returns)
            performance_results[mechanism_name] = performance

        # Display performance analysis
        display_performance_analysis(performance_results)

        print(f"\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

"""
main.py
Simple sequential portfolio optimization with Bayesian learning
"""

from optimization import optimize_portfolio
from learning import update_beliefs
from scenario_analysis import ScenarioAnalyzer
from constants import SCENARIOS


def print_dataset_info(analyzer):
    """Print head and tail of processed dataset"""
    print("\n1) PROCESSED DATASET")
    print("=" * 50)

    data = analyzer.data
    print("Dataset Head (first 5 rows):")
    print(data.head())
    print("\nDataset Tail (last 5 rows):")
    print(data.tail())
    print(f"\nDataset shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")


def print_scenarios():
    """Print scenario paths"""
    print("\n2) SCENARIOS WITH PATHS")
    print("=" * 50)

    print(f"{'Scenario':<15} {'GDP Year 1':<10} {'GDP Year 2':<10} {'GDP Year 3':<10} {'INF Year 1':<10} {'INF Year 2':<10} {'INF Year 3':<10}")
    print("-" * 80)

    for name, scenario in SCENARIOS.items():
        gdp_str = [f"{g:+.1f}%" for g in scenario.gdp_growth]
        inf_str = [f"{i:+.1f}%" for i in scenario.inflation]
        print(f"{name:<15} {gdp_str[0]:<10} {gdp_str[1]:<10} {gdp_str[2]:<10} {inf_str[0]:<10} {inf_str[1]:<10} {inf_str[2]:<10}")


def print_probabilities(probabilities):
    """Print scenario probabilities"""
    print("\n3) SCENARIO PROBABILITIES")
    print("=" * 50)

    for scenario, prob in probabilities.items():
        print(f"{scenario:<15}: {prob:6.1%}")


def print_forecasts(forecasts):
    """Print asset return forecasts"""
    print("\n4) ASSET RETURN FORECASTS")
    print("=" * 50)

    for scenario, asset_forecasts in forecasts.items():
        print(f"\n{scenario}:")
        for asset, returns in asset_forecasts.items():
            returns_str = [f"{r:+.1f}%" for r in returns]
            print(f"  {asset:<8}: {returns_str[0]:<8} {returns_str[1]:<8} {returns_str[2]:<8}")


def print_optimization_header():
    """Print optimization section header"""
    print("\n5) OPTIMIZED PORTFOLIO")
    print("=" * 50)


def main():
    print("SEQUENTIAL BAYESIAN PORTFOLIO OPTIMIZATION")
    print("=" * 50)

    # Setup
    analyzer = ScenarioAnalyzer("data/usa_macro_var_and_asset_returns.csv")
    anchor_year = 2017

    # 1) Print dataset info
    print_dataset_info(analyzer)

    # 2) Print scenarios
    print_scenarios()

    # Get initial probabilities and forecasts
    probabilities = analyzer.estimate_probabilities(anchor_year)
    forecasts = analyzer.forecast_returns(anchor_year)

    # 3) Print probabilities
    print_probabilities(probabilities)

    # 4) Print forecasts
    print_forecasts(forecasts)

    # 5) Print optimization header
    print_optimization_header()

    # Extract actual 2018-2020 economic path from data
    actual_years = [2018, 2019, 2020]
    actual_gdp = []
    actual_inflation = []

    for year in actual_years:
        if year in analyzer.data.index:
            actual_gdp.append(analyzer.data.loc[year, 'gdp_growth'])
            actual_inflation.append(analyzer.data.loc[year, 'inflation'])
        else:
            print(f"Warning: Year {year} not found in data")

    print(f"Actual GDP growth: {actual_gdp}")
    print(f"Actual inflation: {actual_inflation}")

    # Extract actual asset returns for performance calculation from data
    actual_returns = {'Cash': [], 'Stocks': [], 'Bonds': []}

    for year in actual_years:
        if year in analyzer.data.index:
            # Cash returns: use bill_rate (nominal short-term rate)
            cash_return = analyzer.data.loc[year, 'bill_rate'] * 100
            actual_returns['Cash'].append(cash_return)

            # Stock returns: use eq_tr (equity total return)
            stock_return = analyzer.data.loc[year, 'eq_tr'] * 100
            actual_returns['Stocks'].append(stock_return)

            # Bond returns: use bond_tr (bond total return)
            bond_return = analyzer.data.loc[year, 'bond_tr'] * 100
            actual_returns['Bonds'].append(bond_return)
        else:
            print(f"Warning: Year {year} not found in data for returns")

    print(f"Actual returns: {actual_returns}")

    # Convert scenarios for learning
    scenarios_dict = {
        name: {
            'gdp_growth': scenario.gdp_growth,
            'inflation': scenario.inflation
        }
        for name, scenario in SCENARIOS.items()
    }

    # Get covariance matrix
    covariance_matrix = analyzer._calculate_covariance_matrix(anchor_year, 0)

    # Sequential optimization
    portfolio_weights = []
    current_beliefs = probabilities.copy()

    for year in range(3):
        print(f"\n--- YEAR {year + 1} ({2018 + year}) ---")

        # Get this year's return forecasts
        year_forecasts = {
            scenario: {asset: forecasts[scenario][asset][year]
                      for asset in forecasts[scenario].keys()}
            for scenario in forecasts.keys()
        }

        # Show current beliefs
        print("Current Beliefs:")
        for scenario, prob in current_beliefs.items():
            print(f"  {scenario:<15}: {prob:6.1%}")

        # Optimize portfolio
        weights = optimize_portfolio(current_beliefs, year_forecasts, year)
        portfolio_weights.append(weights)

        print("Optimal Weights:")
        for asset, weight in weights.items():
            print(f"  {asset:<8}: {weight:6.1%}")

        # Update beliefs for next year (except last year)
        if year < 2:
            observed_gdp = actual_gdp[:year+1]
            observed_inf = actual_inflation[:year+1]

            print(f"Observed Evidence: GDP={observed_gdp}, INF={observed_inf}")

            current_beliefs = update_beliefs(
                current_beliefs, observed_gdp, observed_inf,
                scenarios_dict, covariance_matrix
            )

    # Calculate performance
    portfolio_value = 1.0
    print(f"\n--- PERFORMANCE ANALYSIS ---")
    print("=" * 30)

    for year, weights in enumerate(portfolio_weights):
        # Annual portfolio return
        portfolio_return = sum(
            weights[asset] * actual_returns[asset][year] / 100.0
            for asset in weights.keys()
        )
        portfolio_value *= (1 + portfolio_return)

        print(f"Year {year+1} ({2018+year}): {portfolio_return*100:+.2f}%")

    print(f"\nFinal Portfolio Value: ${portfolio_value:.3f}")
    print(f"Cumulative Return: {(portfolio_value-1)*100:+.2f}%")


if __name__ == "__main__":
    main()

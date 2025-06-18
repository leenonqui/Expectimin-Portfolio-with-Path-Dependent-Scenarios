"""
Basic usage example for the GIC methodology package with CRRA utility

Shows simple workflow for:
1. Loading data and running analysis
2. Getting results
3. Basic portfolio optimization with different utility functions
"""

from expectiminimax_portfolio import GICAnalyzer, ExpectiminimaxOptimizer

def basic_example():
    """Simple example of package usage with CRRA utility"""

    print("Basic GIC Methodology Example with CRRA Utility")
    print("="*50)

    # Step 1: Initialize analyzer
    analyzer = GICAnalyzer("data/usa_macro_var_and_asset_returns.csv")

    # Step 2: Run analysis
    print("Running GIC analysis...")
    results = analyzer.analyze(prediction_year=2020)

    # Step 3: View scenario probabilities
    print("\nScenario Probabilities:")
    for scenario, prob in results.scenario_probabilities.items():
        print(f"  {scenario}: {prob:.1%}")

    # Step 4: Portfolio optimization with different utility functions
    print("\nOptimizing portfolios...")
    optimizer = ExpectiminimaxOptimizer(
        scenario_probabilities=results.scenario_probabilities,
        asset_returns=results.asset_returns
    )

    # Test both utility functions
    utility_types = ["crra", "mean_variance"]

    for utility_type in utility_types:
        print(f"\n{utility_type.upper()} UTILITY FUNCTION:")

        # Moderate risk aversion
        risk_aversion = 1.0 if utility_type == "crra" else 2.0

        portfolio = optimizer.optimize_single_profile(
            risk_aversion=risk_aversion,
            utility_type=utility_type
        )

        print(f"  Risk Aversion (γ or λ): {portfolio.risk_aversion}")
        print(f"  Optimal Portfolio:")
        for asset, weight in portfolio.optimal_weights.items():
            print(f"    {asset}: {weight:.1%}")

        print(f"  Expected Utility: {portfolio.expectiminimax_value:.4f}")
        print(f"  Expected Cumulative Return: {portfolio.expected_return:.2f}%")
        print(f"  Expected Volatility: {portfolio.expected_volatility:.2f}%")

    # Compare risk aversion levels with CRRA
    print(f"\nCRRA UTILITY - RISK AVERSION COMPARISON:")
    risk_levels = [0.0, 0.5, 1.0, 2.0, 4.0]

    for gamma in risk_levels:
        portfolio = optimizer.optimize_single_profile(
            risk_aversion=gamma,
            utility_type="crra"
        )

        cash = portfolio.optimal_weights['Cash']
        stocks = portfolio.optimal_weights['Stocks']
        bonds = portfolio.optimal_weights['Bonds']

        print(f"  γ={gamma:3.1f}: Cash={cash:5.1%}, Stocks={stocks:5.1%}, Bonds={bonds:5.1%}")

if __name__ == "__main__":
    basic_example()

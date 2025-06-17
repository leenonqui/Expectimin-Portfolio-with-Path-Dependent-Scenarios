"""
Basic usage example for the GIC methodology package

Shows simple workflow for:
1. Loading data and running analysis
2. Getting results
3. Basic portfolio optimization
"""

from expectiminimax_portfolio import GICAnalyzer, ExpectiminimaxOptimizer

def basic_example():
    """Simple example of package usage"""

    print("Basic GIC Methodology Example")
    print("="*40)

    # Step 1: Initialize analyzer
    analyzer = GICAnalyzer("data/usa_macro_var_and_asset_returns.csv")

    # Step 2: Run analysis
    print("Running GIC analysis...")
    results = analyzer.analyze(prediction_year=2020)

    # Step 3: View scenario probabilities
    print("\nScenario Probabilities:")
    for scenario, prob in results.scenario_probabilities.items():
        print(f"  {scenario}: {prob:.1%}")

    # Step 4: Portfolio optimization
    print("\nOptimizing portfolios...")
    optimizer = ExpectiminimaxOptimizer(
        scenario_probabilities=results.scenario_probabilities,
        asset_returns=results.asset_returns
    )

    # Optimize for moderate risk aversion
    portfolio = optimizer.optimize_single_profile(risk_aversion=1.0)

    print(f"\nOptimal Portfolio (λ=1.0):")
    for asset, weight in portfolio.optimal_weights.items():
        print(f"  {asset}: {weight:.1%}")

    print(f"\nExpected Utility: {portfolio.expectiminimax_value:.4f}")
    print(f"Expected Return: {portfolio.expected_return:.2f}% (over 3 years)")
    print(f"Expected Volatility: {portfolio.expected_volatility:.2f}%")

if __name__ == "__main__":
    basic_example()

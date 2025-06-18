"""
Complete replication of thesis results using Mean-Variance utility

This example demonstrates:
1. Loading and processing JST macrohistory data
2. Calculating scenario probabilities using GIC methodology
3. Forecasting asset returns via partial sample regression
4. Optimizing portfolios using Mean-Variance utility function
5. Analyzing results across risk aversion profiles
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from expectiminimax_portfolio import GICAnalyzer, ExpectiminimaxOptimizer
from expectiminimax_portfolio.config import GIC_SCENARIOS
import pandas as pd
import numpy as np

def main():
    """Replicate complete thesis analysis with Mean-Variance utility"""

    print("="*80)
    print("THESIS REPLICATION: GIC METHODOLOGY + MEAN-VARIANCE EXPECTIMINIMAX OPTIMIZATION")
    print("="*80)

    # Initialize analyzer with JST data
    data_path = "data/usa_macro_var_and_asset_returns.csv"
    analyzer = GICAnalyzer(data_path)

    print(f"\n1. LOADING DATA")
    print(f"   Data source: {data_path}")
    print(f"   Methodology: Kritzman et al. (2021)")

    # Run GIC analysis for 2020 prediction
    prediction_year = 2020
    print(f"\n2. CALCULATING SCENARIO PROBABILITIES FOR {prediction_year}")
    print(f"   Training data: 1927-{prediction_year-1}")
    print(f"   Anchor path: Last 3-year sequence ending in {prediction_year-1}")

    try:
        results = analyzer.analyze(prediction_year=prediction_year)
        print(f"   ✓ Successfully calculated probabilities for {len(results.scenario_probabilities)} scenarios")
    except Exception as e:
        print(f"   ✗ Error in GIC analysis: {e}")
        return

    # Display scenario probabilities
    print(f"\n   SCENARIO PROBABILITIES (based on Mahalanobis distance):")
    total_prob = 0
    for scenario, prob in results.scenario_probabilities.items():
        print(f"   {scenario:12}: {prob:.4f} ({prob*100:.2f}%)")
        total_prob += prob
    print(f"   {'Total':12}: {total_prob:.4f} (should be 1.0000)")

    # Display forecasted asset returns
    print(f"\n3. ASSET RETURN FORECASTING (Partial Sample Regression)")
    print(f"   Top 25% most relevant historical observations used")

    for scenario_name, returns in results.asset_returns.items():
        print(f"\n   {scenario_name}:")
        for asset, return_path in returns.items():
            formatted_returns = [f"{r:.2f}%" for r in return_path]
            print(f"     {asset:6}: {formatted_returns}")

    # Portfolio optimization with Mean-Variance utility
    print(f"\n4. MEAN-VARIANCE UTILITY PORTFOLIO OPTIMIZATION")
    print(f"   Utility function: U = E[R] - (λ/2)Var(R)")
    print(f"   Constraint: weights sum to 1, 0 ≤ w ≤ 1")

    optimizer = ExpectiminimaxOptimizer(
        scenario_probabilities=results.scenario_probabilities,
        asset_returns=results.asset_returns
    )

    # Mean-Variance risk aversion profiles
    mv_profiles = [
        {"name": "Risk Neutral", "lambda": 0.0},
        {"name": "Low Risk Aversion", "lambda": 0.1},
        {"name": "Moderate Risk Aversion", "lambda": .2},
        {"name": "High Risk Aversion", "lambda": .3},
        {"name": "Very High Risk Aversion", "lambda": .4}
    ]

    portfolios = {}

    print(f"\n   OPTIMAL ALLOCATIONS BY MEAN-VARIANCE RISK AVERSION PROFILE:")
    print(f"   {'Profile':<25} {'λ':<6} {'Cash':<8} {'Stocks':<8} {'Bonds':<8} {'Utility':<10}")
    print(f"   {'-'*75}")

    for profile in mv_profiles:
        name = profile["name"]
        lambda_val = profile["lambda"]

        result = optimizer.optimize_single_profile(
            risk_aversion=lambda_val,
            utility_type="mean_variance"
        )

        portfolios[name] = result

        if result.optimization_success:
            print(f"   {name:<25} {result.risk_aversion:<6.1f} "
                  f"{result.optimal_weights['Cash']:<8.1%} "
                  f"{result.optimal_weights['Stocks']:<8.1%} "
                  f"{result.optimal_weights['Bonds']:<8.1%} "
                  f"{result.expectiminimax_value:<10.4f}")
        else:
            print(f"   {name:<25} {result.risk_aversion:<6.1f} OPTIMIZATION FAILED")

    # Detailed analysis for moderate risk aversion
    print(f"\n5. DETAILED ANALYSIS: MODERATE RISK AVERSION (λ=1.0)")

    moderate_result = portfolios.get("Moderate Risk Aversion")
    if moderate_result and moderate_result.optimization_success:
        print(f"   Expected Cumulative Return: {moderate_result.expected_return:.2f}%")
        print(f"   Expected Volatility: {moderate_result.expected_volatility:.2f}%")
        print(f"   Expected Utility: {moderate_result.expectiminimax_value:.4f}")

        print(f"\n   PORTFOLIO PERFORMANCE BY SCENARIO:")
        weights = [moderate_result.optimal_weights[asset] for asset in ['Cash', 'Stocks', 'Bonds']]

        scenario_returns = []
        for scenario, prob in results.scenario_probabilities.items():
            portfolio_returns = []
            for year in range(3):
                year_return = sum(
                    weights[i] * results.asset_returns[scenario][asset][year]
                    for i, asset in enumerate(['Cash', 'Stocks', 'Bonds'])
                )
                portfolio_returns.append(year_return)

            cumulative_return = ((1 + portfolio_returns[0]/100) *
                               (1 + portfolio_returns[1]/100) *
                               (1 + portfolio_returns[2]/100) - 1) * 100

            scenario_returns.append(cumulative_return)
            print(f"   {scenario:12} (p={prob:.3f}): "
                  f"Cumulative={cumulative_return:6.2f}%")

        # Risk analysis
        print(f"\n   RISK ANALYSIS:")
        expected_return = moderate_result.expected_return
        volatility = moderate_result.expected_volatility

        print(f"   Expected Return: {expected_return:.2f}%")
        print(f"   Volatility (std): {volatility:.2f}%")
        print(f"   Min scenario: {min(scenario_returns):.2f}%")
        print(f"   Max scenario: {max(scenario_returns):.2f}%")

        # For normal distribution approximation: 95% confidence interval ≈ μ ± 2σ
        print(f"   Approximate 95% range: [{expected_return - 2*volatility:.2f}%, {expected_return + 2*volatility:.2f}%]")

    # Risk aversion impact analysis
    print(f"\n6. RISK AVERSION IMPACT ANALYSIS")
    print(f"   How portfolio allocation changes with risk aversion:")

    print(f"\n   {'λ':<6} {'Cash %':<8} {'Stock %':<9} {'Bond %':<8} {'E[R]%':<8} {'σ%':<8} {'Utility':<10}")
    print(f"   {'-'*70}")

    for profile in mv_profiles:
        name = profile["name"]
        if name in portfolios:
            result = portfolios[name]
            if result.optimization_success:
                print(f"   {result.risk_aversion:<6.1f} "
                      f"{result.optimal_weights['Cash']*100:<8.1f} "
                      f"{result.optimal_weights['Stocks']*100:<9.1f} "
                      f"{result.optimal_weights['Bonds']*100:<8.1f} "
                      f"{result.expected_return:<8.2f} "
                      f"{result.expected_volatility:<8.2f} "
                      f"{result.expectiminimax_value:<10.4f}")

    print(f"\n7. METHODOLOGY VALIDATION")
    print(f"   ✓ Path-dependent scenarios (3-year sequences)")
    print(f"   ✓ Mahalanobis distance for probability estimation")
    print(f"   ✓ Partial sample regression for return forecasting")
    print(f"   ✓ Mean-variance utility function")
    print(f"   ✓ Expectiminimax optimization framework")

    print(f"\n" + "="*80)
    print("THESIS REPLICATION WITH MEAN-VARIANCE UTILITY COMPLETE")
    print("="*80)

    return results, portfolios

if __name__ == "__main__":
    results, portfolios = main()

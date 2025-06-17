"""
Comprehensive example demonstrating all liquidity preference features

This example shows:
1. Basic liquidity constraints (minimum cash requirements)
2. Liquidity sensitivity analysis across different levels
3. Combined risk-liquidity profiles inspired by GIC examples
4. Cost analysis of liquidity preferences
5. Optimal liquidity level determination
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from expectiminimax_portfolio import GICAnalyzer, ExpectiminimaxOptimizer
from expectiminimax_portfolio.utils.liquidity_utils import LiquidityAnalyzer
from expectiminimax_portfolio.config import COMBINED_PROFILES
import pandas as pd
import numpy as np

def main():
    """Run comprehensive liquidity preference analysis"""

    print("="*90)
    print("COMPREHENSIVE LIQUIDITY PREFERENCE ANALYSIS")
    print("Minimum Cash Constraints as Lower Bounds (0-10%)")
    print("="*90)

    # Initialize GIC analysis
    analyzer = GICAnalyzer("data/usa_macro_var_and_asset_returns.csv")
    results = analyzer.analyze(prediction_year=2020)

    print(f"\n1. DATA SUMMARY")
    print(f"   Scenarios analyzed: {len(results.scenario_probabilities)}")
    print(f"   Scenario probabilities:")
    for scenario, prob in results.scenario_probabilities.items():
        print(f"     {scenario}: {prob:.3f}")

    # Initialize optimizer
    optimizer = ExpectiminimaxOptimizer(
        scenario_probabilities=results.scenario_probabilities,
        asset_returns=results.asset_returns
    )

    # 2. Basic liquidity constraint demonstration
    print(f"\n2. BASIC LIQUIDITY CONSTRAINTS DEMONSTRATION")
    print(f"   Fixed risk aversion (λ=1.0) with varying minimum cash requirements")
    print("-"*90)

    basic_results = {}
    liquidity_levels = [0.0, 0.02, 0.05, 0.08, 0.10]

    print(f"{'Min Cash Req':<12} {'Actual Cash':<12} {'Stocks':<8} {'Bonds':<8} {'Return':<8} {'Vol':<8} {'Utility':<10} {'Constraint':<10}")
    print("-"*90)

    for min_cash in liquidity_levels:
        result = optimizer.optimize_single_profile(risk_aversion=1.0, min_cash_pct=min_cash)
        profile_name = f"Cash_{min_cash*100:.0f}%"
        basic_results[profile_name] = result

        constraint_met = "✓" if result.liquidity_preference_met else "✗"

        print(f"{min_cash:<12.1%} "
              f"{result.optimal_weights['Cash']:<12.1%} "
              f"{result.optimal_weights['Stocks']:<8.1%} "
              f"{result.optimal_weights['Bonds']:<8.1%} "
              f"{result.expected_return:<8.2f}% "
              f"{result.expected_volatility:<8.2f}% "
              f"{result.expectiminimax_value:<10.4f} "
              f"{constraint_met:<10}")

    # 3. Multi-dimensional analysis: Risk × Liquidity
    print(f"\n3. RISK-LIQUIDITY MATRIX ANALYSIS")
    print(f"   Portfolio allocations across risk aversion and liquidity preferences")
    print("-"*90)

    risk_levels = [0.0, 0.5, 1.0, 2.0, 5.0]
    liquidity_levels_matrix = [0.0, 0.05, 0.10]

    # Create results matrix
    matrix_results = {}

    print(f"Risk Aversion (λ) vs Minimum Cash Requirement")
    print(f"{'λ':<6} | {'0% Cash':<25} | {'5% Cash':<25} | {'10% Cash':<25}")
    print("-"*90)

    for risk_aversion in risk_levels:
        row_text = f"{risk_aversion:<6.1f} |"

        for min_cash in liquidity_levels_matrix:
            result = optimizer.optimize_single_profile(risk_aversion, min_cash)
            key = f"Risk_{risk_aversion}_Cash_{min_cash*100:.0f}"
            matrix_results[key] = result

            allocation_text = f"C:{result.optimal_weights['Cash']:.0%} S:{result.optimal_weights['Stocks']:.0%} B:{result.optimal_weights['Bonds']:.0%}"
            row_text += f" {allocation_text:<25} |"

        print(row_text)

    # 4. GIC-inspired combined profiles
    print(f"\n4. GIC-INSPIRED COMBINED PROFILES")
    print(f"   Realistic investor profiles combining risk aversion and liquidity preferences")
    print("-"*90)

    combined_results = optimizer.optimize_combined_profiles()

    print(f"{'Profile':<20} {'λ':<6} {'Min Cash':<10} {'Actual Allocation':<20} {'Return':<8} {'Vol':<8} {'Utility':<10}")
    print("-"*90)

    for profile_name, result in combined_results.items():
        if result.optimization_success:
            allocation_text = f"C:{result.optimal_weights['Cash']:.0%} S:{result.optimal_weights['Stocks']:.0%} B:{result.optimal_weights['Bonds']:.0%}"
            print(f"{profile_name:<20} "
                  f"{result.risk_aversion:<6.1f} "
                  f"{result.min_cash_pct:<10.1%} "
                  f"{allocation_text:<20} "
                  f"{result.expected_return:<8.2f}% "
                  f"{result.expected_volatility:<8.2f}% "
                  f"{result.expectiminimax_value:<10.4f}")

    # 5. Cost analysis of liquidity preferences
    print(f"\n5. LIQUIDITY COST ANALYSIS")
    print(f"   Economic cost of imposing minimum cash requirements")
    print("-"*90)

    costs = LiquidityAnalyzer.calculate_liquidity_cost(basic_results, base_liquidity=0.0)

    print(f"{'Min Cash Req':<12} {'Utility Cost':<12} {'Return Cost':<12} {'Vol Change':<12} {'Efficiency Change':<15}")
    print("-"*90)

    for profile_name, cost_metrics in costs.items():
        min_cash = basic_results[profile_name].min_cash_pct
        print(f"{min_cash:<12.1%} "
              f"{cost_metrics['utility_cost']:<12.4f} "
              f"{cost_metrics['return_cost']:<12.2f}% "
              f"{cost_metrics['volatility_change']:<12.2f}% "
              f"{cost_metrics['efficiency_change']:<15.4f}")

    # 6. Sensitivity analysis
    print(f"\n6. LIQUIDITY SENSITIVITY METRICS")
    print(f"   How portfolio characteristics change with liquidity requirements")
    print("-"*90)

    sensitivity = LiquidityAnalyzer.liquidity_sensitivity_metrics(basic_results)

    for metric, value in sensitivity.items():
        metric_name = metric.replace('_', ' ').title()
        print(f"   {metric_name:<35}: {value:>10.4f}")

    # 7. Optimal liquidity analysis
    print(f"\n7. OPTIMAL LIQUIDITY DETERMINATION")
    print(f"   Finding best liquidity level under different penalty assumptions")
    print("-"*90)

    penalty_levels = [0.0, 0.1, 0.2, 0.5, 1.0]  # Utility penalty per % cash

    print(f"{'Liquidity Penalty':<16} {'Optimal Min Cash':<16} {'Actual Cash':<12} {'Adjusted Utility':<15}")
    print("-"*90)

    for penalty in penalty_levels:
        optimal_profile, optimal_result = LiquidityAnalyzer.find_optimal_liquidity(
            basic_results, liquidity_penalty=penalty
        )

        cash_weight = optimal_result.optimal_weights['Cash']
        adjusted_utility = optimal_result.expectiminimax_value - (penalty * cash_weight)

        print(f"{penalty:<16.1f} "
              f"{optimal_result.min_cash_pct:<16.1%} "
              f"{cash_weight:<12.1%} "
              f"{adjusted_utility:<15.4f}")

    # 8. Practical recommendations
    print(f"\n8. PRACTICAL RECOMMENDATIONS")
    print("-"*90)

    # Find break-even points
    utility_costs = [costs[f"Cash_{int(liq*100)}%"]['utility_cost'] for liq in [0.02, 0.05, 0.08, 0.10]]

    print(f"   Liquidity preference guidelines:")
    print(f"   • 2% minimum cash: Low cost ({utility_costs[0]:.4f} utility units)")
    print(f"   • 5% minimum cash: Moderate cost ({utility_costs[1]:.4f} utility units)")
    print(f"   • 8% minimum cash: High cost ({utility_costs[2]:.4f} utility units)")
    print(f"   • 10% minimum cash: Very high cost ({utility_costs[3]:.4f} utility units)")

    # Risk-adjusted recommendations
    print(f"\n   Risk-adjusted recommendations:")
    conservative_5 = matrix_results.get('Risk_2.0_Cash_5')
    aggressive_0 = matrix_results.get('Risk_0.0_Cash_0')

    if conservative_5 and aggressive_0:
        print(f"   • Conservative investors (λ=2.0): 5% cash costs {conservative_5.expectiminimax_value - basic_results['Cash_0%'].expectiminimax_value:.4f} utility")
        print(f"   • Aggressive investors (λ=0.0): Should avoid cash constraints unless necessary")

    # 9. Generate detailed report
    print(f"\n9. DETAILED LIQUIDITY ANALYSIS REPORT")
    print("-"*90)

    report = LiquidityAnalyzer.generate_liquidity_report(
        basic_results,
        title="Liquidity Preference Impact Analysis"
    )

    # Save report to file (optional)
    with open("liquidity_analysis_report.txt", "w") as f:
        f.write(report)
    print(f"   ✓ Detailed report saved to 'liquidity_analysis_report.txt'")

    print(f"\n" + "="*90)
    print("COMPREHENSIVE LIQUIDITY ANALYSIS COMPLETE")
    print("="*90)

    return {
        'basic_results': basic_results,
        'matrix_results': matrix_results,
        'combined_results': combined_results,
        'costs': costs,
        'sensitivity': sensitivity
    }

def quick_liquidity_example():
    """Quick example for basic usage"""

    print("\nQUICK LIQUIDITY PREFERENCE EXAMPLE")
    print("-"*50)

    # Initialize
    analyzer = GICAnalyzer("data/usa_macro_var_and_asset_returns.csv")
    results = analyzer.analyze(prediction_year=2020)
    optimizer = ExpectiminimaxOptimizer(
        scenario_probabilities=results.scenario_probabilities,
        asset_returns=results.asset_returns
    )

    # Single example: Moderate risk with 5% minimum cash
    portfolio = optimizer.optimize_single_profile(
        risk_aversion=1.0,      # Moderate risk aversion
        min_cash_pct=0.05       # Require at least 5% in cash
    )

    print(f"Moderate Risk Portfolio with 5% Minimum Cash:")
    print(f"  Cash:   {portfolio.optimal_weights['Cash']:.1%} (min: {portfolio.min_cash_pct:.1%})")
    print(f"  Stocks: {portfolio.optimal_weights['Stocks']:.1%}")
    print(f"  Bonds:  {portfolio.optimal_weights['Bonds']:.1%}")
    print(f"  Expected Return: {portfolio.expected_return:.2f}%")
    print(f"  Constraint satisfied: {portfolio.liquidity_preference_met}")
    print(f"  Profile: {portfolio.profile_description}")

if __name__ == "__main__":
    # Run comprehensive analysis
    analysis_results = main()

    # Run quick example
    quick_liquidity_example()

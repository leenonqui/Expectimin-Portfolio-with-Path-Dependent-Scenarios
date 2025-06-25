"""
main_thesis_analysis.py
Main execution script for Game Theory Portfolio Selection thesis analysis
"""

import os
import pandas as pd
from typing import Optional, List
import json

from portfolio_analysis import GameTheoryPortfolioAnalyzer
from constants import HORIZON


def save_detailed_results(results: dict, output_dir: str = "thesis_game_theory_results"):
    """Save comprehensive results to CSV files using corrected table structure"""

    os.makedirs(output_dir, exist_ok=True)

    # Save the three main tables as specified
    detailed_tables = results['detailed_tables']

    # Table 1: Preset Portfolios Analysis
    detailed_tables['preset_portfolios_table'].to_csv(
        f"{output_dir}/table1_preset_portfolios_analysis.csv", index=False
    )

    # Table 2: Portfolio Spread Analysis
    detailed_tables['portfolio_spread_table'].to_csv(
        f"{output_dir}/table2_portfolio_spread_analysis.csv", index=False
    )

    # Table 3: Comprehensive Comparison
    detailed_tables['comprehensive_comparison_table'].to_csv(
        f"{output_dir}/table3_comprehensive_comparison.csv", index=False
    )

    # Save initial probabilities
    prob_df = pd.DataFrame([results['initial_probabilities']])
    prob_df.to_csv(f"{output_dir}/initial_scenario_probabilities.csv", index=False)

    # Save optimized portfolios details for each learning mechanism
    optimized_details = []
    for mechanism_name, data in results['optimized_portfolios'].items():
        metrics = data['metrics']
        for scenario, cum_return in metrics.cumulative_returns.items():
            optimized_details.append({
                'Learning_Mechanism': mechanism_name,
                'Scenario': scenario,
                'Probability_%': results['initial_probabilities'][scenario] * 100,
                'Cumulative_Return_%': cum_return * 100,
                'Max_Drawdown_%': metrics.max_drawdowns[scenario] * 100,
                'Within_Horizon_Loss_%': metrics.within_horizon_losses[scenario] * 100,
                'Worst_Annual_Loss_%': metrics.worst_annual_losses[scenario] * 100,
                'Num_Annual_Losses': metrics.annual_loss_counts[scenario]
            })

    optimized_df = pd.DataFrame(optimized_details)
    optimized_df.to_csv(f"{output_dir}/optimized_portfolios_details.csv", index=False)

    print(f"\n💾 Results saved to '{output_dir}/' directory:")
    print(f"   • table1_preset_portfolios_analysis.csv - Main preset portfolio table")
    print(f"   • table2_portfolio_spread_analysis.csv - Portfolio risk spread table")
    print(f"   • table3_comprehensive_comparison.csv - Preset vs Optimized comparison")
    print(f"   • optimized_portfolios_details.csv - Learning mechanism details")
    print(f"   • initial_scenario_probabilities.csv - Scenario probabilities")


def run_thesis_analysis(data_path: str = "data/usa_macro_var_and_asset_returns.csv",
                       anchor_year: int = 2019,
                       observed_gdp: Optional[List[float]] = None,
                       observed_inflation: Optional[List[float]] = None,
                       save_results: bool = True) -> dict:
    """
    Run complete thesis analysis

    Args:
        data_path: Path to historical data
        anchor_year: Anchor year for probability estimation
        observed_gdp: Actual observed GDP growth path (optional)
        observed_inflation: Actual observed inflation path (optional)
        save_results: Whether to save results to files

    Returns:
        Complete analysis results
    """

    print("🎓 BACHELOR THESIS: GAME THEORY PORTFOLIO SELECTION")
    print("Expectimin Loss Portfolio with Learning Mechanisms")
    print("="*80)
    print(f"Data source: {data_path}")
    print(f"Anchor year: {anchor_year}")
    print(f"Investment horizon: {HORIZON} years")

    # Check data file
    if not os.path.exists(data_path):
        print(f"\n❌ ERROR: Data file '{data_path}' not found!")
        return None

    # Initialize analyzer
    analyzer = GameTheoryPortfolioAnalyzer(data_path, anchor_year)

    print(f"\n✅ Analysis initialized")
    print(f"📊 Initial scenario probabilities:")
    for scenario, prob in analyzer.initial_probabilities.items():
        print(f"   {scenario:15}: {prob:6.1%}")

    # Run analysis
    try:
        results = analyzer.run_complete_analysis(observed_gdp, observed_inflation)

        # Save results if requested
        if save_results:
            save_detailed_results(results)

        print(f"\n{'THESIS ANALYSIS COMPLETED SUCCESSFULLY':=^80}")
        print("🎉 All portfolio types analyzed")
        print("📈 Learning mechanisms compared")
        print("📊 Risk metrics calculated")
        print("💾 Results saved for thesis writing")

        return results

    except Exception as e:
        print(f"\n❌ ANALYSIS FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_with_actual_2018_2020_data():
    """Run analysis using actual 2018-2020 economic data"""

    print("🎯 RUNNING ANALYSIS WITH ACTUAL 2018-2020 DATA")
    print("="*60)

    # Actual observed data (approximate values)
    observed_gdp = [2.9, 2.2, -3.4]      # 2018, 2019, 2020 GDP growth
    observed_inflation = [2.4, 1.8, 1.2]  # 2018, 2019, 2020 inflation

    print(f"Observed GDP growth: {observed_gdp}")
    print(f"Observed inflation: {observed_inflation}")

    results = run_thesis_analysis(
        anchor_year=2017,  # Train up to 2017, predict 2018-2020
        observed_gdp=observed_gdp,
        observed_inflation=observed_inflation,
        save_results=True
    )

    return results


def run_scenario_analysis():
    """Run analysis using most likely scenario for theoretical analysis"""

    print("📊 RUNNING THEORETICAL SCENARIO ANALYSIS")
    print("="*50)
    print("Using most likely scenario path for portfolio comparison")

    results = run_thesis_analysis(
        anchor_year=2019,
        observed_gdp=None,  # Will use most likely scenario
        observed_inflation=None,
        save_results=True
    )

    return results


def main():
    """Main execution function"""

    print("🎓 BACHELOR THESIS ANALYSIS")
    print("Game Theory Applied to Portfolio Selection")
    print(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*80)

    # Option 1: Run with actual historical data
    print("Option 1: Analysis with actual 2018-2020 data")
    results_actual = run_with_actual_2018_2020_data()

    print("\n" + "="*80)

    # Option 2: Run theoretical scenario analysis
    print("Option 2: Theoretical scenario analysis")
    results_theoretical = run_scenario_analysis()

    print(f"\n{'THESIS ANALYSIS COMPLETE':=^80}")

    if results_actual or results_theoretical:
        print("🎉 Analysis completed successfully!")
        print("📁 Check 'thesis_game_theory_results/' for detailed results")
        print("📊 Use the CSV files for your thesis tables and charts")

        print(f"\n📋 Next steps for your thesis:")
        print(f"   1. Analyze the portfolio comparison results")
        print(f"   2. Discuss learning mechanism effectiveness")
        print(f"   3. Compare preset vs optimized portfolio performance")
        print(f"   4. Write conclusions about game theory applications")

    else:
        print("❌ Analysis failed - check error messages above")

    return results_actual, results_theoretical


if __name__ == "__main__":
    results = main()

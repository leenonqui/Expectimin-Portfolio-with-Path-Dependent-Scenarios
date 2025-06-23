"""
Main Analysis for Bachelor Thesis:
"Expectimin Optimal Portfolio with Scenario-Dependent Paths"

Clean pipeline:
1. Scenario probability estimation using Mahalanobis distance
2. Asset return forecasting using partial sample regression
3. Portfolio optimization minimizing expected cumulative loss
"""

import os
import pandas as pd
from typing import Dict, List

from constants import SCENARIOS, ASSET_CLASSES, HORIZON
from scenario_analysis import ScenarioAnalyzer
from portfolio_optimization import LinearExpectiminOptimizer


def run_bachelor_thesis_analysis(data_path: str,
                                anchor_year: int = 2019,
                                min_required_returns: List[float] = None,
                                save_results: bool = True) -> Dict:
    """
    Complete bachelor thesis analysis pipeline.

    Args:
        data_path: Path to historical economic data CSV
        anchor_year: Anchor year for scenario analysis
        min_required_returns: Different minimum return requirements to test
        save_results: Whether to save results to files

    Returns:
        Complete analysis results
    """
    if min_required_returns is None:
        min_required_returns = [0.02, 0.04, 0.06]  # 2%, 4%, 6% minimum returns

    print(f"BACHELOR THESIS ANALYSIS")
    print(f"Expectimin Optimal Portfolio with Scenario-Dependent Paths")
    print(f"{'='*80}")
    print(f"Data source: {data_path}")
    print(f"Anchor year: {anchor_year}")
    print(f"Investment horizon: {HORIZON} years")
    print(f"Minimum return requirements: {[f'{ret:.0%}' for ret in min_required_returns]}")
    print(f"Scenarios analyzed: {len(SCENARIOS)}")
    print(f"Asset classes: {', '.join(ASSET_CLASSES)}")

    results = {
        'config': {
            'data_path': data_path,
            'anchor_year': anchor_year,
            'min_required_returns': min_required_returns,
            'scenarios': list(SCENARIOS.keys()),
            'asset_classes': ASSET_CLASSES,
            'investment_horizon': HORIZON
        }
    }

    try:
        # ================================================================
        # STEP 1: ESTIMATE SCENARIO PROBABILITIES
        # ================================================================

        print(f"\n{'STEP 1: SCENARIO PROBABILITY ESTIMATION':=^80}")

        analyzer = ScenarioAnalyzer(data_path, horizon=HORIZON)
        probabilities = analyzer.estimate_probabilities(anchor_year)

        print(f"\nEstimated Scenario Probabilities:")
        print(f"{'-'*50}")
        for scenario, prob in probabilities.items():
            print(f"  {scenario:15}: {prob:6.1%}")
        print(f"{'-'*50}")
        print(f"  {'Total':15}: {sum(probabilities.values()):6.1%}")

        most_likely = max(probabilities.items(), key=lambda x: x[1])
        least_likely = min(probabilities.items(), key=lambda x: x[1])

        print(f"\nKey Findings:")
        print(f"  • Most likely scenario: {most_likely[0]} ({most_likely[1]:.1%})")
        print(f"  • Least likely scenario: {least_likely[0]} ({least_likely[1]:.1%})")

        results['probabilities'] = probabilities

        # ================================================================
        # STEP 2: FORECAST ASSET RETURNS
        # ================================================================

        print(f"\n{'STEP 2: ASSET RETURN FORECASTING':=^80}")

        forecasts = analyzer.forecast_returns(anchor_year)

        print(f"\nAsset Return Forecasts (Annual %):")
        print(f"{'-'*80}")

        header = f"{'Scenario':<15} {'Prob':<8} {'Year':<6}"
        for asset in ASSET_CLASSES:
            header += f"{asset:>10}"
        print(header)
        print(f"{'-'*80}")

        for scenario_name, asset_returns in forecasts.items():
            prob = probabilities[scenario_name]

            for year in range(HORIZON):
                year_label = f"Year {year+1}" if year == 0 else f"Year {year+1}"
                prob_label = f"{prob:>6.1%}" if year == 0 else ""
                scenario_label = scenario_name if year == 0 else ""

                row = f"{scenario_label:<15} {prob_label:<8} {year_label:<6}"
                for asset in ASSET_CLASSES:
                    return_val = asset_returns[asset][year]
                    row += f"{return_val:>9.1f}%"
                print(row)
            print()

        results['forecasts'] = forecasts

        # ================================================================
        # STEP 3: OPTIMIZE PORTFOLIO
        # ================================================================

        print(f"\n{'STEP 3: PORTFOLIO OPTIMIZATION':=^80}")

        optimizer = LinearExpectiminOptimizer(ASSET_CLASSES)
        optimization_results = {}

        for min_return in min_required_returns:
            print(f"\nOptimizing for minimum required return: {min_return:.0%}")

            result = optimizer.optimize_expectimin_cumulative_loss(
                scenario_forecasts=forecasts,
                probabilities=probabilities,
                min_return=min_return
            )

            optimization_results[min_return] = result

            if result['success']:
                print(f"  ✅ Expected Loss: {result['expected_cumulative_loss']:.4f} ({result['expected_cumulative_loss']*100:.2f}%)")
                print(f"  ✅ Expected Return: {result['expected_cumulative_return']:.4f} ({result['expected_cumulative_return']*100:.2f}%)")
                print(f"  ✅ Probability of Loss: {result['probability_of_loss']:.1%}")
                print(f"  ✅ Weights: {', '.join(f'{asset}={weight:.1%}' for asset, weight in result['weights'].items())}")
            else:
                print(f"  ❌ Optimization failed: {result['message']}")

        results['optimization_results'] = optimization_results

        # ================================================================
        # RESULTS SUMMARY
        # ================================================================

        print(f"\n{'THESIS RESULTS SUMMARY':=^80}")

        successful_results = {k: v for k, v in optimization_results.items() if v['success']}

        if successful_results:
            print(f"\nOptimal Portfolio Allocations:")
            print(f"{'-'*100}")

            header = f"{'Min Return':<12} {'Expected Loss':<15} {'Expected Return':<16} {'Prob of Loss':<13}"
            for asset in ASSET_CLASSES:
                header += f"{asset:>12}"
            print(header)
            print(f"{'-'*100}")

            for min_return, result in successful_results.items():
                row = f"{min_return:>10.0%} "
                row += f"{result['expected_cumulative_loss']*100:>13.2f}% "
                row += f"{result['expected_cumulative_return']*100:>14.2f}% "
                row += f"{result['probability_of_loss']*100:>11.1f}% "

                for asset in ASSET_CLASSES:
                    weight = result['weights'][asset]
                    row += f"{weight:>11.1%}"
                print(row)

            print(f"{'-'*100}")

            # Key insights
            min_loss_result = min(successful_results.items(), key=lambda x: x[1]['expected_cumulative_loss'])
            max_return_result = max(successful_results.items(), key=lambda x: x[1]['expected_cumulative_return'])

            print(f"\nKey Findings:")
            print(f"  • Lowest expected loss: {min_loss_result[1]['expected_cumulative_loss']*100:.2f}% (with {min_loss_result[0]:.0%} min return requirement)")
            print(f"  • Highest expected return: {max_return_result[1]['expected_cumulative_return']*100:.2f}% (with {max_return_result[0]:.0%} min return requirement)")

            print(f"\nAllocation Patterns:")
            for min_return, result in successful_results.items():
                weights = result['weights']
                dominant_asset = max(weights.keys(), key=lambda k: weights[k])
                print(f"  • {min_return:.0%} min return: {dominant_asset} dominant ({weights[dominant_asset]:.1%})")

        else:
            print("❌ No successful optimizations!")
            for min_return, result in optimization_results.items():
                print(f"  {min_return:.0%}: {result['message']}")

        # ================================================================
        # SCENARIO ANALYSIS
        # ================================================================

        if successful_results:
            print(f"\n{'SCENARIO IMPACT ANALYSIS':=^80}")

            representative_result = list(successful_results.values())[0]

            print(f"Portfolio Performance by Scenario:")
            print(f"{'Scenario':<15} {'Probability':<12} {'Cumulative Return':<18} {'Loss Amount':<15} {'Loss Contribution':<18}")
            print(f"{'-'*90}")

            total_expected_loss = 0
            for scenario in probabilities.keys():
                prob = probabilities[scenario]
                cum_return = representative_result['scenario_cumulative_returns'][scenario]
                loss_amount = representative_result['scenario_losses'][scenario]
                loss_contrib = loss_amount * prob
                total_expected_loss += loss_contrib

                print(f"{scenario:<15} {prob:>10.1%} {cum_return:>16.2%} {loss_amount:>13.2%} {loss_contrib:>16.4%}")

            print(f"{'-'*90}")
            print(f"{'Total Expected':<15} {'100.0%':>10} {representative_result['expected_cumulative_return']:>16.2%} {'':>13} {total_expected_loss:>16.4%}")

        # ================================================================
        # SAVE RESULTS
        # ================================================================

        if save_results:
            try:
                output_dir = "thesis_results"
                os.makedirs(output_dir, exist_ok=True)

                # Save main results
                if successful_results:
                    results_data = []
                    for min_return, result in successful_results.items():
                        results_data.append({
                            'min_return_requirement': f"{min_return:.0%}",
                            'expected_cumulative_loss_pct': result['expected_cumulative_loss'] * 100,
                            'expected_cumulative_return_pct': result['expected_cumulative_return'] * 100,
                            'probability_of_loss_pct': result['probability_of_loss'] * 100,
                            'worst_case_pct': result['worst_case_cumulative'] * 100,
                            'best_case_pct': result['best_case_cumulative'] * 100,
                            **{f'weight_{asset}_pct': result['weights'][asset] * 100 for asset in ASSET_CLASSES}
                        })

                    pd.DataFrame(results_data).to_csv(f"{output_dir}/portfolio_results.csv", index=False)

                # Save scenario probabilities
                pd.DataFrame([probabilities]).to_csv(f"{output_dir}/scenario_probabilities.csv", index=False)

                # Save forecasts
                forecast_data = []
                for scenario, asset_returns in forecasts.items():
                    for year in range(HORIZON):
                        forecast_data.append({
                            'scenario': scenario,
                            'year': year + 1,
                            'probability': probabilities[scenario],
                            **{asset: asset_returns[asset][year] for asset in ASSET_CLASSES}
                        })

                pd.DataFrame(forecast_data).to_csv(f"{output_dir}/return_forecasts.csv", index=False)

                print(f"\n✅ Results saved to '{output_dir}/' directory")

            except Exception as e:
                print(f"\n❌ Error saving results: {str(e)}")

        # ================================================================
        # FINAL STATUS
        # ================================================================

        successful_optimizations = len(successful_results)
        total_optimizations = len(optimization_results)

        print(f"\n{'ANALYSIS COMPLETED':=^80}")
        print(f"✅ Scenario probabilities: Estimated for {len(SCENARIOS)} scenarios")
        print(f"✅ Asset return forecasts: Generated for {HORIZON}-year horizon")
        print(f"✅ Portfolio optimizations: {successful_optimizations}/{total_optimizations} successful")

        if successful_optimizations == total_optimizations:
            print(f"🎉 Bachelor thesis analysis completed successfully!")
        elif successful_optimizations > 0:
            print(f"⚠️  Partial success - {total_optimizations - successful_optimizations} optimizations failed")
        else:
            print(f"❌ All optimizations failed")

        results['success'] = successful_optimizations > 0
        results['success_rate'] = successful_optimizations / total_optimizations if total_optimizations > 0 else 0

        return results

    except Exception as e:
        print(f"\n❌ ANALYSIS FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

        results['success'] = False
        results['error'] = str(e)
        return results


def main():
    """Main function to run the bachelor thesis analysis."""

    DATA_PATH = "data/usa_macro_var_and_asset_returns.csv"
    ANCHOR_YEAR = 2019
    MIN_REQUIRED_RETURNS = [0, 0.0448, 0.0448*1.5, 0.0448*2]  # 2%, 4%, 6% minimum returns

    print(f"BACHELOR THESIS")
    print(f"Title: Expectimin Optimal Portfolio with Scenario-Dependent Paths")
    print(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"\n❌ ERROR: Data file '{DATA_PATH}' not found!")
        return None

    print(f"\n✅ Data file found: {DATA_PATH}")
    print(f"✅ Configuration:")
    print(f"   • Anchor year: {ANCHOR_YEAR}")
    print(f"   • Investment horizon: {HORIZON} years")
    print(f"   • Minimum return requirements: {[f'{ret:.0%}' for ret in MIN_REQUIRED_RETURNS]}")
    print(f"   • Scenarios: {len(SCENARIOS)}")
    print(f"   • Asset classes: {len(ASSET_CLASSES)}")

    print(f"\n🚀 Starting bachelor thesis analysis...")

    results = run_bachelor_thesis_analysis(
        data_path=DATA_PATH,
        anchor_year=ANCHOR_YEAR,
        min_required_returns=MIN_REQUIRED_RETURNS,
        save_results=True
    )

    print(f"\n{'BACHELOR THESIS ANALYSIS COMPLETE':=^80}")

    if results['success']:
        print(f"🎓 Analysis completed successfully!")
        print(f"📈 Portfolio optimization successful")
        print(f"💾 Results saved to 'thesis_results/' directory")
        print(f"📊 Success rate: {results['success_rate']:.0%}")

        print(f"\n📋 Next steps:")
        print(f"   1. Review results in 'thesis_results/' directory")
        print(f"   2. Analyze portfolio allocation patterns")
        print(f"   3. Interpret economic scenario impacts")
        print(f"   4. Write discussion and conclusions")

    else:
        print(f"💥 Analysis failed!")
        if 'error' in results:
            print(f"❌ Error: {results['error']}")

    return results


if __name__ == "__main__":
    results = main()

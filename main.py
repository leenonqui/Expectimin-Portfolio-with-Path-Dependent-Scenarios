"""
Simple example demonstrating the complete workflow
Following the thesis methodology from Chapter 3
"""

from scenario_analysis import ScenarioAnalyzer
from portfolio_optimizer import CVaROptimizer
from constants import HORIZON, DEFAULT_CONFIDENCE_LEVEL, ASSET_CLASSES
from utils import get_risk_free_rate, calculate_cumulative_return

def print_section_header(title: str):
    """Print formatted section header"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

def main():
    """Run complete analysis following thesis methodology"""
    
    print_section_header("PATH-DEPENDENT SCENARIO ANALYSIS AND CVAR OPTIMIZATION")
    print("Following Chapter 3 Methodology")
    
    # Parameters
    data_path = "data/usa_macro_var_and_asset_returns.csv"
    anchor_year = 2019  # Last year of historical data
    prediction_start_year = 2020  # First year of prediction period
    
    # Step 1: Initialize analyzer
    print_section_header("STEP 1: LOADING HISTORICAL DATA")
    analyzer = ScenarioAnalyzer(data_path)
    print(f"  ✓ Loaded data up to year {anchor_year}")
    print(f"  ✓ Investment horizon: {HORIZON} years ({prediction_start_year}-{prediction_start_year + HORIZON - 1})")
    
    # Get real risk-free rate from data
    try:
        real_rf_rate_annual = get_risk_free_rate(analyzer.data, prediction_start_year)
        real_rf_cumulative = calculate_cumulative_return([real_rf_rate_annual] * HORIZON, as_log=False)
        print(f"  ✓ Real risk-free rate: {real_rf_rate_annual:.2f}% annual ({real_rf_cumulative:.2f}% cumulative)")
    except Exception as e:
        # Fallback if data not available
        real_rf_rate_annual = 0.5
        real_rf_cumulative = calculate_cumulative_return([real_rf_rate_annual] * HORIZON, as_log=False)
        print(f"  ⚠ Using default real risk-free rate: {real_rf_rate_annual:.2f}% annual ({real_rf_cumulative:.2f}% cumulative)")
        print(f"    (Error: {e})")
    
    # Step 2: Estimate scenario probabilities (Section 3.1.2)
    print_section_header("STEP 2: ESTIMATING SCENARIO PROBABILITIES")
    probabilities = analyzer.estimate_probabilities(anchor_year)
    
    print(f"  Using Mahalanobis distance from {anchor_year-HORIZON+1}-{anchor_year} anchor path")
    print(f"\n  Scenario Probabilities:")
    print(f"  {'Scenario':<15} {'Probability':>10} {'Percentage':>10}")
    print(f"  {'-'*40}")
    for scenario, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        print(f"  {scenario:<15} {prob:>10.4f} {prob*100:>9.2f}%")
    
    # Verify probabilities sum to 1
    total_prob = sum(probabilities.values())
    print(f"  {'-'*40}")
    print(f"  {'Total':<15} {total_prob:>10.4f} {total_prob*100:>9.2f}%")
    
    # Step 3: Forecast asset returns (Section 3.1.3)
    print_section_header("STEP 3: FORECASTING ASSET RETURNS")
    returns = analyzer.forecast_returns(anchor_year)
    
    print(f"  Using top 25% most relevant historical observations")
    print(f"\n  Sample forecasts (most probable scenario):")
    most_probable = max(probabilities.items(), key=lambda x: x[1])[0]
    sample_returns = returns[most_probable]
    
    print(f"  Scenario: {most_probable}")
    print(f"  {'Asset':<10} {'Year 1':>8} {'Year 2':>8} {'Year 3':>8} {'Cumulative':>12}")
    print(f"  {'-'*50}")
    
    for asset in ASSET_CLASSES:
        yearly = sample_returns[asset]
        cumul = calculate_cumulative_return(yearly, as_log=False)
        print(f"  {asset:<10} {yearly[0]:>7.1f}% {yearly[1]:>7.1f}% {yearly[2]:>7.1f}% {cumul:>11.1f}%")
    
    # Step 4: Portfolio optimization (Section 3.2)
    print_section_header("STEP 4: PORTFOLIO OPTIMIZATION (CVaR MINIMIZATION)")
    print(f"  Confidence level: {DEFAULT_CONFIDENCE_LEVEL:.0%}")
    print(f"  Alpha (tail probability): {1-DEFAULT_CONFIDENCE_LEVEL:.0%}")
    
    optimizer = CVaROptimizer(probabilities, returns)
    
    # Optimize without constraint
    print(f"\n  A. Unconstrained CVaR Minimization:")
    result_unconstrained = optimizer.optimize()
    
    if result_unconstrained.success:
        print(f"     Optimal Weights:")
        for asset, weight in result_unconstrained.weights.items():
            print(f"       {asset}: {weight:>6.1%}")
        print(f"\n     Risk Metrics:")
        print(f"       CVaR (95%): {result_unconstrained.cvar:>7.2f}%")
        print(f"       VaR (95%):  {result_unconstrained.var:>7.2f}%")
        print(f"       Expected Return: {result_unconstrained.expected_return:>7.2f}%")
    else:
        print(f"     ✗ Optimization failed")
    
    # Optimize with risk-free rate constraint
    print(f"\n  B. CVaR Minimization with Minimum Return Constraint:")
    print(f"     Constraint: Expected return ≥ {real_rf_cumulative:.2f}% (real risk-free rate)")
    
    result_constrained = optimizer.optimize(min_return=real_rf_cumulative)
    
    if result_constrained.success:
        print(f"     Optimal Weights:")
        for asset, weight in result_constrained.weights.items():
            print(f"       {asset}: {weight:>6.1%}")
        print(f"\n     Risk Metrics:")
        print(f"       CVaR (95%): {result_constrained.cvar:>7.2f}%")
        print(f"       VaR (95%):  {result_constrained.var:>7.2f}%")
        print(f"       Expected Return: {result_constrained.expected_return:>7.2f}%")
        print(f"       ✓ Meets minimum return constraint")
    else:
        print(f"     ✗ Optimization failed - constraint may be infeasible")
    
    # Step 5: Portfolio analysis
    print_section_header("STEP 5: PORTFOLIO ANALYSIS")
    
    # Use constrained result if successful, otherwise unconstrained
    portfolio_to_analyze = result_constrained if result_constrained.success else result_unconstrained
    
    if portfolio_to_analyze.success:
        analysis = optimizer.analyze_portfolio(portfolio_to_analyze.weights)
        
        print(f"\n  Scenario-by-Scenario Returns ({HORIZON}-year cumulative):")
        print(f"  {'Scenario':<15} {'Probability':>12} {'Return':>10} {'Status':<10}")
        print(f"  {'-'*50}")
        
        # Sort by return for clarity
        sorted_scenarios = sorted(analysis['scenario_returns'].items(), key=lambda x: x[1])
        
        for scenario, ret in sorted_scenarios:
            prob = probabilities[scenario]
            status = "TAIL RISK" if ret <= analysis['var'] else "NORMAL"
            print(f"  {scenario:<15} {prob:>12.3f} {ret:>9.2f}%  {status:<10}")
        
        print(f"\n  Risk Metrics Summary:")
        print(f"    Expected Return:     {analysis['expected_return']:>7.2f}%")
        print(f"    Standard Deviation:  {analysis['std_dev']:>7.2f}%")
        print(f"    VaR (95%):          {analysis['var']:>7.2f}% (5% chance of worse)")
        print(f"    CVaR (95%):         {analysis['cvar']:>7.2f}% (expected loss if worse than VaR)")
        print(f"    Worst Case:         {analysis['worst_case']:>7.2f}% ({min(analysis['scenario_returns'], key=analysis['scenario_returns'].get)})")
        print(f"    Best Case:          {analysis['best_case']:>7.2f}% ({max(analysis['scenario_returns'], key=analysis['scenario_returns'].get)})")
        
        # Risk-return comparison
        print(f"\n  Risk-Return Trade-off:")
        print(f"    Return above risk-free: {analysis['expected_return'] - real_rf_cumulative:>6.2f}%")
        print(f"    Sharpe Ratio:           {analysis['sharpe_ratio']:>6.3f}")
        print(f"    Return-to-CVaR ratio:   {-analysis['expected_return'] / analysis['cvar']:>6.3f}")
        
        # Portfolio composition analysis
        print(f"\n  Portfolio Composition Analysis:")
        weights = portfolio_to_analyze.weights
        
        # Classify portfolio style
        if weights['Cash'] > 0.6:
            style = "Conservative (Cash-heavy)"
        elif weights['Stocks'] > 0.6:
            style = "Aggressive (Stock-heavy)"
        elif weights['Bonds'] > 0.6:
            style = "Income-focused (Bond-heavy)"
        else:
            style = "Balanced"
        
        print(f"    Portfolio Style: {style}")
        print(f"    Diversification: ", end="")
        
        # Calculate concentration
        concentration = sum(w**2 for w in weights.values())
        if concentration > 0.8:
            print("Low (concentrated)")
        elif concentration > 0.5:
            print("Moderate")
        else:
            print("High (well-diversified)")
    
    # Step 6: Efficient Frontier Analysis (optional)
    print_section_header("STEP 6: EFFICIENT FRONTIER ANALYSIS (OPTIONAL)")
    
    print("  Generating efficient frontier...")
    frontier = optimizer.efficient_frontier(n_points=10)
    
    if frontier:
        print(f"\n  {'Expected Return':>15} {'CVaR':>10} {'Cash':>8} {'Stocks':>8} {'Bonds':>8}")
        print(f"  {'-'*60}")
        
        for exp_ret, cvar, weights in frontier:
            print(f"  {exp_ret:>14.2f}% {cvar:>9.2f}% {weights['Cash']:>7.1%} {weights['Stocks']:>7.1%} {weights['Bonds']:>7.1%}")
        
        print(f"\n  Frontier Summary:")
        print(f"    Minimum CVaR: {min(f[1] for f in frontier):.2f}%")
        print(f"    Maximum Return: {max(f[0] for f in frontier):.2f}%")
    else:
        print("  ✗ Could not generate efficient frontier")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    
    # Return results for further analysis
    return {
        'probabilities': probabilities,
        'returns': returns,
        'unconstrained': result_unconstrained,
        'constrained': result_constrained,
        'optimizer': optimizer
    }

if __name__ == "__main__":
    results = main()

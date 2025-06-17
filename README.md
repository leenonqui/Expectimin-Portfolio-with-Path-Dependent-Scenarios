"""
# Expectiminimax Portfolio Optimization with Path-Dependent Scenarios

A Python package implementing the GIC methodology for scenario analysis and expectiminimax portfolio optimization framework.

## Overview

This package replicates the methodology described in:

**Kritzman, M., Li, D., Qiu, G., & Turkington, D. (2021). "Portfolio Choice with Path-Dependent Scenarios." Journal of Portfolio Management, 47(4), 69-79.**

### Key Features

- **Path-Dependent Scenarios**: Define economic scenarios as sequences rather than averages
- **Mahalanobis Distance**: Estimate scenario probabilities based on statistical similarity
- **Partial Sample Regression**: Forecast asset returns using relevance-weighted observations
- **Expectiminimax Optimization**: Maximize expected utility across multiple risk aversion profiles

## Installation

```bash
pip install -e .
```

## Requirements

- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.20.0
- scipy >= 1.7.0
- pydantic >= 1.8.0

## Quick Start

```python
from expectiminimax_portfolio import GICAnalyzer, ExpectiminimaxOptimizer

# Initialize with historical data
analyzer = GICAnalyzer("data/usa_macro_var_and_asset_returns.csv")

# Run GIC methodology
results = analyzer.analyze(prediction_year=2020)

# View scenario probabilities
print("Scenario Probabilities:")
for scenario, prob in results.scenario_probabilities.items():
    print(f"  {scenario}: {prob:.1%}")

# Optimize portfolios
optimizer = ExpectiminimaxOptimizer(
    scenario_probabilities=results.scenario_probabilities,
    asset_returns=results.asset_returns
)

portfolios = optimizer.optimize_all_profiles()

# View optimal allocations
for profile_name, result in portfolios.items():
    print(f"\n{profile_name} (λ={result.risk_aversion}):")
    for asset, weight in result.optimal_weights.items():
        print(f"  {asset}: {weight:.1%}")
```

## Methodology

### 1. Economic Scenarios

Six path-dependent scenarios with 3-year GDP growth and inflation sequences:

- **Baseline V**: Recovery with initial decline then strong growth
- **Shallow V**: Mild recession with steady recovery
- **U-Shaped**: Extended recession with delayed recovery
- **W-Shaped**: Double-dip recession pattern
- **Depression**: Prolonged economic decline
- **Stagflation**: Stagnant growth with high inflation

### 2. Probability Estimation

Scenario probabilities calculated using Mahalanobis distance:

```
d = (x - γ)'Ω⁻¹(x - γ)
Likelihood ∝ e^(-d/2)
```

Where:
- `x` = prospective scenario path
- `γ` = recent economic experience
- `Ω` = covariance matrix of historical path changes

### 3. Asset Return Forecasting

Partial sample regression using top 25% most relevant observations:

```
Relevance = Similarity + Informativeness
Forecast = Weighted average of relevant historical returns
```

### 4. Portfolio Optimization

Expectiminimax framework maximizing expected utility:

```
U(R) = E[R] - (λ/2)Var(R)
max Σ P(scenario) × U(R_scenario)
```

Risk aversion profiles (λ):
- 0.0: Risk Neutral
- 0.5: Low Risk Aversion
- 1.0: Moderate Risk Aversion
- 2.0: High Risk Aversion
- 5.0: Very High Risk Aversion

## Data Requirements

Historical data should include:
- Real GDP per capita (rgdpmad)
- Consumer Price Index (cpi)
- Short-term interest rate (bill_rate)
- Equity total return (eq_tr)
- Bond total return (bond_tr)

Sample data format (CSV with 'year' index):
```
year,rgdpmad,cpi,bill_rate,eq_tr,bond_tr
1929,100.0,50.0,0.045,0.08,-0.02
1930,95.0,49.0,0.035,-0.25,0.05
...
```

## Examples

See the `examples/` directory for:
- `basic_usage.py`: Simple workflow example
- `replicate_thesis_results`.

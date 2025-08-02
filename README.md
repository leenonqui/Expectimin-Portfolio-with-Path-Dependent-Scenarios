# Expectimin Portfolio Optimization with Path-Dependent Scenarios

A Python package implementing scenario-based portfolio optimization using the expectimin framework for minimizing expected cumulative losses.

## Overview

This package implements a methodology for portfolio optimization under economic uncertainty using:

- **Path-Dependent Economic Scenarios**: Define economic conditions as multi-year sequences rather than point estimates
- **Mahalanobis Distance-Based Probability Estimation**: Calculate scenario probabilities based on statistical similarity to recent economic history
- **Partial Sample Regression**: Forecast asset returns using relevance-weighted historical observations
- **Expectimin Optimization**: Minimize expected cumulative losses across scenarios

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd expectimin-portfolio

# Install dependencies
pip install -r requirements.txt
```

## Requirements

Core dependencies for expectimin portfolio optimization
pandas>=1.3.0
numpy>=1.20.0
scipy>=1.7.0
pulp

Data validation and modeling
pydantic>=1.8.0

Visualization and analysis
matplotlib>=3.3.0
seaborn>=0.11.0

Development and analysis environment
jupyterlab>=3.0.0

## Methodology

### 1. Economic Scenarios

Six path-dependent scenarios with 3-year GDP growth and inflation sequences:

| Scenario | Year 1 GDP | Year 2 GDP | Year 3 GDP | Year 1 Inf | Year 2 Inf | Year 3 Inf |
|----------|------------|------------|------------|------------|------------|------------|
| **Baseline V** | -3.5% | 3.8% | 2.3% | 1.0% | 1.7% | 2.0% |
| **Shallow V** | -1.9% | 5.4% | 3.9% | 1.0% | 1.7% | 2.0% |
| **U-Shaped** | -3.5% | 0.0% | 3.9% | 1.0% | 0.4% | 0.7% |
| **W-Shaped** | -3.5% | 3.8% | -4.2% | 1.0% | 1.7% | 2.0% |
| **Depression** | -5.1% | -5.9% | -7.4% | -0.3% | -5.9% | -5.6% |
| **Stagflation** | -5.1% | -2.7% | -0.9% | 2.3% | 4.2% | 5.8% |

### 2. Probability Estimation

Scenario probabilities calculated using Mahalanobis distance from recent economic experience:

```
d = (x - γ)'Ω⁻¹(x - γ)
P(scenario) ∝ e^(-d/2)
```

Where:
- `x` = prospective scenario path
- `γ` = recent 3-year economic path (anchor)
- `Ω` = covariance matrix of historical path changes

### 3. Asset Return Forecasting

Partial sample regression using the top 25% most relevant historical observations:

```
Relevance = Similarity + Informativeness
Forecast = Weighted average of relevant historical returns
```

**Similarity**: How close a historical path is to the target scenario
**Informativeness**: How much a historical observation deviates from the average

### 4. Portfolio Optimization

Expectimin framework minimizing expected cumulative loss:

```
minimize: E[Loss] = Σ P(s) × max(0, -CumulativeReturn_s)
subject to: Σ weights = 1
            weights ≥ 0
            cash ≤ .1
            stocks ≥ .25
```

## Project Structure

```
expectimin-portfolio/
├── main.py                    # Main analysis pipeline
├── learning.py                # Bayesian Learning for belief updating
├── scenario_analysis.py       # ScenarioAnalyzer class
├── optimization.py  # ExpectiminOptimizer class
├── constants.py              # Scenario definitions and parameters
├── utils.py                  # Mathematical utility functions
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore patterns
├── data/
    └── usa_macro_var_and_asset_returns.csv  # Historical data
```

## Data Requirements

Historical data should include:
- **rgdpmad**: Real GDP per capita (Maddison Project)
- **cpi**: Consumer Price Index
- **bill_rate**: Short-term interest rate (3-month Treasury)
- **eq_tr**: Equity total return
- **bond_tr**: Bond total return

Data format (CSV with 'year' index):
```csv
year,rgdpmad,cpi,bill_rate,eq_tr,bond_tr
1929,6898.72,13.85,0.0407,-0.0337,0.034
1930,6212.71,13.52,0.0404,-0.2294,0.0434
...
```

## Running the Analysis

Execute the complete analysis pipeline:

```bash
python main.py
```

This will:
1. Estimate scenario probabilities based on 2015-17 anchor path
2. Forecast 3-year asset returns for each scenario
3. Optimize portfolios under expectimin selection rule


## Mathematical Framework

The methodology combines:

1. **Multivariate statistical analysis** for scenario probability estimation
2. **Relevance-weighted regression** for return forecasting  
3. **Non-convex optimization** for portfolio construction
4. **Dynamic weight modeling** accounting for realistic portfolio drift

This provides a robust framework for portfolio optimization under economic uncertainty with realistic assumptions about investor behavior and market dynamics.

## References

Methodology inspired by portfolio optimization research focusing on scenario-based approaches and Game Theory for handling economic uncertainty.

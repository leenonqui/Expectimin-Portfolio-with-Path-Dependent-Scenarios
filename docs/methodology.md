"""
# GIC Methodology Implementation

## Overview

This package implements the portfolio choice methodology with path-dependent scenarios as described in:

**Kritzman, M., Li, D., Qiu, G., & Turkington, D. (2021). Portfolio Choice with Path-Dependent Scenarios. *Journal of Portfolio Management*, 47(4), 69-79.**

## Methodology Components

### 1. Path-Dependent Scenario Definition

Unlike traditional scenario analysis that uses single-period averages, the GIC methodology defines scenarios as sequences of values over multiple periods. For a 3-year horizon:

- Traditional: GDP growth = 2.5% (average)
- Path-dependent: GDP growth = [-3.5%, 3.8%, 2.3%] (year-by-year path)

**Advantages:**
- More information for statistical similarity measurement
- Better mapping to asset class returns via partial sample regression
- Alignment with qualitative forecasting intuition

### 2. Scenario Probability Estimation

Probabilities are estimated using the Mahalanobis distance to measure statistical similarity between prospective scenarios and recent economic experience.

**Key Equations:**

**Equation 1: Mahalanobis Distance**
```
d = (x - γ)'Ω⁻¹(x - γ)
```
Where:
- x = prospective scenario path vector
- γ = recent economic experience path vector
- Ω = covariance matrix of historical path changes

**Equation 2: Scenario Likelihood**
```
Likelihood ∝ e^(-d/2)
```

**Implementation Details:**
- Path vectors: [GDP₁, GDP₂, GDP₃, INF₁, INF₂, INF₃]
- Covariance estimated from differences between consecutive 3-year paths
- Anchor path: most recent 3-year historical sequence

### 3. Partial Sample Regression for Asset Returns

Converts economic scenarios to asset class returns using relevance-weighted regression.

**Key Equations:**

**Equation 3: Multivariate Similarity**
```
Similarity(xᵢ, xₜ) = -(xᵢ - xₜ)'Ω⁻¹(xᵢ - xₜ)
```

**Equation 4: Informativeness**
```
Informativeness(xᵢ) = (xᵢ - x̄)'Ω⁻¹(xᵢ - x̄)
```

**Equation 5: Relevance**
```
Relevance(xᵢ) = Similarity(xᵢ, xₜ) + Informativeness(xᵢ)
```

**Equations 6-7: Partial Sample Regression**
```
ŷₜ = ȳ + (1/2n) Σᵢ₌₁ⁿ [Similarity(xᵢ, xₜ) + Informativeness(xᵢ)](yᵢ - ȳ)
```

Where n is the subset of most relevant observations (typically top 25%).

### 4. Expectiminimax Portfolio Optimization

Maximizes expected utility across scenarios using mean-variance utility function:

```
U(R) = E[R] - (λ/2)Var(R)
```

Where λ is the risk aversion parameter:
- λ = 0: Risk neutral (pure return maximization)
- λ > 0: Risk averse (penalizes volatility)

**Optimization Problem:**
```
max_w Σₛ P(s) × U(Rₛ(w))
```

Subject to:
- Σᵢ wᵢ = 1 (weights sum to 1)
- 0 ≤ wᵢ ≤ 1 (long-only constraints)

## Data Requirements

### Historical Data (JST Macrohistory Database)
- Real GDP per capita (rgdpmad)
- Consumer Price Index (cpi)
- Short-term interest rate (bill_rate)
- Equity total return (eq_tr)
- Bond total return (bond_tr)

### Scenario Definitions (Table 1, Kritzman et al.)
Six 3-year scenarios with GDP growth and inflation paths:
1. Baseline V
2. Shallow V
3. U-Shaped
4. W-Shaped
5. Depression
6. Stagflation

## Implementation Validation

The implementation preserves the exact methodology from the original paper:

1. **Dimensional Consistency**: Macro paths (6D) → Asset paths (9D) via PSR
2. **Covariance Estimation**: From path differences, not path levels
3. **Anchor Selection**: Last historical path as recent experience
4. **Relevance Calculation**: Combines similarity and informativeness
5. **Return Conversion**: From changes/premiums to absolute levels

## References

Kritzman, M., Li, D., Qiu, G., & Turkington, D. (2021). Portfolio Choice with Path-Dependent Scenarios. *Journal of Portfolio Management*, 47(4), 69-79.

Czasonis, M., Kritzman, M., & Turkington, D. (2020). Addition by Subtraction: A Better Way to Forecast Factor Returns (and Everything Else). *Journal of Portfolio Management*, 46(8), 98-107.
"""

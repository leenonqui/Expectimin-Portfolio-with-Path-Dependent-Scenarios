"""
Constants and Configuration
All fixed parameters and scenario definitions for the thesis
"""

from typing import Dict, List
from dataclasses import dataclass

# ============================================================================
# INVESTMENT PARAMETERS
# ============================================================================

# Investment horizon (T years)
HORIZON = 3

# Default confidence level for CVaR
DEFAULT_CONFIDENCE_LEVEL = 0.95

# Partial sample regression percentile
PSR_TOP_PERCENTILE = 0.25

# ============================================================================
# SCENARIO DEFINITIONS (Table 1 from Kritzman et al. 2021)
# ============================================================================

@dataclass
class ScenarioDefinition:
    """Path-dependent economic scenario"""
    name: str
    gdp_growth: List[float]  # T-year GDP growth path
    inflation: List[float]   # T-year inflation path

# Six scenarios from the thesis
SCENARIOS: Dict[str, ScenarioDefinition] = {
    "Baseline V": ScenarioDefinition(
        name="Baseline V",
        gdp_growth=[-3.5, 3.8, 2.3],
        inflation=[1.0, 1.7, 2.0]
    ),
    "Shallow V": ScenarioDefinition(
        name="Shallow V",
        gdp_growth=[-1.9, 5.4, 3.9],
        inflation=[1.0, 1.7, 2.0]
    ),
    "U-Shaped": ScenarioDefinition(
        name="U-Shaped",
        gdp_growth=[-3.5, 0.0, 3.9],
        inflation=[1.0, 0.4, 0.7]
    ),
    "W-Shaped": ScenarioDefinition(
        name="W-Shaped",
        gdp_growth=[-3.5, 3.8, -4.2],
        inflation=[1.0, 1.7, 2.0]
    ),
    "Depression": ScenarioDefinition(
        name="Depression",
        gdp_growth=[-5.1, -5.9, -7.4],
        inflation=[-0.3, -5.9, -5.6]
    ),
    "Stagflation": ScenarioDefinition(
        name="Stagflation",
        gdp_growth=[-5.1, -2.7, -0.9],
        inflation=[2.3, 4.2, 5.8]
    )
}

# ============================================================================
# ASSET CLASSES
# ============================================================================

ASSET_CLASSES = ["Cash", "Stocks", "Bonds"]

# ============================================================================
# RISK FREE RATES - Historical and Test Period
# ============================================================================

# Risk-free rate: 3-Year U.S. Treasury Constant Maturity Rate as of January 2, 2020
# Source: Federal Reserve Economic Data (FRED), series GS3
RISK_FREE_RATE = 0.0152  # 1.52%

# Expected inflation rate for 2020-2022 forecast period
# Calculation: Average of actual inflation rates 2020-2022
# 2020: 1.2%, 2021: 4.7%, 2022: 8.0%
INFLATION = (1.2 + 4.7 + 8.0) / 3.0 / 100  # 4.63% average, converted to decimal

# Current interest rate for cash forecasting: 3-Month Treasury Bill Rate as of January 2, 2020
CURRENT_INTEREST_RATE = 0.0154  # 1.54% (3-Month Treasury on Jan 2, 2020)

# Real risk-free rates for start of january 2018-2020
REAL_RISK_FREE_RATES = {
    2018: 0.0275,  # 2.75%
    2019: 0.02625, # 2.625%
    2020: 0.01625  # 1.625%
}

# ============================================================================
# OPTIMIZATION PARAMETERS
# ============================================================================

# Minimum weight constraints (no short selling)
MIN_WEIGHT = 0.0
MAX_WEIGHT = 1.0

# Numerical tolerance for optimization
OPTIMIZATION_TOLERANCE = 1e-12

# Risk aversion parameter for mean-variance utility
# A = 2: Risk neutral
# A = 3-4: Moderate risk aversion
# A = 5: High risk aversion
RISK_AVERSION = 3.0  # Default moderate risk aversion

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================

# Number of economic variables (GDP growth, inflation)
N_ECONOMIC_VARS = 2

# Start year for full historical analysis
HISTORICAL_START_YEAR = 1927

# Minimum years of data required for analysis
MIN_HISTORICAL_YEARS = 30

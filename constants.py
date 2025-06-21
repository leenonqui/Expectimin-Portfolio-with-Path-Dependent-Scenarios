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
# DATA COLUMN MAPPINGS
# ============================================================================

# JST Macrohistory Database column names
DATA_COLUMNS = {
    'year': 'year',
    'gdp': 'rgdpmad',      # Real GDP per capita
    'cpi': 'cpi',          # Consumer Price Index
    'cash': 'bill_rate',   # Short-term interest rate
    'stocks': 'eq_tr',     # Equity total return
    'bonds': 'bond_tr'     # Bond total return
}

# ============================================================================
# OPTIMIZATION PARAMETERS
# ============================================================================

# Minimum weight constraints (no short selling)
MIN_WEIGHT = 0.0
MAX_WEIGHT = 1.0

# Numerical tolerance for optimization
OPTIMIZATION_TOLERANCE = 1e-12

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================

# Number of economic variables (GDP growth, inflation)
N_ECONOMIC_VARS = 2

# Start year for full historical analysis
HISTORICAL_START_YEAR = 1929

# Minimum years of data required for analysis
MIN_HISTORICAL_YEARS = 30

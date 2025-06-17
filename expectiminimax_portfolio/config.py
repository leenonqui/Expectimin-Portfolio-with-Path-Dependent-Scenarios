
"""
Configuration constants for GIC methodology and expectiminimax optimization

All scenarios and parameters are based on Kritzman et al. (2021) Table 1
"""

# Scenario Analysis Parameters
SCENARIO_HORIZON_YEARS = 3
NUM_MACRO_VARS = 2  # Real GDP Growth, Inflation
NUM_ASSET_VARS = 3  # Cash changes, Stock excess, Bond excess

# Data Parameters
FULL_MACRO_START_DATE = '1929'
FULL_MACRO_END_DATE = '2019'

# Partial Sample Regression Parameters
PSR_TOP_PERCENTILE = 0.25  # Use top 25% most relevant observations

# GIC Economic Scenarios (Table 1, Page 14 of Kritzman et al. 2021)
GIC_SCENARIOS = {
    "Baseline V": {
        "GDP Growth": [-3.5, 3.8, 2.3],
        "Inflation": [1.0, 1.7, 2.0]
    },
    "Shallow V": {
        "GDP Growth": [-1.9, 5.4, 3.9],
        "Inflation": [1.0, 1.7, 2.0]
    },
    "U-Shaped": {
        "GDP Growth": [-3.5, 0.0, 3.9],
        "Inflation": [1.0, 0.4, 0.7]
    },
    "W-Shaped": {
        "GDP Growth": [-3.5, 3.8, -4.2],
        "Inflation": [1.0, 1.7, 2.0]
    },
    "Depression": {
        "GDP Growth": [-5.1, -5.9, -7.4],
        "Inflation": [-0.3, -5.9, -5.6]
    },
    "Stagflation": {
        "GDP Growth": [-5.1, -2.7, -0.9],
        "Inflation": [2.3, 4.2, 5.8]
    }
}

# Risk Aversion Profiles for Expectiminimax Optimization
RISK_AVERSION_PROFILES = [
    {"name": "Risk Neutral", "risk_aversion": 0.0},
    {"name": "Low Risk Aversion", "risk_aversion": 0.5},
    {"name": "Moderate Risk Aversion", "risk_aversion": 1.0},
    {"name": "High Risk Aversion", "risk_aversion": 2.0},
    {"name": "Very High Risk Aversion", "risk_aversion": 5.0}
]

# Asset Classes
ASSET_CLASSES = ["Cash", "Stocks", "Bonds"]

# Data Column Mappings for JST Dataset
JST_COLUMNS = {
    'gdp': 'rgdpmad',      # Real GDP per capita
    'inflation': 'cpi',     # Consumer Price Index
    'cash': 'bill_rate',    # Short-term interest rate
    'stocks': 'eq_tr',      # Equity total return
    'bonds': 'bond_tr'      # Bond total return
}

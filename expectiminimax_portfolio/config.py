"""
All scenarios and parameters are based on Kritzman et al. (2021) Table 1
includes liquidity preference as minimum cash allocation constraint
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

# NEW: Liquidity Preference Profiles
# Defines minimum cash allocation as percentage (0-10%)
LIQUIDITY_PREFERENCE_PROFILES = [
    {"name": "No Liquidity Preference", "min_cash_pct": 0.0, "description": "No minimum cash requirement"},
    {"name": "Minimal Liquidity", "min_cash_pct": 0.02, "description": "2% minimum cash for transactions"},
    {"name": "Low Liquidity Preference", "min_cash_pct": 0.05, "description": "5% minimum cash buffer"},
    {"name": "Moderate Liquidity Preference", "min_cash_pct": 0.08, "description": "8% minimum cash for opportunities"},
    {"name": "High Liquidity Preference", "min_cash_pct": 0.10, "description": "10% minimum cash for security"}
]

# Combined Risk-Liquidity Profiles (inspired by GIC examples)
COMBINED_PROFILES = [
    # Growth-oriented profiles (low risk aversion)
    {"name": "Aggressive Growth", "risk_aversion": 0.0, "min_cash_pct": 0.00},
    {"name": "Growth with Safety Net", "risk_aversion": 0.5, "min_cash_pct": 0.02},

    # Balanced profiles (moderate risk aversion)
    {"name": "Balanced", "risk_aversion": 1.0, "min_cash_pct": 0.05},
    {"name": "Balanced Conservative", "risk_aversion": 1.0, "min_cash_pct": 0.08},

    # Conservative profiles (high risk aversion)
    {"name": "Conservative", "risk_aversion": 2.0, "min_cash_pct": 0.08},
    {"name": "Very Conservative", "risk_aversion": 5.0, "min_cash_pct": 0.10}
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

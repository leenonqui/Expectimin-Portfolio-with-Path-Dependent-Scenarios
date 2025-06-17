import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis

# --- 1. Configuration Parameters ---

SCENARIO_HORIZON_YEARS = 3
PERIODS_PER_YEAR_IN_PATH = 1 # Annual data for paths
TOTAL_SCENARIO_PERIODS = SCENARIO_HORIZON_YEARS * PERIODS_PER_YEAR_IN_PATH # This will be 3

NUM_MACRO_VARS = 2 # Real GDP Growth (per capita), Inflation (as per GIC paper)

# Helper functions
def similarity(x_i, x_t, inv_cov):
    return -(mahalanobis(x_i, x_t, inv_cov) ** 2)

def informativeness(x_i, x_mean, inv_cov):
    return (mahalanobis(x_i, x_mean, inv_cov) ** 2)

def create_path_vector(*args):
    """
    Combines Macro values into a flattened path vector
    ex: [GDP_Y1, GDP_Y2, GDP_Y3, INF_Y1, INF_Y2, INF_Y3, ...].
    """
    path_vector = []
    for arg in args:
        for i in range(SCENARIO_HORIZON_YEARS):
            path_vector.append(arg[i])
    return np.array(path_vector)

# --- 2. Macroeconomic Data (ACTUAL JST DATA) ---

JST_FILE_PATH = '../usa_macro_var_and_asset_returns.csv'
FULL_MACRO_START_DATE = '1929'
FULL_MACRO_END_DATE = '2019'

# --- Load JST data ---
try:
    jst_raw_data = pd.read_csv(JST_FILE_PATH, index_col='year')

    # Filter to desired range AFTER processing
    historical_macro_data_full = jst_raw_data.loc[
        (jst_raw_data.index >= pd.to_datetime(FULL_MACRO_START_DATE).year - SCENARIO_HORIZON_YEARS) &
        (jst_raw_data.index <= pd.to_datetime(FULL_MACRO_END_DATE).year)
    ]

except FileNotFoundError:
    print(f"Error: {JST_FILE_PATH} not found. Please ensure it's in the script directory.")
except KeyError as e:
    print(f"Error: Required column not found in JST dataset: {e}. Please check and adjust 'jst_relevant_columns'.")
    print(f"Available columns in JST data: {jst_raw_data.columns.tolist()}")
    exit()
except Exception as e:
    print(f"An error occurred while loading or processing JST data: {e}")
    exit()

# Create DataFrame with GDP Growth and Inflation

macro_var_df = pd.DataFrame()
macro_var_df.index = historical_macro_data_full.index
macro_var_df['GDP Growth'] = historical_macro_data_full['rgdpmad'].pct_change() * 100
macro_var_df['Inflation'] = historical_macro_data_full['cpi'].pct_change() * 100

# Drop rows with NaN values resulting from pct_change (typically the first year)
macro_var_df.dropna(inplace=True)

print(f"\nLoaded and Processed Actual JST Macro Data (Annual, {pd.to_datetime(FULL_MACRO_START_DATE).year - 2} to {FULL_MACRO_END_DATE}):")
print(macro_var_df.head())
print(macro_var_df.tail())
print(f"Total annual periods: {len(macro_var_df)}")

NUM_FULL_MACRO_PERIODS = len(macro_var_df)


# --- 3. GIC Macroeconomic Scenario Definitions (from Kritzman et al., 2020, Table 1, Page 14) ---
# These are the six specific 3-year paths for Real GDP Growth and Inflation.
# I have now explicitly used the values from Table 1, page 14 of the GIC paper.
# These values are percentages, so they are used as such (e.g., 3.5 for 3.5%).

gic_scenarios = {
    "Baseline V": {
        "GDP Growth": [-3.5, 3.8, 2.3], # Year 1, Year 2, Year 3 annual growth (%)
        "Inflation": [1.0, 1.7, 2.0]   # Year 1, Year 2, Year 3 annual inflation (%)
    },
    "Shallow V": {
        "GDP Growth": [-1.9, 5.4, 3.9],
        "Inflation": [1.0, 1.7, 2.0]
    },
    "U-Shaped": { # Note: Paper uses "U-Shaped" not "U-Shaped V" in Table 1
        "GDP Growth": [-3.5, 0.0, 3.9],
        "Inflation": [1.0, 0.4, 0.7]
    },
    "W-Shaped": { # Note: Paper uses "W-Shaped" not "W-Shaped V" in Table 1
        "GDP Growth": [-3.5, 3.8, -4.2], # Corrected from table values
        "Inflation": [1.0, 1.7, 2.0] # Corrected from table values
    },
    "Depression": {
        "GDP Growth": [-5.1, -5.9, -7.4],
        "Inflation": [-0.3, -5.9, -5.6] # Deflationary
    },
    "Stagflation": {
        "GDP Growth": [-5.1, -2.7, -0.9],
        "Inflation": [2.3, 4.2, 5.8]
    }
}

prospective_scenario_paths = {}
for name, data in gic_scenarios.items():
    path_vector = create_path_vector(data["GDP Growth"], data["Inflation"])
    prospective_scenario_paths[name] = path_vector

print(f"\nExample prospective path vector shape: {prospective_scenario_paths['Baseline V'].shape}")
print(f"Expected path vector dimension: {NUM_MACRO_VARS * SCENARIO_HORIZON_YEARS}")


    # --- 4. Probability Prediction for 2019 ---
# Predict probabilities for the year 2019 only, using training data up to end of 2018.

predicted_probabilities_over_time = [] # To store probabilities for the prediction year

# Loop only for prediction_year_start
prediction_year_start = 2020

# Define the training period end year (end of the year PRIOR to prediction_year_start)
training_end_year = prediction_year_start - 1

# Filter historical macro data for the current training window (1927 to training_end_year)
current_training_macro_data = macro_var_df.loc[
    (macro_var_df.index >= pd.to_datetime(FULL_MACRO_START_DATE).year - 2) & (macro_var_df.index <= training_end_year)
].copy()

# Check if there's enough data for training to form paths AND calculate differences for covariance
# Need at least SCENARIO_HORIZON_YEARS for a path, and at least 2 paths to calculate a difference.
# So, minimum (SCENARIO_HORIZON_YEARS + 1) years to have at least one valid difference.
if len(current_training_macro_data) < SCENARIO_HORIZON_YEARS + 1:
    print()
    predicted_probabilities_over_time.append({
        'year': prediction_year_start,
        **{s_name: np.nan for s_name in gic_scenarios.keys()}
    })
    raise Exception(f"Skipping {prediction_year_start}: Not enough training data before {training_end_year} to form sufficient paths for covariance calculation.")

print(f"\n--- Predicting Probabilities for {prediction_year_start} using training data up to {training_end_year} ---")

    # --- 4.1. Historical Path Database Construction for current training window ---
current_historical_paths = []
num_training_years = len(current_training_macro_data)

# Paths are P_t-2, P_t-1, P_t
# The loop should go from the first possible start year up to the year that allows the last full path
# ending in training_end_year.

# This loop generates all overlapping 3-year paths.
#    for i in range(0, num_training_years - SCENARIO_HORIZON_YEARS + 1, SCENARIO_HORIZON_YEARS):
for i in range(num_training_years - SCENARIO_HORIZON_YEARS + 1):
    path_segment = current_training_macro_data.iloc[i : i + SCENARIO_HORIZON_YEARS]
    flattened_path = create_path_vector(
        path_segment['GDP Growth'].values,
        path_segment['Inflation'].values
    )
    current_historical_paths.append(flattened_path)

current_historical_paths_array = np.array(current_historical_paths)
print(f"  Shape of current Historical Paths Array (all overlapping paths): {current_historical_paths_array.shape}")

    # --- 4.2. Covariance Matrix Estimation from CHANGES IN PATHS ---
# "We computed the covariance matrix from the changes in the values of the economic variables
# from one three-year period to the next three-year period (in other words, the differences in the paths of the variables)."

if len(current_historical_paths_array) < 2:
    print(f"  Not enough historical paths ({len(current_historical_paths_array)}) to compute changes for covariance. Skipping.")
    predicted_probabilities_over_time.append({
        'year': prediction_year_start,
        **{s_name: np.nan for s_name in gic_scenarios.keys()}
    })
    raise Exception(f"  Not enough historical paths ({len(current_historical_paths_array)}) to compute changes for covariance. Interrupting.")
    # Calculate differences between consecutive paths
#delta_paths_array = np.diff(current_historical_paths_array, axis=0)
delta_paths = []
for i in range(num_training_years - SCENARIO_HORIZON_YEARS - 2):
    delta_paths.append(current_historical_paths_array[i + SCENARIO_HORIZON_YEARS] - current_historical_paths_array[i])
delta_paths_array = np.array(delta_paths)
print(f"  Shape of Delta Paths Array (differences between consecutive paths): {delta_paths_array.shape}")

try:
    # Compute covariance of these *differences*
    cov_matrix_historical_changes = np.cov(delta_paths_array, rowvar=False)
    inv_cov_matrix_historical_changes = np.linalg.pinv(cov_matrix_historical_changes)
    print(f"  Inverse of Covariance Matrix: {inv_cov_matrix_historical_changes}")
except Exception as e:
    print(f"  Error estimating covariance matrix from path differences for {prediction_year_start}: {e}. Interrupting.")
    predicted_probabilities_over_time.append({
        'year': prediction_year_start,
        **{s_name: np.nan for s_name in gic_scenarios.keys()}
    })

    # --- 4.3. Mahalanobis Distance Calculation & Probability Assignment ---
current_scenario_likelihoods = {}

    # The GIC paper states: "Using the three-year paths for these variables ending in 2019 as the anchor"
    # This is the LAST path generated in current_historical_paths_array.
anchor_path_vector = current_historical_paths_array[-1]
print(f"  Anchor Path (last historical path ending in {training_end_year}): {anchor_path_vector}")

for name, p_path in prospective_scenario_paths.items():
    # Ensure the prospective path has the same dimension as the historical paths
    if p_path.shape != anchor_path_vector.shape:
        raise Exception(f"  Dimension mismatch for scenario {name}. Expected {anchor_path_vector.shape}, got {p_path.shape}. Interrupting.")

    # Calculate Mahalanobis distance. scipy.spatial.distance.mahalanobis returns sqrt(d).
    # The paper's Equation 1 `d` is (x-y)'*Omega^-1*(x-y), which is the squared Mahalanobis distance.
    dist_scipy = mahalanobis(p_path, anchor_path_vector, inv_cov_matrix_historical_changes)
    d_gic_paper = dist_scipy**2 # Square it to match the paper's 'd' in Equation 1

    # Calculate likelihood using the GIC paper's Equation 2: Likelihood ∝ e^(-d/2)
    likelihood = np.exp(-d_gic_paper / 2.0)
    current_scenario_likelihoods[name] = likelihood

valid_likelihoods = {k: v for k, v in current_scenario_likelihoods.items() if not np.isinf(v) and not np.isnan(v)}

if not valid_likelihoods:
    predicted_probabilities_over_time.append({
        'year': prediction_year_start,
        **{s_name: np.nan for s_name in gic_scenarios.keys()}
    })
    raise Exception(f"  No valid likelihoods calculated for {prediction_year_start}. Interrupting probability assignment.")

# Normalize likelihoods to sum to 1 to get probabilities
total_likelihood = sum(valid_likelihoods.values())
print(f"  Total Likelihood: {total_likelihood}")

current_scenario_probabilities = {}
if total_likelihood == 0:
    print(f"  Total likelihood is zero for {prediction_year_start}. Assigning 0.0 to all probabilities.")
    for name in gic_scenarios.keys():
        current_scenario_probabilities[name] = 0.0
else:
    for name, likelihood in valid_likelihoods.items():
        current_scenario_probabilities[name] = likelihood / total_likelihood

    # Ensure all scenarios are present, even if with 0 probability if they were invalid
full_year_probabilities = {s_name: current_scenario_probabilities.get(s_name, 0.0) for s_name in gic_scenarios.keys()}
full_year_probabilities['year'] = prediction_year_start
predicted_probabilities_over_time.append(full_year_probabilities)

# Convert results to a DataFrame for easy analysis
predicted_probabilities_df = pd.DataFrame(predicted_probabilities_over_time).set_index('year')
print("\nPredicted Scenario Probabilities (2019 - based on GIC Paper Exact Methodology):")
print(predicted_probabilities_df)

    # --- 5. Asset Return Data ---
# Create DataFrame to store Asset Returns Data
asset_returns_df = pd.DataFrame()
asset_returns_df.index = historical_macro_data_full.index

# Cash Returns and YoY Change (Interest Rate Change)
asset_returns_df["Cash Total Return"] = historical_macro_data_full["bill_rate"] * 100
asset_returns_df["Cash Real Return"] = asset_returns_df["Cash Total Return"] - macro_var_df["Inflation"]
asset_returns_df["Interest Rate Change (Cash YoY Change)"] = asset_returns_df["Cash Total Return"].diff()

# Stock Returns
asset_returns_df["Stock Total Return"] = historical_macro_data_full["eq_tr"] * 100
asset_returns_df["Stock Real Return"] = asset_returns_df["Stock Total Return"] - macro_var_df["Inflation"]
asset_returns_df["Stock Excess Return"] = asset_returns_df["Stock Real Return"] - asset_returns_df["Cash Real Return"]

# Bond Returns
asset_returns_df["Bond Total Return"] = historical_macro_data_full["bond_tr"] * 100
asset_returns_df["Bond Real Return"] = asset_returns_df["Bond Total Return"] - macro_var_df["Inflation"]
asset_returns_df["Bond Excess Return"] = asset_returns_df["Bond Real Return"] - asset_returns_df["Cash Real Return"]

# Remove NaN
asset_returns_df.dropna(inplace=True)

print(asset_returns_df.head())
print(asset_returns_df.tail())

    # --- 6. Create Path Vectors for Asset Return Data ---
#create the 3 year paths for each Asset Class

asset_returns_prediction = []

# Define the training period end year (end of the year PRIOR to prediction_year_start)
training_end_year = prediction_year_start - 1

current_training_asset_returns = asset_returns_df.iloc[
    (asset_returns_df.index >= pd.to_datetime(FULL_MACRO_START_DATE).year - 2) & (asset_returns_df.index <= training_end_year)
].copy()

if len(current_training_asset_returns) < SCENARIO_HORIZON_YEARS + 1:
    print()
    predicted_probabilities_over_time.append({
        'year': prediction_year_start,
        **{s_name: np.nan for s_name in gic_scenarios.keys()}
    })
    raise Exception(f"Interrupting {prediction_year_start}: Not enough training data before {training_end_year} to form sufficient paths.")

print(f"\n--- Predicting Asset Returns for {prediction_year_start} Path using training data up to {training_end_year} ---")

asset_historical_paths = []
num_training_years = len(current_training_asset_returns)

for i in range(num_training_years - SCENARIO_HORIZON_YEARS + 1):
    path_segment = current_training_asset_returns.iloc[i : i + SCENARIO_HORIZON_YEARS]
    asset_flattened_path = create_path_vector(
        path_segment['Interest Rate Change (Cash YoY Change)'].values,
        path_segment['Stock Excess Return'].values,
        path_segment['Bond Excess Return'].values
    )
    asset_historical_paths.append(asset_flattened_path)

# Transform paths into Arrays
asset_historical_paths_array = np.array(asset_historical_paths)
print(f"  Shape of current Historical Paths Array (all overlapping paths): {asset_historical_paths_array.shape}")

    # --- 7. Calculate Relevance for Sample Partial Regression

x_mean = current_historical_paths_array.mean(axis=0)
y_mean = asset_historical_paths_array.mean(axis=0)
prospective_relevant_indices = {}
prospective_scenario_relevance = {}

for name, p_path in prospective_scenario_paths.items():
    relevance = []
    for h_path in current_historical_paths_array:
        # Ensure the prospective path has the same dimension as the historical paths
        if p_path.shape != h_path.shape:
            raise Exception(f"  Dimension mismatch for scenario {name}. Expected {h_path.shape}, got {p_path.shape}. Interrupting.")
        # Calculate relevance for each path each year
        relevance.append(similarity(h_path, p_path, inv_cov_matrix_historical_changes) + informativeness(h_path, x_mean, inv_cov_matrix_historical_changes))
        prospective_scenario_relevance[name] = relevance
        prospective_relevant_indices[name] = np.argsort(relevance)[-(int(len(relevance) * 0.25)):]

prospecive_asset_returns = {}

for name, indices in prospective_relevant_indices.items():
    asset_returns = []

    # Extract Dependant Variables and Corresponding
    Y_top = asset_historical_paths_array[indices]
    relevance_top = np.array(prospective_scenario_relevance[name])[indices]

    n = len(Y_top)
    adjustment = (1/(2*n)) * np.sum([relevance_top[i] * (Y_top[i] - y_mean) for i in range(n)], axis=0)

    prospecive_asset_returns[name] = y_mean + adjustment

cash_returns = pd.DataFrame()
stock_returns = pd.DataFrame()
bond_returns = pd.DataFrame()

for name, p_return in prospecive_asset_returns.items():
    cash_returns[name] = np.cumsum(p_return[:3])
    stock_returns[name] = cash_returns[name] + p_return[3:6]
    bond_returns[name] = cash_returns[name] + p_return[6:9]

print(f"Cash Return:\n{cash_returns}\nStock Returns:\n{stock_returns}\nBond Returns:\n{bond_returns}")

# ===============================================================================
# 8. EXPECTIMINIMAX PORTFOLIO OPTIMIZATION WITH RISK AVERSION PROFILES
# ===============================================================================

print(f"\n{'='*80}")
print("EXPECTIMINIMAX PORTFOLIO OPTIMIZATION")
print(f"{'='*80}")

# Convert your results to the format needed for expectiminimax optimization
def convert_gic_results_to_expectiminimax_format(predicted_probabilities_df, cash_returns, stock_returns, bond_returns):
    """
    Convert GIC results to the format needed for expectiminimax optimization.
    """
    # Get probabilities for the prediction year (2020)
    latest_probs = predicted_probabilities_df.iloc[-1]  # Last row
    probabilities = {}
    for scenario in gic_scenarios.keys():
        probabilities[scenario] = latest_probs[scenario]

    # Convert returns to the required format (DataFrames with proper structure)
    asset_returns_dict = {}
    for scenario in gic_scenarios.keys():
        if scenario in cash_returns.columns:
            # Create DataFrame with returns for each year of the 3-year path
            returns_df = pd.DataFrame({
                'Cash': cash_returns[scenario].values,
                'Stocks': stock_returns[scenario].values,
                'Bonds': bond_returns[scenario].values
            })
            asset_returns_dict[scenario] = returns_df

    return probabilities, asset_returns_dict

# Convert your GIC results
scenario_probabilities, asset_returns_formatted = convert_gic_results_to_expectiminimax_format(
    predicted_probabilities_df, cash_returns, stock_returns, bond_returns
)

print(f"\nScenario Probabilities for {prediction_year_start}:")
for scenario, prob in scenario_probabilities.items():
    print(f"  {scenario}: {prob:.4f} ({prob*100:.2f}%)")

print(f"\nAsset Returns by Scenario (3-year paths):")
for scenario, returns_df in asset_returns_formatted.items():
    print(f"\n{scenario}:")
    print(returns_df.round(4))

# Initialize the Expectiminimax Portfolio Optimizer
class ExpectiminimaxPortfolioOptimizer:
    """Expectiminimax Portfolio Optimizer for Path-Dependent Scenarios"""

    def __init__(self, scenarios_probabilities, asset_returns, risk_aversion=1.0,
                 min_weight=0.0, max_weight=1.0, use_minimax_within_scenario=False):
        self.scenarios_probabilities = scenarios_probabilities
        self.asset_returns = asset_returns
        self.risk_aversion = risk_aversion
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.use_minimax_within_scenario = use_minimax_within_scenario

        # Get asset information
        first_scenario = list(asset_returns.keys())[0]
        self.asset_names = list(asset_returns[first_scenario].columns)
        self.n_assets = len(self.asset_names)
        self.n_years = len(asset_returns[first_scenario])

    def portfolio_return_path(self, weights, scenario_name):
        """Calculate portfolio returns for a given scenario and weights."""
        returns_df = self.asset_returns[scenario_name]
        portfolio_returns = []

        for year in range(self.n_years):
            year_return = sum(weights[i] * returns_df.iloc[year, i] for i in range(self.n_assets))
            portfolio_returns.append(year_return)

        return np.array(portfolio_returns)

    def utility_function(self, returns):
        """Calculate utility using mean-variance utility function."""
        mean_return = np.mean(returns)

        # Handle risk-neutral case (risk_aversion = 0)
        if self.risk_aversion == 0.0:
            return mean_return

        variance = np.var(returns)
        utility = mean_return - (self.risk_aversion / 2) * variance
        return utility

    def scenario_value(self, weights, scenario_name):
        """Calculate the value for a specific scenario given portfolio weights."""
        portfolio_returns = self.portfolio_return_path(weights, scenario_name)

        if self.use_minimax_within_scenario:
            # Minimax: consider worst single-year return within the scenario
            return np.min(portfolio_returns)
        else:
            # Expected utility over the scenario path
            return self.utility_function(portfolio_returns)

    def expectiminimax_value(self, weights):
        """Calculate the expectiminimax value for given portfolio weights."""
        expected_value = 0.0

        for scenario_name, probability in self.scenarios_probabilities.items():
            if scenario_name in self.asset_returns:
                scenario_val = self.scenario_value(weights, scenario_name)
                expected_value += probability * scenario_val

        return expected_value

    def optimize_portfolio(self, initial_weights=None):
        """Find optimal portfolio weights using expectiminimax criterion."""
        from scipy.optimize import minimize

        # Initial weights (equal allocation if not provided)
        if initial_weights is None:
            initial_weights = np.ones(self.n_assets) / self.n_assets

        # Constraints: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

        # Bounds: weight limits for each asset
        bounds = [(self.min_weight, self.max_weight) for _ in range(self.n_assets)]

        # Objective function (minimize negative expectiminimax value)
        def objective(weights):
            return -self.expectiminimax_value(weights)

        # Optimize
        result = minimize(
            objective, initial_weights, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'ftol': 1e-12, 'disp': False}
        )

        optimal_weights = result.x
        optimal_value = -result.fun

        return {
            'optimal_weights': dict(zip(self.asset_names, optimal_weights)),
            'expectiminimax_value': optimal_value,
            'optimization_success': result.success,
            'optimization_message': result.message if hasattr(result, 'message') else 'No message'
        }

# Define different risk aversion profiles
risk_aversion_profiles = [
    {"name": "Risk Neutral", "risk_aversion": 0.0},
    {"name": "Low Risk Aversion", "risk_aversion": 0.5},
    {"name": "Moderate Risk Aversion", "risk_aversion": 1.0},
    {"name": "High Risk Aversion", "risk_aversion": 2.0},
    {"name": "Very High Risk Aversion", "risk_aversion": 5.0}
]

print(f"\n{'='*80}")
print("OPTIMAL PORTFOLIOS FOR DIFFERENT RISK AVERSION PROFILES")
print(f"{'='*80}")

optimization_results = {}

for profile in risk_aversion_profiles:
    name = profile["name"]
    risk_aversion = profile["risk_aversion"]

    print(f"\n{'-'*50}")
    print(f"RISK AVERSION PROFILE: {name}")
    print(f"Risk Aversion Parameter (λ): {risk_aversion}")
    print(f"{'-'*50}")

    # Special case explanation for risk neutral
    if risk_aversion == 0.0:
        print("IMPLICATIONS OF RISK NEUTRAL (λ=0):")
        print("• Utility function becomes: U = E[returns] (no variance penalty)")
        print("• Only cares about expected returns, ignores volatility completely")
        print("• Will choose allocation with highest expected return across scenarios")
        print("• Represents pure expected return maximization")
        print("• No consideration for downside risk or return distribution")

    # Initialize optimizer
    optimizer = ExpectiminimaxPortfolioOptimizer(
        scenarios_probabilities=scenario_probabilities,
        asset_returns=asset_returns_formatted,
        risk_aversion=risk_aversion,
        min_weight=0.0,   # Allow no investment in an asset
        max_weight=1.0,   # Allow up to 100% in any asset
        use_minimax_within_scenario=False  # Use expected utility, not minimax
    )

    # Optimize portfolio
    result = optimizer.optimize_portfolio()
    optimization_results[name] = result

    print(f"Optimization Success: {result['optimization_success']}")
    if not result['optimization_success']:
        print(f"Warning: {result['optimization_message']}")

    print(f"Expectiminimax Value: {result['expectiminimax_value']:.6f}")
    print("Optimal Portfolio Weights:")
    for asset, weight in result['optimal_weights'].items():
        print(f"  {asset}: {weight:.4f} ({weight*100:.2f}%)")

    # Calculate detailed portfolio performance for each scenario
    print("\nPortfolio Performance by Scenario:")
    total_expected_return = 0.0
    total_expected_utility = 0.0

    for scenario, prob in scenario_probabilities.items():
        if scenario in asset_returns_formatted:
            weights_array = np.array([result['optimal_weights'][asset] for asset in optimizer.asset_names])
            portfolio_returns = optimizer.portfolio_return_path(weights_array, scenario)
            scenario_value = optimizer.scenario_value(weights_array, scenario)

            avg_return = np.mean(portfolio_returns)
            cumulative_return = np.prod(1 + portfolio_returns/100) - 1
            volatility = np.std(portfolio_returns)

            total_expected_return += prob * avg_return
            total_expected_utility += prob * scenario_value

            print(f"  {scenario} (p={prob:.3f}):")
            print(f"    Avg Annual Return: {avg_return:.2f}%")
            print(f"    3-Year Cumulative: {cumulative_return*100:.2f}%")
            print(f"    Volatility: {volatility:.2f}%")
            print(f"    Scenario Utility: {scenario_value:.4f}")

    print(f"\nExpected Portfolio Return: {total_expected_return:.2f}%")
    print(f"Expected Portfolio Utility: {total_expected_utility:.4f}")

# Summary comparison across risk aversion levels
print(f"\n{'='*80}")
print("SUMMARY: OPTIMAL PORTFOLIOS ACROSS RISK AVERSION LEVELS")
print(f"{'='*80}")

summary_df_data = []
for name, result in optimization_results.items():
    profile = next(p for p in risk_aversion_profiles if p["name"] == name)
    row = {
        'Profile': name,
        'Risk Aversion (λ)': profile['risk_aversion'],
        'Expectiminimax Value': result['expectiminimax_value'],
        **{f'{asset} Weight': result['optimal_weights'][asset]
           for asset in ['Cash', 'Stocks', 'Bonds']}
    }
    summary_df_data.append(row)

summary_df = pd.DataFrame(summary_df_data)
print(summary_df.round(4))

# Detailed analysis of risk aversion effects
print(f"\n{'='*80}")
print("RISK AVERSION IMPACT ANALYSIS")
print(f"{'='*80}")

print("\nKey Insights:")
print("1. RISK AVERSION (λ) controls the penalty for portfolio volatility")
print("2. λ=0 (Risk Neutral): Only maximizes expected return, ignores risk")
print("3. λ>0 (Risk Averse): Higher λ → stronger preference for stable returns")
print("4. Portfolio allocation shifts from aggressive to conservative as λ increases")
print("5. Expected utility decreases as risk aversion increases (cost of safety)")

print(f"\nRisk Aversion Parameter Effects:")
print("• λ=0.0: Pure expected return maximization (no risk penalty)")
print("• λ=0.5: Slight risk adjustment, still growth-oriented")
print("• λ=1.0: Balanced risk-return trade-off")
print("• λ=2.0: Conservative bias, stability over growth")
print("• λ=5.0: Very conservative, strong volatility avoidance")

# Highlight the recommended portfolio based on moderate risk aversion
recommended_profile = "Moderate Risk Aversion"
if recommended_profile in optimization_results:
    recommended_portfolio = optimization_results[recommended_profile]

    print(f"\n{'='*80}")
    print(f"RECOMMENDED PORTFOLIO: {recommended_profile}")
    print(f"{'='*80}")
    print(f"Based on expectiminimax optimization with balanced risk preferences")
    print(f"Risk aversion parameter: λ = 1.0")
    print(f"Expected utility: {recommended_portfolio['expectiminimax_value']:.6f}")
    print(f"\nOptimal Asset Allocation:")
    for asset, weight in recommended_portfolio['optimal_weights'].items():
        print(f"  {asset}: {weight:.1%}")

# Calculate and display risk metrics for different profiles
print(f"\n{'='*80}")
print("RISK-RETURN ANALYSIS ACROSS PROFILES")
print(f"{'='*80}")

for name, result in optimization_results.items():
    weights_array = np.array([result['optimal_weights'][asset] for asset in ['Cash', 'Stocks', 'Bonds']])

    # Calculate expected return and risk across scenarios
    expected_return = 0.0
    expected_variance = 0.0
    worst_case_return = float('inf')
    best_case_return = float('-inf')

    for scenario, prob in scenario_probabilities.items():
        if scenario in asset_returns_formatted:
            optimizer = ExpectiminimaxPortfolioOptimizer(scenario_probabilities, asset_returns_formatted, 1.0)
            portfolio_returns = optimizer.portfolio_return_path(weights_array, scenario)

            scenario_avg_return = np.mean(portfolio_returns)
            scenario_variance = np.var(portfolio_returns)
            scenario_worst = np.min(portfolio_returns)
            scenario_best = np.max(portfolio_returns)

            expected_return += prob * scenario_avg_return
            expected_variance += prob * scenario_variance

            if scenario_worst < worst_case_return:
                worst_case_return = scenario_worst
            if scenario_best > best_case_return:
                best_case_return = scenario_best

    expected_volatility = np.sqrt(expected_variance)

    print(f"{name}:")
    print(f"  Expected Return: {expected_return:.2f}%")
    print(f"  Expected Volatility: {expected_volatility:.2f}%")
    if expected_volatility > 0:
        print(f"  Sharpe Ratio: {(expected_return - 0.041)/ expected_volatility:.3f}")
    print(f"  Worst Case: {worst_case_return:.2f}%")
    print(f"  Best Case: {best_case_return:.2f}%")
    print("")

print(f"\n{'='*80}")
print("EXPECTIMINIMAX PORTFOLIO OPTIMIZATION COMPLETE")
print(f"{'='*80}")
print("✓ Risk aversion-based investor profiles (no liquidity constraints)")
print("✓ Continuous optimization for optimal portfolio weights")
print("✓ Risk neutral case (λ=0) shows pure expected return maximization")
print("✓ Higher risk aversion leads to more conservative allocations")
print("✓ Expectiminimax framework maximizes expected utility across scenarios")

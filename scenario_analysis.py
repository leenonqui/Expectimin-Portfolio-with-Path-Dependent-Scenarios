"""
Simplified Scenario Analysis Module
Implements Section 3.1 of the thesis methodology
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from constants import (
    SCENARIOS, DATA_COLUMNS, HORIZON, PSR_TOP_PERCENTILE,
    HISTORICAL_START_YEAR, MIN_HISTORICAL_YEARS
)
from utils import (
    create_path_vector, calculate_relevance, calculate_mahalanobis_distance,
    scenario_likelihood, calculate_real_return, prepare_historical_paths,
    calculate_path_differences, safe_matrix_inverse, normalize_probabilities
)

class ScenarioAnalyzer:
    """
    Implements scenario probability estimation (Section 3.1.2)
    and asset return forecasting (Section 3.1.3)
    """

    def __init__(self, data_path: str, horizon: int = HORIZON):
        """
        Args:
            data_path: Path to historical data CSV
            horizon: Investment horizon T (default from constants)
        """
        self.horizon = horizon
        self.data = self._load_data(data_path)
        self.scenarios = SCENARIOS

    def _load_data(self, path: str) -> pd.DataFrame:
        """Load and prepare historical data"""
        df = pd.read_csv(path, sep=';', index_col=DATA_COLUMNS['year'])

        # Convert European decimal format if needed
        numeric_cols = [DATA_COLUMNS[key] for key in ['gdp', 'cpi', 'cash', 'stocks', 'bonds']]
        for col in numeric_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col].str.replace(',', '.'), errors='coerce')

        # Calculate growth rates
        df['gdp_growth'] = df[DATA_COLUMNS['gdp']].pct_change() * 100
        df['inflation'] = df[DATA_COLUMNS['cpi']].pct_change() * 100

        # Keep original columns for return calculations
        df['bond_tr'] = df[DATA_COLUMNS['bonds']]
        df['eq_tr'] = df[DATA_COLUMNS['stocks']]
        df['bill_rate'] = df[DATA_COLUMNS['cash']]

        return df.dropna()

    def estimate_probabilities(self, anchor_year: int) -> Dict[str, float]:
        """
        Estimate scenario probabilities using Mahalanobis distance
        Implements Section 3.1.2 of the thesis

        Args:
            anchor_year: End year of anchor path γ

        Returns:
            Dictionary mapping scenario names to probabilities P_s
        """
        # Validate data availability
        if anchor_year - self.horizon + 1 < self.data.index.min() + MIN_HISTORICAL_YEARS:
            raise ValueError(f"Insufficient historical data before {anchor_year}")

        # Step 1: Define anchor path γ
        anchor_path = self._get_anchor_path(anchor_year)

        # Step 2: Calculate covariance matrix Ω
        omega = self._calculate_covariance_matrix(anchor_year, 3)
        omega_inv = safe_matrix_inverse(omega)

        # Steps 3-4: Calculate Mahalanobis distances and likelihoods
        likelihoods = {}
        for name, scenario_def in self.scenarios.items():
            x_s = create_path_vector(scenario_def.gdp_growth, scenario_def.inflation)

            # Mahalanobis distance: d_s = (x_s - γ)' Ω^(-1) (x_s - γ)
            d_s = calculate_mahalanobis_distance(x_s, anchor_path, omega_inv)

            # Likelihood: L_s ∝ exp(-d_s/2)
            likelihoods[name] = scenario_likelihood(d_s)

        # Step 5: Normalize to probabilities
        probabilities = normalize_probabilities(likelihoods)

        return probabilities

    def _get_anchor_path(self, end_year: int) -> np.ndarray:
        """Extract anchor path γ from historical data"""
        start_year = end_year - self.horizon + 1

        path_data = self.data.loc[start_year:end_year]
        gdp_path = path_data['gdp_growth'].values
        inf_path = path_data['inflation'].values

        return create_path_vector(gdp_path.tolist(), inf_path.tolist())

    def _calculate_covariance_matrix(self, end_year: int, lag=0) -> np.ndarray:
        """
        Calculate covariance matrix Ω from path differences
        Following thesis methodology (consecutive path differences)
        """        # Generate all historical T-year paths
        paths, _ = prepare_historical_paths(self.data, self.horizon, end_year)

        if lag == 0:
            return np.cov(paths)


        # Calculate differences between consecutive paths
        path_differences = calculate_path_differences(paths, lag=lag)

        # Covariance matrix of path differences
        return np.cov(path_differences)

    def forecast_returns(self, anchor_year: int,
                        top_percentile: float = PSR_TOP_PERCENTILE) -> Dict[str, Dict[str, List[float]]]:
        """
        Forecast asset returns using partial sample regression
        Implements Section 3.1.3 of the thesis

        Args:
            anchor_year: End year for historical data
            top_percentile: Fraction of most relevant observations to use

        Returns:
            Nested dict: {scenario_name: {asset_name: [year1, year2, year3]}}
        """
        # Prepare historical data
        hist_paths, hist_returns = self._prepare_historical_data(anchor_year)

        # Get covariance for relevance calculation
        omega_inv = safe_matrix_inverse(self._calculate_covariance_matrix(anchor_year))
        x_bar = hist_paths.mean(axis=0)

        # Get anchor year cash real rate for cumulative calculation
        anchor_cash_real = self.data.loc[anchor_year, 'cash_real']

        # Forecast for each scenario
        forecasts = {}

        for scenario_name, scenario_def in self.scenarios.items():
            x_t = create_path_vector(scenario_def.gdp_growth, scenario_def.inflation)

            # Calculate relevance for all historical paths
            relevances = []
            for x_i in hist_paths:
                relevance = calculate_relevance(x_i, x_t, x_bar, omega_inv)
                relevances.append(relevance)

            # Select top percentile most relevant observations
            n_top = max(1, int(len(relevances) * top_percentile))
            top_indices = np.argsort(relevances)[-n_top:]

            # Forecast using partial sample regression (Equation 6 from thesis)
            scenario_forecasts = {}

            # Apply partial sample regression to the entire return vector
            # hist_returns shape: (n_observations, 9)
            # Get the most relevant historical return vectors
            top_returns = hist_returns[top_indices]  # Shape: (n_top, 9)

            # Calculate y_bar as average of most relevant observations
            y_bar = top_returns.mean(axis=0)  # Shape: (9,)

            # Apply partial sample regression formula for the entire vector
            weighted_sum = np.zeros(9)
            for i, idx in enumerate(top_indices):
                rel = relevances[idx]
                weighted_sum += rel * (hist_returns[idx] - y_bar)

            # Predicted return vector
            y_hat = y_bar + weighted_sum / (2 * n_top)  # Shape: (9,)

            # Extract asset-specific forecasts from predicted vector
            # 0:3 = interest rate changes (cash)
            # 3:6 = stock excess returns
            # 6:9 = bond excess returns

            cash_changes = y_hat[:self.horizon]
            stock_excess = y_hat[self.horizon:2*self.horizon]
            bond_excess = y_hat[2*self.horizon:3*self.horizon]

            # Convert interest rate changes to cumulative cash returns
            cumulative_cash = [anchor_cash_real]
            for change in cash_changes:
                cumulative_cash.append(cumulative_cash[-1] + change)

            scenario_forecasts['Cash'] = cumulative_cash[1:]  # Exclude anchor year

            # Stock returns (excess returns + cash returns)
            stock_forecasts = []
            for year in range(self.horizon):
                stock_total = stock_excess[year] + scenario_forecasts['Cash'][year]
                stock_forecasts.append(stock_total)

            scenario_forecasts['Stocks'] = stock_forecasts

            # Bond returns (excess returns + cash returns)
            bond_forecasts = []
            for year in range(self.horizon):
                bond_total = bond_excess[year] + scenario_forecasts['Cash'][year]
                bond_forecasts.append(bond_total)

            scenario_forecasts['Bonds'] = bond_forecasts

            forecasts[scenario_name] = scenario_forecasts

        return forecasts

    def _prepare_historical_data(self, end_year: int, start_year: int = HISTORICAL_START_YEAR) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare aligned historical paths and returns"""
        macro_paths = []
        asset_paths = []

    # First, calculate real returns for all rows
        cash_real = []
        stock_real = []
        bond_real = []
    
        for _, row in self.data.iterrows():
            cash_real.append(calculate_real_return(row['bill_rate'], row['inflation']))
            stock_real.append(calculate_real_return(row['eq_tr'], row['inflation']))
            bond_real.append(calculate_real_return(row['bond_tr'], row['inflation']))
    
        # Add calculated columns to dataframe
        self.data['cash_real'] = cash_real
        self.data['stock_real'] = stock_real
        self.data['bond_real'] = bond_real
    
        # Calculate derived columns
        self.data['interest_rate_changes'] = self.data['cash_real'].diff()
        self.data['stock_excess_return'] = self.data['stock_real'] - self.data['cash_real']
        self.data['bond_excess_return'] = self.data['bond_real'] - self.data['cash_real']

        self.data.dropna(inplace=True)

        # Get T-year rolling windows
        for start in self.data.index[start_year-2:-self.horizon]: # start_year - 2 to get the paths to end on start year [1927, 1928, 1929]
            if start + self.horizon - 1 > end_year:
                break# Asset returns (real returns)

            # Economic path
            window = self.data.loc[start:start+self.horizon-1]
            gdp_path = window['gdp_growth'].values
            inf_path = window['inflation'].values
            macro_paths.append(create_path_vector(gdp_path.tolist(), inf_path.tolist()))

            # Asset path
            interest_path = window['interest_rate_changes'].values
            stock_path = window['stock_excess_return'].values
            bond_path = window['bond_excess_return'].values
            asset_paths.append(create_path_vector(interest_path.tolist(), stock_path.tolist(), bond_path.tolist()))

        return np.array(macro_paths), np.array(asset_paths)

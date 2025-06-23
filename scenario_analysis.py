"""
Simplified Scenario Analysis Module
Implements Section 3.1 of the thesis methodology
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from constants import (
    SCENARIOS, HORIZON, PSR_TOP_PERCENTILE,
    MIN_HISTORICAL_YEARS, CURRENT_INTEREST_RATE
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
        self.current_rate = calculate_real_return(CURRENT_INTEREST_RATE, 1.2)

    def _load_data(self, path: str) -> pd.DataFrame:
        """Load and prepare all required data columns"""

        # Load raw data
        df = pd.read_csv(path, sep=';', index_col='year')

        # Convert European decimal format if needed
        numeric_cols = ['rgdpmad', 'cpi', 'bill_rate', 'eq_tr', 'bond_tr']
        for col in numeric_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col].str.replace(',', '.'), errors='coerce')

        # Calculate growth rates
        df['gdp_growth'] = df['rgdpmad'].pct_change() * 100
        df['inflation'] = df['cpi'].pct_change() * 100

        # Keep original asset columns with cleaner names
        df['bill_rate'] = df['bill_rate']  # Already named correctly
        df['eq_tr'] = df['eq_tr']          # Already named correctly
        df['bond_tr'] = df['bond_tr']      # Already named correctly

        # Calculate real returns
        df['bill_rr'] = ((1 + df['bill_rate']) / (1 + df['inflation']/100) - 1) * 100
        df['stock_rr'] = ((1 + df['eq_tr']) / (1 + df['inflation']/100) - 1) * 100
        df['bond_rr'] = ((1 + df['bond_tr']) / (1 + df['inflation']/100) - 1) * 100

        # Calculate derived series
        df['bill_rr_changes'] = df['bill_rr'].diff()
        df['stock_excess'] = df['stock_rr'] - df['bill_rr']
        df['bond_excess'] = df['bond_rr'] - df['bill_rr']

        # Filter to our analysis period (1927-2019) and remove NaN
        df = df.loc[1927:2019].dropna()



        return df

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
        print(f"Anchor Path = {anchor_path}\n")

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

    def _get_path(self, end_year: int) -> np.ndarray:
        """Extract path from historical data"""
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
            return np.cov(paths, rowvar=False)


        # Calculate differences between consecutive paths
        path_differences = calculate_path_differences(paths, lag=lag)

        # Covariance matrix of path differences
        return np.cov(path_differences, rowvar=False)

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
        omega_inv = safe_matrix_inverse(self._calculate_covariance_matrix(anchor_year, 0))
        x_bar = hist_paths.mean(axis=0)

        # Get anchor year cash real rate for cumulative calculation
        # Make sure we have this column calculated
        if 'bill_rr' not in self.data.columns:
            cash_real = []
            for _, row in self.data.iterrows():
                cash_real.append(calculate_real_return(row['bill_rate'], row['inflation']))
            self.data['bill_rr'] = cash_real

        anchor_cash_real = self.current_rate
        print(f"Debug: Anchor year ({anchor_year}) cash real rate: {anchor_cash_real:.2f}%")

        # Debug: Check historical return data structure
        print(f"Debug: Historical paths shape: {hist_paths.shape}")
        print(f"Debug: Historical returns shape: {hist_returns.shape}")
        print(f"Debug: Expected returns shape: (n_obs, {3 * self.horizon}) for 3 assets × {self.horizon} years")

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

            print(f"Debug: Scenario {scenario_name} - Using top {n_top} out of {len(relevances)} observations")

            # Apply partial sample regression to the entire return vector
            # hist_returns shape: (n_observations, 9)
            # Get the most relevant historical return vectors
            top_returns = hist_returns[top_indices]  # Shape: (n_top, 9)

            # Calculate y_bar as average of most relevant observations
            y_bar = top_returns.mean(axis=0)  # Shape: (9,)

            # Apply partial sample regression formula for the entire vector
            weighted_sum = np.zeros(9)
            total_relevance = 0
            for i, idx in enumerate(top_indices):
                rel = relevances[idx]
                weighted_sum += rel * (hist_returns[idx] - y_bar)
                total_relevance += rel

            # Predicted return vector
            y_hat = y_bar + weighted_sum / (2 * n_top)  # Shape: (9,)

            # Debug output for first scenario
            if scenario_name == list(self.scenarios.keys())[0]:
                print(f"Debug: y_bar (top {n_top} avg): {y_bar[:3]} (cash changes)")
                print(f"Debug: y_hat (predicted): {y_hat[:3]} (cash changes)")

            # Extract asset-specific forecasts from predicted vector
            # 0:3 = interest rate changes (cash)
            # 3:6 = stock excess returns
            # 6:9 = bond excess returns

            cash_changes = y_hat[:self.horizon]
            stock_excess = y_hat[self.horizon:2*self.horizon]
            bond_excess = y_hat[2*self.horizon:3*self.horizon]

            # Convert interest rate changes to cumulative cash returns
            # Start from anchor year cash rate and add predicted changes
            cash_levels = []
            current_cash_rate = anchor_cash_real

            for change in cash_changes:
                current_cash_rate += change  # Add the predicted change
                cash_levels.append(current_cash_rate)

            scenario_forecasts = {}
            scenario_forecasts['Cash'] = cash_levels

            # Stock returns (excess returns + cash returns)
            stock_forecasts = []
            for year in range(self.horizon):
                # Total stock return = cash return + excess return
                stock_total = scenario_forecasts['Cash'][year] + stock_excess[year]
                stock_forecasts.append(stock_total)

            scenario_forecasts['Stocks'] = stock_forecasts

            # Bond returns (excess returns + cash returns)
            bond_forecasts = []
            for year in range(self.horizon):
                # Total bond return = cash return + excess return
                bond_total = scenario_forecasts['Cash'][year] + bond_excess[year]
                bond_forecasts.append(bond_total)

            scenario_forecasts['Bonds'] = bond_forecasts

            # Debug output for first scenario
            if scenario_name == list(self.scenarios.keys())[0]:
                print(f"Debug: Cash levels: {[f'{x:.2f}%' for x in cash_levels]}")
                print(f"Debug: Stock excess: {[f'{x:.2f}%' for x in stock_excess]}")
                print(f"Debug: Bond excess: {[f'{x:.2f}%' for x in bond_excess]}")
                print(f"Debug: Final stocks: {[f'{x:.2f}%' for x in stock_forecasts]}")
                print(f"Debug: Final bonds: {[f'{x:.2f}%' for x in bond_forecasts]}")

            forecasts[scenario_name] = scenario_forecasts

        return forecasts

    def _prepare_historical_data(self, end_year: int, start_year: int = 1927) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare aligned historical paths and returns"""

        macro_paths = []
        asset_paths = []

        # Create 3-year rolling windows from start_year to end_year
        for year in range(start_year, end_year - self.horizon + 2):  # +2 to include end_year
            window_end = year + self.horizon - 1

            if window_end > end_year:
                break

            # Extract 3-year window
            try:
                window = self.data.loc[year:window_end]

                if len(window) != self.horizon:
                    continue

                # Macro path: [GDP1, GDP2, GDP3, INF1, INF2, INF3]
                gdp_path = window['gdp_growth'].values
                inf_path = window['inflation'].values
                macro_vector = np.concatenate([gdp_path, inf_path], axis=None)
                macro_paths.append(macro_vector)

                # Asset path: [bill_changes1, bill_changes2, bill_changes3,
                #              stock_excess1, stock_excess2, stock_excess3,
                #              bond_excess1, bond_excess2, bond_excess3]
                bill_changes = window['bill_rr_changes'].values
                stock_excess = window['stock_excess'].values
                bond_excess = window['bond_excess'].values
                asset_vector = np.concatenate([bill_changes, stock_excess, bond_excess], axis=None)
                asset_paths.append(asset_vector)

            except (KeyError, IndexError) as e:
                print(f"Debug: Skipping window {year}-{window_end}: {e}")
                continue

        print(f"Debug: Generated {len(macro_paths)} historical paths")
        print(f"Debug: Expected paths from {start_year} to {end_year - self.horizon + 1}: {end_year - self.horizon + 1 - start_year + 1}")

        if len(asset_paths) > 0:
            print(f"Debug: Macro path shape: {np.array(macro_paths).shape} (expected: n_paths × 6)")
            print(f"Debug: Asset path shape: {np.array(asset_paths).shape} (expected: n_paths × 9)")

            # Show sample path
            print(f"Debug: Sample macro path: {macro_paths[0]}")
            print(f"Debug: Sample asset path: {asset_paths[0]}")

        return np.array(macro_paths), np.array(asset_paths)

import numpy as np
import pandas as pd
from typing import Dict, List
from ..data.loader import DataLoader
from ..utils.math_utils import create_path_vector
from .partial_sample_regression import PartialSampleRegression
from .scenario_probability import GICScenarioProbability
from ..config import GIC_SCENARIOS, SCENARIO_HORIZON_YEARS

class GICAssetForecasting:
    """
    Asset return forecasting using GIC partial sample regression methodology

    Converts economic scenarios into asset return paths using:
    - Partial sample regression for relevance-weighted forecasting
    - Path-dependent asset return modeling
    - Conversion from changes/premiums to return levels
    """

    def __init__(self, data_path: str):
        self.data_loader = DataLoader(data_path)
        self.psr = PartialSampleRegression()
        self.macro_data = None
        self.asset_data = None
        self.macro_paths = None
        self.asset_paths = None

    def forecast_returns(self, prediction_year: int) -> Dict[str, Dict[str, List[float]]]:
        """
        Forecast asset returns for all scenarios in given prediction year

        Returns dict mapping scenario names to asset return dictionaries
        """
        training_end_year = prediction_year - 1

        # Load training data
        self.macro_data, self.asset_data = self.data_loader.load_data(end_year=training_end_year)

        if len(self.macro_data) < SCENARIO_HORIZON_YEARS + 1:
            raise ValueError(f"Insufficient data before {training_end_year} for forecasting")

        # Create historical path databases
        self._create_historical_paths()

        # Get covariance matrix from probability estimator
        prob_estimator = GICScenarioProbability(self.data_loader.file_path)
        prob_estimator.macro_data = self.macro_data
        prob_estimator._create_historical_paths()
        prob_estimator._estimate_covariance_matrices()

        # Forecast returns for each scenario
        scenario_returns = {}
        base_cash_level = self.asset_data['Cash_Real_Level'].iloc[-1]

        for scenario_name, scenario_data in GIC_SCENARIOS.items():
            # Create prospective macro path
            prospective_macro_path = create_path_vector(
                scenario_data["GDP Growth"],
                scenario_data["Inflation"]
            )

            # Apply partial sample regression
            asset_forecast, _ = self.psr.forecast(
                self.macro_paths,
                self.asset_paths,
                prospective_macro_path,
                prob_estimator.get_psr_covariance_matrix()
            )

            # Convert to final return levels
            returns = self._convert_to_return_levels(asset_forecast, base_cash_level)
            scenario_returns[scenario_name] = returns

        return scenario_returns

    def _create_historical_paths(self):
        """Create historical path databases for macro and asset variables"""

        # Macro paths
        macro_paths = []
        asset_paths = []
        num_periods = len(self.macro_data)

        for i in range(num_periods - SCENARIO_HORIZON_YEARS + 1):
            # Macro path segment
            macro_segment = self.macro_data.iloc[i:i + SCENARIO_HORIZON_YEARS]
            macro_path = create_path_vector(
                macro_segment['GDP Growth'].values,
                macro_segment['Inflation'].values
            )
            macro_paths.append(macro_path)

            # Corresponding asset path segment
            asset_segment = self.asset_data.iloc[i:i + SCENARIO_HORIZON_YEARS]
            asset_path = create_path_vector(
                asset_segment['Cash_YoY_Change'].values,
                asset_segment['Stock_Excess'].values,
                asset_segment['Bond_Excess'].values
            )
            asset_paths.append(asset_path)

        self.macro_paths = np.array(macro_paths)
        self.asset_paths = np.array(asset_paths)

    def _convert_to_return_levels(self, asset_forecast: np.ndarray, base_cash_level: float):
        """Build forward path from current levels using forecasted changes"""

        cash_changes = asset_forecast[:3]  # These are YoY changes
        stock_excess = asset_forecast[3:6]
        bond_excess = asset_forecast[6:9]

        # Build the path forward from current interest rate level
        cash_returns = []
        current_level = base_cash_level

        for change in cash_changes:
            current_level += change  # Add forecasted change
            cash_returns.append(current_level)

        # Add premiums to get stock/bond returns
        stock_returns = [cash_returns[i] + stock_excess[i] for i in range(3)]
        bond_returns = [cash_returns[i] + bond_excess[i] for i in range(3)]

        return {
            'Cash': cash_returns,
            'Stocks': stock_returns,
            'Bonds': bond_returns
        }

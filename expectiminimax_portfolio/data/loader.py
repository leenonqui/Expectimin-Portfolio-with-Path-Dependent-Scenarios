
import pandas as pd
import numpy as np
from typing import Tuple
from ..config import JST_COLUMNS, FULL_MACRO_START_DATE, FULL_MACRO_END_DATE, SCENARIO_HORIZON_YEARS

class DataLoader:
    """Load and preprocess JST macrohistory data"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._raw_data = None
        self._macro_data = None
        self._asset_data = None

    def load_data(self, start_year: int = None, end_year: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and process historical macro and asset data"""

        if start_year is None:
            start_year = int(FULL_MACRO_START_DATE)
        if end_year is None:
            end_year = int(FULL_MACRO_END_DATE)

        # Load raw data
        try:
            self._raw_data = pd.read_csv(self.file_path, index_col='year')
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        except Exception as e:
            raise ValueError(f"Error loading data: {e}")

        # Filter to desired range
        filtered_data = self._raw_data.loc[
            (self._raw_data.index >= start_year - SCENARIO_HORIZON_YEARS) &
            (self._raw_data.index <= end_year)
        ]

        # Process macro variables
        self._macro_data = self._process_macro_data(filtered_data)

        # Process asset data
        self._asset_data = self._process_asset_data(filtered_data, self._macro_data)

        # Align dataframes
        common_index = self._macro_data.index.intersection(self._asset_data.index)
        self._macro_data = self._macro_data.loc[common_index]
        self._asset_data = self._asset_data.loc[common_index]

        return self._macro_data, self._asset_data

    def _process_macro_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Process macroeconomic variables"""
        macro_df = pd.DataFrame(index=raw_data.index)

        # Real GDP growth (percentage change)
        macro_df['GDP Growth'] = raw_data[JST_COLUMNS['gdp']].pct_change() * 100

        # Inflation (percentage change in CPI)
        macro_df['Inflation'] = raw_data[JST_COLUMNS['inflation']].pct_change() * 100

        # Remove NaN values
        macro_df.dropna(inplace=True)

        return macro_df

    def _process_asset_data(self, raw_data: pd.DataFrame, macro_data: pd.DataFrame) -> pd.DataFrame:
        """Process asset return data"""
        asset_df = pd.DataFrame(index=raw_data.index)

        # Convert to percentages
        cash_total = raw_data[JST_COLUMNS['cash']] * 100
        stock_total = raw_data[JST_COLUMNS['stocks']] * 100
        bond_total = raw_data[JST_COLUMNS['bonds']] * 100

        # Calculate real returns
        inflation = macro_data['Inflation']
        cash_real = cash_total - inflation
        stock_real = stock_total - inflation
        bond_real = bond_total - inflation

        # Asset variables for regression
        temp = np.zeros(cash_real.diff().shape)
        for i in range(len(temp)):
            if i == 0:
                temp[i] = cash_real[-1]
            else:
                temp[i] = temp[i-1] + cash_real.diff()[i]

        asset_df['Cash_YoY_Change'] = temp
        asset_df['Stock_Excess'] = stock_real - cash_real
        asset_df['Bond_Excess'] = bond_real - cash_real

        # Store levels for final calculations
        asset_df['Cash_Real_Level'] = cash_real
        asset_df['Stock_Real_Level'] = stock_real
        asset_df['Bond_Real_Level'] = bond_real

        # Remove NaN values
        asset_df.dropna(inplace=True)

        return asset_df

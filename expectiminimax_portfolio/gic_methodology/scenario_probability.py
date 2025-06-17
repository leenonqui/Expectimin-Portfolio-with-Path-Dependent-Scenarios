import numpy as np
import pandas as pd
from typing import Dict, List
from ..data.loader import DataLoader
from ..utils.math_utils import create_path_vector, mahalanobis_distance_squared, scenario_likelihood
from ..config import GIC_SCENARIOS, SCENARIO_HORIZON_YEARS

class GICScenarioProbability:
    """
    Scenario probability estimation using GIC methodology

    Implements the approach from Kritzman et al. (2021) using:
    - Path-dependent scenario definitions
    - Mahalanobis distance for similarity measurement
    - Covariance estimation from path differences
    """

    def __init__(self, data_path: str):
        self.data_loader = DataLoader(data_path)
        self.macro_data = None
        self.historical_paths = None
        self.covariance_matrix = None

    def calculate_probabilities(self, prediction_year: int) -> Dict[str, float]:
        """
        Calculate scenario probabilities for given prediction year

        Uses training data up to (prediction_year - 1) to estimate probabilities
        for prospective scenarios using Mahalanobis distance methodology
        """
        training_end_year = prediction_year - 1

        # Load historical data up to training end year
        self.macro_data, _ = self.data_loader.load_data(end_year=training_end_year)

        if len(self.macro_data) < SCENARIO_HORIZON_YEARS + 1:
            raise ValueError(f"Insufficient data before {training_end_year} for scenario analysis")

        # Create historical path database
        self._create_historical_paths()

        # Estimate covariance matrix from path differences
        self._estimate_covariance_matrix()

        # Calculate probabilities for each prospective scenario
        scenario_likelihoods = self._calculate_scenario_likelihoods()

        # Normalize to probabilities
        total_likelihood = sum(scenario_likelihoods.values())
        if total_likelihood == 0:
            raise ValueError("All scenario likelihoods are zero - check data quality")

        probabilities = {
            name: likelihood / total_likelihood
            for name, likelihood in scenario_likelihoods.items()
        }

        return probabilities

    def _create_historical_paths(self):
        """Create overlapping 3-year paths from historical macro data"""
        paths = []
        num_periods = len(self.macro_data)

        # Generate all overlapping 3-year paths
        for i in range(num_periods - SCENARIO_HORIZON_YEARS + 1):
            path_segment = self.macro_data.iloc[i:i + SCENARIO_HORIZON_YEARS]

            # Create path vector [GDP1, GDP2, GDP3, INF1, INF2, INF3]
            path_vector = create_path_vector(
                path_segment['GDP Growth'].values,
                path_segment['Inflation'].values
            )
            paths.append(path_vector)

        self.historical_paths = np.array(paths)

    def _estimate_covariance_matrix(self):
        """
        Estimate covariance matrix from changes in path values

        Following GIC paper: "We computed the covariance matrix from the changes
        in the values of the economic variables from one three-year period to
        the next three-year period"
        """
        if len(self.historical_paths) < 2:
            raise ValueError("Need at least 2 historical paths to compute differences")

        # Calculate differences between consecutive paths
        delta_paths = []
        for i in range(len(self.historical_paths) - SCENARIO_HORIZON_YEARS):
            delta = self.historical_paths[i + SCENARIO_HORIZON_YEARS] - self.historical_paths[i]
            delta_paths.append(delta)

        delta_paths_array = np.array(delta_paths)

        # Compute covariance of path differences
        self.covariance_matrix = np.cov(delta_paths_array, rowvar=False)

        # Handle singular matrices
        try:
            self.inv_covariance_matrix = np.linalg.inv(self.covariance_matrix)
        except np.linalg.LinAlgError:
            self.inv_covariance_matrix = np.linalg.pinv(self.covariance_matrix)

    def _calculate_scenario_likelihoods(self) -> Dict[str, float]:
        """Calculate likelihood for each prospective scenario"""

        # Use last historical path as anchor (most recent economic experience)
        anchor_path = self.historical_paths[-1]

        scenario_likelihoods = {}

        for scenario_name, scenario_data in GIC_SCENARIOS.items():
            # Create prospective scenario path vector
            prospective_path = create_path_vector(
                scenario_data["GDP Growth"],
                scenario_data["Inflation"]
            )

            # Calculate Mahalanobis distance squared (GIC Equation 1)
            d_squared = mahalanobis_distance_squared(
                prospective_path,
                anchor_path,
                self.inv_covariance_matrix
            )

            # Calculate likelihood (GIC Equation 2)
            likelihood = scenario_likelihood(d_squared)
            scenario_likelihoods[scenario_name] = likelihood

        return scenario_likelihoods

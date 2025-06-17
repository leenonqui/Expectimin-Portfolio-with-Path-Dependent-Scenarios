"""
Expectiminimax Portfolio Optimization with Path-Dependent Scenarios and Liquidity Preferences

This package implements the GIC methodology for scenario analysis and
expectiminimax framework for portfolio optimization with enhanced liquidity preferences.

Based on:
Kritzman, M., Li, D., Qiu, G., & Turkington, D. (2021).
"Portfolio Choice with Path-Dependent Scenarios."

Enhanced with liquidity preference constraints as minimum cash allocation bounds.
"""

from .gic_methodology.scenario_probability import GICScenarioProbability
from .gic_methodology.asset_return_forecasting import GICAssetForecasting
from .portfolio_optimization.expectiminimax import ExpectiminimaxOptimizer
from .models.scenario import EconomicScenario, AssetReturns
from .models.portfolio import OptimizationResult

__version__ = "0.2.0"
__author__ = "Your Name"

class GICAnalyzer:
    """
    Main interface for GIC methodology implementation

    Combines scenario probability estimation and asset return forecasting
    following the exact methodology from Kritzman et al. (2021)
    """

    def __init__(self, data_path: str):
        self.probability_estimator = GICScenarioProbability(data_path)
        self.return_forecaster = GICAssetForecasting(data_path)

    def analyze(self, prediction_year: int = 2020):
        """Run complete GIC analysis for given prediction year"""
        # Calculate scenario probabilities using Mahalanobis distance
        probabilities = self.probability_estimator.calculate_probabilities(prediction_year)

        # Forecast asset returns using partial sample regression
        returns = self.return_forecaster.forecast_returns(prediction_year)

        return GICResults(probabilities, returns)

class GICResults:
    """Container for GIC analysis results"""

    def __init__(self, probabilities: dict, returns: dict):
        self.scenario_probabilities = probabilities
        self.asset_returns = returns

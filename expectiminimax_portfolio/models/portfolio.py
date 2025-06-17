
from pydantic import BaseModel
from typing import Dict

class OptimizationResult(BaseModel):
    """Results from portfolio optimization"""

    risk_profile: str
    risk_aversion: float
    optimal_weights: Dict[str, float]
    expectiminimax_value: float
    expected_return: float
    expected_volatility: float
    optimization_success: bool

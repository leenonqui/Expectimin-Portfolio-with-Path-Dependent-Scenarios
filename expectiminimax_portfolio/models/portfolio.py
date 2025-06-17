from pydantic import BaseModel, validator
from typing import Dict, Optional

class OptimizationResult(BaseModel):
    """Enhanced results from portfolio optimization with liquidity preference"""

    risk_profile: str
    risk_aversion: float
    min_cash_pct: Optional[float] = 0.0  # New: minimum cash allocation
    optimal_weights: Dict[str, float]
    expectiminimax_value: float
    expected_return: float
    expected_volatility: float
    optimization_success: bool

    @validator('min_cash_pct')
    def validate_cash_constraint(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("Minimum cash percentage must be between 0.0 and 1.0")
        return v

    @property
    def liquidity_preference_met(self) -> bool:
        """Check if minimum cash constraint is satisfied"""
        if self.min_cash_pct is None or self.min_cash_pct == 0.0:
            return True

        cash_weight = self.optimal_weights.get('Cash', 0.0)
        return cash_weight >= self.min_cash_pct - 1e-6  # Allow small numerical tolerance

    @property
    def excess_cash_allocation(self) -> float:
        """Amount of cash allocated above the minimum requirement"""
        if self.min_cash_pct is None:
            return self.optimal_weights.get('Cash', 0.0)

        cash_weight = self.optimal_weights.get('Cash', 0.0)
        return max(0.0, cash_weight - self.min_cash_pct)

    @property
    def effective_risky_allocation(self) -> float:
        """Total allocation to risky assets (stocks + bonds)"""
        stocks = self.optimal_weights.get('Stocks', 0.0)
        bonds = self.optimal_weights.get('Bonds', 0.0)
        return stocks + bonds

    @property
    def profile_description(self) -> str:
        """Descriptive text for the risk-liquidity profile"""

        # Risk level description
        if self.risk_aversion == 0.0:
            risk_desc = "Risk Neutral"
        elif self.risk_aversion <= 0.5:
            risk_desc = "Low Risk Aversion"
        elif self.risk_aversion <= 1.0:
            risk_desc = "Moderate Risk Aversion"
        elif self.risk_aversion <= 2.0:
            risk_desc = "High Risk Aversion"
        else:
            risk_desc = "Very High Risk Aversion"

        # Liquidity preference description
        if self.min_cash_pct is None or self.min_cash_pct == 0.0:
            liquidity_desc = "No Liquidity Constraint"
        elif self.min_cash_pct <= 0.02:
            liquidity_desc = "Minimal Liquidity Preference"
        elif self.min_cash_pct <= 0.05:
            liquidity_desc = "Low Liquidity Preference"
        elif self.min_cash_pct <= 0.08:
            liquidity_desc = "Moderate Liquidity Preference"
        else:
            liquidity_desc = "High Liquidity Preference"

        return f"{risk_desc} + {liquidity_desc}"

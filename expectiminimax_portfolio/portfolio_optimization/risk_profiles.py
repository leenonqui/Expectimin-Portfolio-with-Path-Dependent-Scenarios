from typing import List, Dict
from ..config import RISK_AVERSION_PROFILES

class RiskProfileManager:
    """Manage risk aversion profiles for portfolio optimization"""

    @staticmethod
    def get_all_profiles() -> List[Dict]:
        """Get all predefined risk aversion profiles"""
        return RISK_AVERSION_PROFILES.copy()

    @staticmethod
    def get_profile_by_name(name: str) -> Dict:
        """Get specific risk profile by name"""
        for profile in RISK_AVERSION_PROFILES:
            if profile["name"] == name:
                return profile.copy()
        raise ValueError(f"Risk profile '{name}' not found")

    @staticmethod
    def create_custom_profile(name: str, risk_aversion: float) -> Dict:
        """Create custom risk aversion profile"""
        return {"name": name, "risk_aversion": risk_aversion}

    @staticmethod
    def get_risk_aversion_description(lambda_value: float) -> str:
        """Get description of risk aversion level"""

        descriptions = {
            0.0: "Pure expected return maximization (no risk penalty)",
            0.5: "Slight risk adjustment, still growth-oriented",
            1.0: "Balanced risk-return trade-off",
            2.0: "Conservative bias, stability over growth",
            5.0: "Very conservative, strong volatility avoidance"
        }

        # Find closest match
        closest_lambda = min(descriptions.keys(), key=lambda x: abs(x - lambda_value))

        if abs(closest_lambda - lambda_value) < 0.1:
            return descriptions[closest_lambda]
        else:
            if lambda_value == 0.0:
                return descriptions[0.0]
            elif lambda_value < 0.5:
                return "Very low risk aversion"
            elif lambda_value < 1.0:
                return "Low to moderate risk aversion"
            elif lambda_value < 2.0:
                return "Moderate to high risk aversion"
            elif lambda_value < 5.0:
                return "High risk aversion"
            else:
                return "Extremely high risk aversion"

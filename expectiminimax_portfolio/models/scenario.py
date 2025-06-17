
from pydantic import BaseModel, validator
from typing import List, Dict, Optional
import numpy as np

class EconomicScenario(BaseModel):
    """Economic scenario with GDP growth and inflation paths"""

    name: str
    gdp_growth: List[float]
    inflation: List[float]
    probability: Optional[float] = None

    @validator('gdp_growth', 'inflation')
    def validate_three_years(cls, v):
        if len(v) != 3:
            raise ValueError("Scenarios must have exactly 3 years of data")
        return v

    @property
    def path_vector(self) -> np.ndarray:
        """Create flattened path vector [GDP1, GDP2, GDP3, INF1, INF2, INF3]"""
        return np.array(self.gdp_growth + self.inflation)

class AssetReturns(BaseModel):
    """Asset returns for a specific scenario"""

    scenario_name: str
    cash: List[float]
    stocks: List[float]
    bonds: List[float]

    @validator('cash', 'stocks', 'bonds')
    def validate_three_years(cls, v):
        if len(v) != 3:
            raise ValueError("Asset returns must have exactly 3 years of data")
        return v

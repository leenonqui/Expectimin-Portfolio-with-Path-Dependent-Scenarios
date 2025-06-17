import numpy as np
from typing import Tuple, List
from ..utils.math_utils import similarity, informativeness
from ..config import PSR_TOP_PERCENTILE

class PartialSampleRegression:
    """
    Partial Sample Regression implementation following GIC methodology

    Implements equations (6) and (7) from Kritzman et al. (2021):
    - Relevance-based observation weighting
    - Subset selection of most relevant historical periods
    """

    def __init__(self, top_percentile: float = PSR_TOP_PERCENTILE):
        self.top_percentile = top_percentile

    def forecast(self,
                 X_historical: np.ndarray,
                 Y_historical: np.ndarray,
                 prospective_X: np.ndarray,
                 inv_cov_matrix: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Generate forecast using partial sample regression

        Args:
            X_historical: Historical macro path vectors (N x 6)
            Y_historical: Historical asset path vectors (N x 9)
            prospective_X: Prospective macro scenario path (6,)
            inv_cov_matrix: Inverse covariance matrix for similarity calculation

        Returns:
            forecast: Forecasted asset path vector
            top_indices: Indices of most relevant historical observations
        """

        if X_historical.shape[0] != Y_historical.shape[0]:
            raise ValueError("X and Y must have same number of observations")

        n_obs = X_historical.shape[0]
        n_top = max(1, int(n_obs * self.top_percentile))

        # Calculate means
        X_mean = np.mean(X_historical, axis=0)
        Y_mean = np.mean(Y_historical, axis=0)

        # Calculate relevance scores for all observations
        relevance_scores = self._calculate_relevance_scores(
            X_historical, prospective_X, X_mean, inv_cov_matrix
        )

        # Select most relevant observations
        top_indices = np.argsort(relevance_scores)[-n_top:]

        # Apply partial sample regression formula
        forecast = self._apply_psr_formula(
            X_historical[top_indices],
            Y_historical[top_indices],
            prospective_X,
            X_mean,
            Y_mean,
            inv_cov_matrix
        )

        return forecast, top_indices.tolist()

    def _calculate_relevance_scores(self,
                                   X_historical: np.ndarray,
                                   prospective_X: np.ndarray,
                                   X_mean: np.ndarray,
                                   inv_cov_matrix: np.ndarray) -> np.ndarray:
        """Calculate relevance score for each historical observation"""

        relevance_scores = []

        for i in range(len(X_historical)):
            # Similarity to prospective scenario (GIC Equation 3)
            sim = similarity(X_historical[i], prospective_X, inv_cov_matrix)

            # Informativeness - distance from historical mean (GIC Equation 4)
            info = informativeness(X_historical[i], X_mean, inv_cov_matrix)

            # Total relevance (GIC Equation 5)
            relevance = sim + info
            relevance_scores.append(relevance)

        return np.array(relevance_scores)

    def _apply_psr_formula(self,
                          X_top: np.ndarray,
                          Y_top: np.ndarray,
                          prospective_X: np.ndarray,
                          X_mean: np.ndarray,
                          Y_mean: np.ndarray,
                          inv_cov_matrix: np.ndarray) -> np.ndarray:
        """Apply PSR formula from GIC equations (6) and (7)"""

        n = len(X_top)
        weighted_sum = np.zeros(Y_top.shape[1])

        # Calculate weighted sum using relevance weights
        for i in range(n):
            # Recalculate relevance components for selected observations
            sim_i = similarity(X_top[i], prospective_X, inv_cov_matrix)
            info_i = informativeness(X_top[i], X_mean, inv_cov_matrix)

            # Weight calculation following GIC methodology
            relevance = (sim_i + info_i)

            # Weighted contribution to forecast
            weighted_sum += relevance * (Y_top[i] - Y_mean)

        # Final forecast calculation
        forecast = Y_mean + (1/2*n)*weighted_sum
        return forecast


import numpy as np
from scipy.spatial.distance import mahalanobis

def similarity(x_i: np.ndarray, x_t: np.ndarray, inv_cov: np.ndarray) -> float:
    """
    Calculate multivariate similarity (negative squared Mahalanobis distance)

    From GIC paper Equation 3: Similarity(xi, xt) = -(xi - xt)'Ω^-1(xi - xt)
    """
    return -(mahalanobis(x_i, x_t, inv_cov) ** 2)

def informativeness(x_i: np.ndarray, x_mean: np.ndarray, inv_cov: np.ndarray) -> float:
    """
    Calculate informativeness (squared Mahalanobis distance from mean)

    From GIC paper Equation 4: Informativeness(xi) = (xi - x̄)'Ω^-1(xi - x̄)
    """
    return (mahalanobis(x_i, x_mean, inv_cov) ** 2)

def create_path_vector(*time_series) -> np.ndarray:
    """Create flattened path vector from multiple time series"""
    path_vector = []
    for series in time_series:
        path_vector.extend(series)
    return np.array(path_vector)

def mahalanobis_distance_squared(x: np.ndarray, y: np.ndarray, inv_cov: np.ndarray) -> float:
    """
    Calculate squared Mahalanobis distance as used in GIC paper Equation 1

    d = (x - γ)'Ω^-1(x - γ)
    """
    return mahalanobis(x, y, inv_cov) ** 2

def scenario_likelihood(mahalanobis_dist_squared: float) -> float:
    """
    Calculate scenario likelihood using GIC paper Equation 2

    Likelihood ∝ e^(-d/2)
    """
    return np.exp(-mahalanobis_dist_squared / 2.0)

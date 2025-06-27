"""
Utility Functions
Common mathematical and data processing functions used across modules
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from constants import RISK_FREE_RATE, INFLATION


def create_path_vector(*args: List[float]) -> np.ndarray:
    """
    CORRECTED: Create path vector x_s from economic variables
    Following thesis notation: x_s = [GDP₁, GDP₂, GDP₃, INF₁, INF₂, INF₃]

    Args:
        *args: Variable number of lists (gdp_growth, inflation, etc.)

    Returns:
        Combined path vector of size (sum of list lengths,)
    """
    # Flatten all input lists and concatenate
    result = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            result.extend(arg)
        elif isinstance(arg, np.ndarray):
            result.extend(arg.flatten())
        else:
            result.append(arg)

    return np.array(result, dtype=float)


def calculate_mahalanobis_distance(x: np.ndarray, y: np.ndarray, inv_cov: np.ndarray) -> float:
    """
    CORRECTED: Calculate Mahalanobis distance between two vectors

    Args:
        x: First vector (e.g., scenario path)
        y: Second vector (e.g., anchor path)
        inv_cov: Inverse covariance matrix Ω^(-1)

    Returns:
        Squared Mahalanobis distance (non-negative)
    """
    try:
        # Ensure vectors are 1D
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()

        # Check dimensions
        if len(x) != len(y):
            raise ValueError(f"Vector dimensions don't match: {len(x)} vs {len(y)}")

        if inv_cov.shape[0] != len(x) or inv_cov.shape[1] != len(x):
            raise ValueError(f"Inverse covariance matrix shape {inv_cov.shape} doesn't match vector length {len(x)}")

        # Calculate difference
        diff = x - y

        # Mahalanobis distance squared: diff^T * inv_cov * diff
        distance_squared = diff.T @ inv_cov @ diff

        # Ensure non-negative (handle numerical errors)
        return max(0.0, float(distance_squared))

    except Exception as e:
        print(f"Warning: Mahalanobis distance calculation failed ({e}), using Euclidean")
        # Fallback to Euclidean distance
        return float(np.sum((x - y) ** 2))


def calculate_similarity(x_i: np.ndarray, x_t: np.ndarray, omega_inv: np.ndarray) -> float:
    """
    Calculate multivariate similarity between two paths
    Equation 3 from thesis: Similarity(x_i, x_t) = -(x_i - x_t)' Ω^(-1) (x_i - x_t)
    """
    return -calculate_mahalanobis_distance(x_i, x_t, omega_inv)


def calculate_informativeness(x_i: np.ndarray, x_bar: np.ndarray, omega_inv: np.ndarray) -> float:
    """
    Calculate informativeness of a historical path
    Equation 4 from thesis: Informativeness(x_i) = (x_i - x̄)' Ω^(-1) (x_i - x̄)
    """
    return calculate_mahalanobis_distance(x_i, x_bar, omega_inv)


def calculate_relevance(x_i: np.ndarray, x_t: np.ndarray, x_bar: np.ndarray, omega_inv: np.ndarray) -> float:
    """
    Calculate total relevance of a historical observation
    Equation 5 from thesis: Relevance = Similarity + Informativeness
    """
    similarity = calculate_similarity(x_i, x_t, omega_inv)
    informativeness = calculate_informativeness(x_i, x_bar, omega_inv)
    return similarity + informativeness


def scenario_likelihood(mahalanobis_dist_squared: float) -> float:
    """
    Calculate scenario likelihood from squared Mahalanobis distance
    Equation 2 from thesis: L_s ∝ exp(-d_s/2)
    """
    # Handle numerical overflow/underflow
    if mahalanobis_dist_squared > 700:  # exp(-350) is very small
        return 1e-300
    elif mahalanobis_dist_squared < 0:
        mahalanobis_dist_squared = 0

    return np.exp(-mahalanobis_dist_squared / 2)


def safe_matrix_inverse(matrix: np.ndarray, regularization: float = 1e-8) -> np.ndarray:
    """
    IMPROVED: Safely compute matrix inverse with better regularization

    Args:
        matrix: Square matrix to invert
        regularization: Small value to add to diagonal if needed

    Returns:
        Inverse matrix
    """
    try:
        # Check if matrix is square
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Matrix must be square, got shape {matrix.shape}")

        # Check condition number
        cond_num = np.linalg.cond(matrix)

        if cond_num > 1e12:  # Matrix is near-singular
            print(f"Warning: Matrix condition number is high ({cond_num:.2e}), adding regularization")
            n = matrix.shape[0]
            regularized = matrix + regularization * np.eye(n)
            return np.linalg.inv(regularized)
        else:
            return np.linalg.inv(matrix)

    except np.linalg.LinAlgError as e:
        print(f"Warning: Matrix inversion failed ({e}), using pseudo-inverse")
        return np.linalg.pinv(matrix)
    except Exception as e:
        print(f"Warning: Unexpected error in matrix inversion ({e}), using pseudo-inverse")
        return np.linalg.pinv(matrix)


def normalize_probabilities(likelihoods: dict) -> dict:
    """
    IMPROVED: Normalize likelihoods to sum to 1 with better error handling
    """
    if not likelihoods:
        raise ValueError("Empty likelihoods dictionary")

    # Remove any non-finite values
    clean_likelihoods = {}
    for k, v in likelihoods.items():
        if np.isfinite(v) and v >= 0:
            clean_likelihoods[k] = v
        else:
            print(f"Warning: Invalid likelihood for {k}: {v}, setting to small value")
            clean_likelihoods[k] = 1e-10

    total = sum(clean_likelihoods.values())

    if total <= 1e-15:
        print("Warning: All likelihoods are effectively zero, using uniform distribution")
        n = len(clean_likelihoods)
        return {k: 1.0/n for k in clean_likelihoods.keys()}

    return {k: v/total for k, v in clean_likelihoods.items()}


def calculate_real_return(nominal_return: float, inflation: float) -> float:
    """
    IMPROVED: Calculate real return from nominal return and inflation
    """
    try:
        # Convert percentages to decimals if needed
        if abs(nominal_return) > 1:
            nominal_return = nominal_return / 100
        if abs(inflation) > 1:
            inflation = inflation / 100

        # Fisher equation: (1 + real) = (1 + nominal) / (1 + inflation)
        real_return = ((1 + nominal_return) / (1 + inflation) - 1) * 100

        return real_return

    except Exception as e:
        print(f"Warning: Real return calculation failed ({e}), using approximation")
        # Approximation: real ≈ nominal - inflation
        return nominal_return - inflation


def simple_to_log_return(simple_return: float) -> float:
    """Convert simple return to log return"""
    if abs(simple_return) > 1:
        simple_return = simple_return / 100

    return np.log(1 + simple_return)


def log_to_simple_return(log_return: float) -> float:
    """Convert log return to simple return"""
    return np.exp(log_return) - 1


def get_risk_free_rate() -> float:
    """Calculate real risk-free rate"""
    real_rate = calculate_real_return(RISK_FREE_RATE, INFLATION)
    return real_rate


def calculate_cumulative_return(annual_returns: List[float], as_log: bool = False) -> float:
    """
    Calculate cumulative return over multiple periods
    """
    if not annual_returns:
        return 0.0

    try:
        if as_log:
            # Log returns are additive
            cumulative_log = sum(simple_to_log_return(r) for r in annual_returns)
            return (np.exp(cumulative_log) - 1) * 100
        else:
            # Simple returns compound multiplicatively
            cumulative = 1.0
            for r in annual_returns:
                cumulative *= (1 + r/100)
            return (cumulative - 1) * 100
    except Exception as e:
        print(f"Warning: Cumulative return calculation failed ({e})")
        return 0.0


def prepare_historical_paths(data: pd.DataFrame,
                           horizon: int,
                           end_year: int) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    IMPROVED: Prepare historical T-year paths for analysis
    """
    paths = []
    path_info = []

    for start_year in data.index:
        end = start_year + horizon - 1

        # Stop if we exceed the end year
        if end > end_year or end > data.index.max():
            break

        try:
            # Extract path data
            window = data.loc[start_year:end]

            if len(window) == horizon and not window.isnull().any().any():
                gdp_path = window['gdp_growth'].values
                inf_path = window['inflation'].values
                path_vector = create_path_vector(gdp_path.tolist(), inf_path.tolist())

                paths.append(path_vector)
                path_info.append({
                    'start_year': start_year,
                    'end_year': end,
                    'path_id': len(paths) - 1
                })

        except (KeyError, IndexError) as e:
            continue  # Skip invalid windows

    return np.array(paths), pd.DataFrame(path_info)

def calculate_path_differences(paths: np.ndarray, lag: int = 1) -> np.ndarray:
    """
    Calculate differences between consecutive paths
    """
    if lag >= len(paths):
        raise ValueError(f"Lag {lag} is too large for {len(paths)} paths")

    return paths[lag:] - paths[:-lag]

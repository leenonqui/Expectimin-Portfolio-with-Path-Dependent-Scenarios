"""
Utility Functions
Common mathematical and data processing functions used across modules
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from typing import List, Tuple, Optional
from constants import RISK_FREE_RATE, INFLATION


def create_path_vector(*args: List[float]) -> np.ndarray:
    """
    Create path vector x_s from economic variables
    Following thesis notation: x_s = [GDP₁, GDP₂, GDP₃, INF₁, INF₂, INF₃]

    Args:
        gdp_growth: T-year GDP growth path
        inflation: T-year inflation path

    Returns:
        Combined path vector of size (2T × 1)
    """

    return np.concatenate([*args], axis=None)

def calculate_mahalanobis_distance(x: np.ndarray, y: np.ndarray, inv_cov: np.ndarray) -> float:
    """
    Calculate Mahalanobis distance between two vectors
    Equation 1 from thesis: d = (x - γ)' Ω^(-1) (x - γ)

    Args:
        x: First vector (e.g., scenario path)
        y: Second vector (e.g., anchor path)
        inv_cov: Inverse covariance matrix Ω^(-1)

    Returns:
        Mahalanobis distance (non-negative)
    """
    return mahalanobis(x, y, inv_cov) ** 2

def calculate_similarity(x_i: np.ndarray, x_t: np.ndarray, omega_inv: np.ndarray) -> float:
    """
    Calculate multivariate similarity between two paths
    Equation 3 from thesis: Similarity(x_i, x_t) = -(x_i - x_t)' Ω^(-1) (x_i - x_t)

    Args:
        x_i: Historical path vector
        x_t: Target path vector (scenario)
        omega_inv: Inverse covariance matrix Ω^(-1)

    Returns:
        Similarity score (higher is more similar)
    """
    # Use negative of squared Mahalanobis distance
    return -(mahalanobis(x_i, x_t, omega_inv) ** 2)

def calculate_informativeness(x_i: np.ndarray, x_bar: np.ndarray, omega_inv: np.ndarray) -> float:
    """
    Calculate informativeness of a historical path
    Equation 4 from thesis: Informativeness(x_i) = (x_i - x̄)' Ω^(-1) (x_i - x̄)

    Args:
        x_i: Historical path vector
        x_bar: Mean of historical paths
        omega_inv: Inverse covariance matrix Ω^(-1)

    Returns:
        Informativeness score (higher is more informative)
    """
    # Use squared Mahalanobis distance from mean
    return mahalanobis(x_i, x_bar, omega_inv) ** 2

def calculate_relevance(x_i: np.ndarray, x_t: np.ndarray, x_bar: np.ndarray, omega_inv: np.ndarray) -> float:
    """
    Calculate total relevance of a historical observation
    Equation 5 from thesis: Relevance = Similarity + Informativeness

    Args:
        x_i: Historical path vector
        x_t: Target path vector (scenario)
        x_bar: Mean of historical paths
        omega_inv: Inverse covariance matrix Ω^(-1)

    Returns:
        Total relevance score
    """
    similarity = calculate_similarity(x_i, x_t, omega_inv)
    informativeness = calculate_informativeness(x_i, x_bar, omega_inv)
    return similarity + informativeness

def scenario_likelihood(mahalanobis_dist_squared: float) -> float:
    """
    Calculate scenario likelihood from squared Mahalanobis distance
    Equation 2 from thesis: L_s ∝ exp(-d_s/2)

    Args:
        mahalanobis_dist_squared: Squared Mahalanobis distance d_s

    Returns:
        Likelihood (unnormalized probability)
    """
    return np.exp(-mahalanobis_dist_squared / 2)

def simple_to_log_return(simple_return: float) -> float:
    """
    Convert simple return to log return
    r = ln(1 + R)

    Args:
        simple_return: Simple return R (can be percentage or decimal)

    Returns:
        Log return r
    """
    # Handle percentage input (convert to decimal)
    if abs(simple_return) > 1:
        simple_return = simple_return / 100

    return np.log(1 + simple_return)

def log_to_simple_return(log_return: float) -> float:
    """
    Convert log return to simple return
    R = exp(r) - 1

    Args:
        log_return: Log return r

    Returns:
        Simple return R (as decimal)
    """
    return np.exp(log_return) - 1

def calculate_real_return(nominal_return: float, inflation: float) -> float:
    """
    Calculate real return from nominal return and inflation
    Real = (1 + Nominal) / (1 + Inflation) - 1

    Args:
        nominal_return: Nominal return (percentage)
        inflation: Inflation rate (percentage)

    Returns:
        Real return (percentage)
    """
    return ((1 + nominal_return) / (1 + inflation/100) - 1) * 100

def get_risk_free_rate() -> float:
    """
    Calculate real risk-free rate for a given year
    Uses bond rate minus inflation for the prediction start year

    Args:
        data: Historical data with bond rates and inflation
        prediction_year: Year to get risk-free rate for

    Returns:
        Real risk-free rate (annual percentage)
    """

    # Calculate real rate
    real_rate = calculate_real_return(RISK_FREE_RATE, INFLATION)

    return real_rate

def calculate_cumulative_return(annual_returns: List[float], as_log: bool = False) -> float:
    """
    Calculate cumulative return over multiple periods

    Args:
        annual_returns: List of annual returns (percentages)
        as_log: If True, returns are log returns; if False, simple returns

    Returns:
        Cumulative return (percentage)
    """
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

def prepare_historical_paths(data: pd.DataFrame,
                           horizon: int,
                           end_year: int) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Prepare historical T-year paths for analysis

    Args:
        data: Historical data with economic variables
        horizon: Investment horizon T
        end_year: Last year to include in historical data

    Returns:
        Tuple of (path_matrix, path_info_df)
        - path_matrix: N × (2T) array of historical paths
        - path_info_df: DataFrame with path metadata
    """
    paths = []
    path_info = []

    for start_year in data.index:
        end = start_year + horizon - 1

        # Stop if we exceed the end year
        if end > end_year or end > data.index.max():
            break

        # Extract path data
        window = data.loc[start_year:end]

        if len(window) == horizon:
            gdp_path = window['gdp_growth'].values
            inf_path = window['inflation'].values
            path_vector = create_path_vector(gdp_path.tolist(), inf_path.tolist())

            paths.append(path_vector)
            path_info.append({
                'start_year': start_year,
                'end_year': end,
                'path_id': len(paths) - 1
            })

    return np.array(paths), pd.DataFrame(path_info)

def calculate_path_differences(paths: np.ndarray, lag: int = 1) -> np.ndarray:
    """
    Calculate differences between consecutive paths
    Used for covariance matrix estimation (Section 3.1.2)

    Args:
        paths: N × D array of path vectors
        lag: Lag for differencing (default 1 for consecutive)

    Returns:
        (N-lag) × D array of path differences
    """
    if lag >= len(paths):
        raise ValueError(f"Lag {lag} is too large for {len(paths)} paths")

    return paths[lag:] - paths[:-lag]

def safe_matrix_inverse(matrix: np.ndarray, regularization: float = 1e-6) -> np.ndarray:
    """
    Safely compute matrix inverse with regularization if needed

    Args:
        matrix: Square matrix to invert
        regularization: Small value to add to diagonal if singular

    Returns:
        Inverse matrix
    """
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        # Add small regularization to diagonal
        n = matrix.shape[0]
        regularized = matrix + regularization * np.eye(n)
        try:
            return np.linalg.inv(regularized)
        except np.linalg.LinAlgError:
            # Fall back to pseudo-inverse
            return np.linalg.pinv(matrix)

def normalize_probabilities(likelihoods: dict) -> dict:
    """
    Normalize likelihoods to sum to 1

    Args:
        likelihoods: Dictionary of unnormalized likelihoods

    Returns:
        Dictionary of normalized probabilities
    """
    total = sum(likelihoods.values())
    if total == 0:
        raise ValueError("All likelihoods are zero")

    return {k: v/total for k, v in likelihoods.items()}

def portfolio_performance_summary(scenario_returns: dict,
                                probabilities: dict,
                                confidence_level: float = 0.95) -> dict:
    """
    Calculate comprehensive portfolio performance metrics

    Args:
        scenario_returns: Dictionary of scenario returns
        probabilities: Dictionary of scenario probabilities
        confidence_level: Confidence level for VaR/CVaR

    Returns:
        Dictionary with performance metrics
    """
    # Convert to arrays
    returns = np.array([scenario_returns[s] for s in probabilities.keys()])
    probs = np.array([probabilities[s] for s in probabilities.keys()])

    # Expected return
    expected_return = np.sum(probs * returns)

    # Variance and standard deviation
    variance = np.sum(probs * (returns - expected_return)**2)
    std_dev = np.sqrt(variance)

    # Sort for VaR/CVaR
    sorted_idx = np.argsort(returns)
    sorted_returns = returns[sorted_idx]
    sorted_probs = probs[sorted_idx]

    # VaR calculation
    alpha = 1 - confidence_level
    cumul_prob = 0
    var = sorted_returns[0]

    for i, prob in enumerate(sorted_probs):
        cumul_prob += prob
        if cumul_prob >= alpha:
            var = sorted_returns[i]
            break

    # CVaR calculation
    cvar = 0
    tail_prob = 0

    for i, (ret, prob) in enumerate(zip(sorted_returns, sorted_probs)):
        if ret <= var:
            cvar += ret * prob
            tail_prob += prob

    if tail_prob > 0:
        cvar = cvar / tail_prob

    return {
        'expected_return': expected_return,
        'std_dev': std_dev,
        'var': var,
        'cvar': cvar,
        'worst_case': np.min(returns),
        'best_case': np.max(returns),
        'sharpe_ratio': expected_return / std_dev if std_dev > 0 else 0
    }

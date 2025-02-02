import numpy as np


def payoff_function(S_range: np.ndarray, K: float, option_type: str) -> np.ndarray:
    
    """
    Calculate the payoff for a given range of stock prices, strike price, and option type.
    
    Parameters:
    S_range (array-like): Range of stock prices at expiration
    K (float): Strike price
    option_type (str): Type of the option ('call' or 'put')
    
    Returns:
    array-like: Payoff values
    """
    
    return np.maximum(S_range - K, 0) if option_type == 'call' else np.maximum(K - S_range, 0)


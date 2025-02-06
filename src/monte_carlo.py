import numpy as np
from sklearn.linear_model import LinearRegression

def monte_carlo_european(S, K, T, r, sigma, num_simulations, num_steps, type_option:str):
    """
    Monte Carlo simulation for European option pricing using Geometric Brownian Motion.
    
    Args:
        S (float) : Initial stock price
        K (float) : Strike price
        T (float) : Time to maturity (in years)
        r (float) : Risk-free interest rate
        sigma (float) : Volatility of the underlying asset
        num_simulations (int) : Number of Monte Carlo simulations
        num_steps (int) : Number of time steps per simulation
        option_type (str) : 'call' or 'put'
    
    Returns:
        float : Option prices
    """
    # Time step
    dt = T / num_steps  
    # Discount factor
    discount_factor = np.exp(-r * T)  

    # Simulate price paths using Geometric Brownian Motion
    stock_prices = np.full((num_simulations, num_steps + 1), S)  

    for t in range(1, num_steps + 1):
        # Generate random standard normal values
        Z = np.random.standard_normal(num_simulations)
        stock_prices[:, t] = stock_prices[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    # Get final stock prices at expiration
    S_T = stock_prices[:, -1]
    
    # Compute payoffs
    if type_option == 'call':
        payoffs = np.maximum(S_T - K, 0)
    elif type_option == 'put':
        payoffs = np.maximum(K - S_T, 0)

    # Compute present value of expected payoff
    option_price = discount_factor * np.mean(payoffs)

    return option_price

def monte_carlo_lsm_american(S, K, T, r, sigma, num_simulations, num_steps, type_option:str):
    """ Least Squares Monte Carlo simulation for American option pricing using Geometric Brownian Motion.

    Args:
        S (float): Initial stock price
        K (float): strike price
        T (float): time to maturity (in years)
        r (float): risk-free interest rate
        sigma (float): volatility
        num_simulations (int): number of Monte Carlo simulations
        num_steps (int): number of time steps per simulation
        type_option (str): 'call' or 'put'
    
    Returns:
        float: Option prices
    """
    
    
    
    
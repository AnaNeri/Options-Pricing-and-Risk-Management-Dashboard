import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import logging 

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
        
        # drift - deterministic - expected growth
        drift = (r - 0.5 * sigma ** 2) * dt
        
        # stochastic component - random shock
        random_shock = sigma * np.sqrt(dt) * Z
        
        # Update stock prices
        stock_prices[:, t] = stock_prices[:, t - 1] * np.exp(drift + random_shock)

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

def regression(Xs, Y, stock_paths, in_the_money):
    """Least Squares Regression for American Option Pricing using Longstaff-Schwartz method.

    Args:
        Xs (float): Xs is the matrix of X (stock price at time t) and X^2
        Y (float): Y is the cash flow at time t+1
        stock_paths (): 
        in_the_money (): 

    Returns:
        np.darray : Continuation values
    """
    model_sklearn = LinearRegression()
    model = model_sklearn.fit(Xs, Y)
    conditional_exp = model.predict(Xs)
    continuations = np.zeros_like(stock_paths[1,:])
    continuations[in_the_money] = conditional_exp
    
    return continuations


def simulate_stock_path(S, T, r, sigma, num_paths, dt):
    """Monte Carlo Simulated Stock Price Paths
    
    Args:
        S (float): Initial stock price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate
        sigma (float): Volatility of the underlying asset
        num_paths (int): Number of Monte Carlo simulations
        num_steps (int): Number of time steps per simulation
    
    Returns:
        np.array: Simulated stock price paths
    """
    # Generate Brownian motion (random component)
    Z = np.random.standard_normal((num_paths, T))  # Normal random numbers
    stock_paths = np.zeros((num_paths, T + 1))  # Store stock prices
    stock_paths[:, 0] = S  # Set initial price

    # Simulate GBM paths
    for t in range(1, T + 1):
        stock_paths[:, t] = stock_paths[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])

    # Reverse the order to match Longstaff-Schwartz format
    stock_paths = np.flip(stock_paths.T, axis=0)  # Flip time axis
    
    return stock_paths

def plot_stock_path(stock_paths, K):
    plt.figure(figsize=(8, 5))
    plt.plot(stock_paths.T, linestyle="--", marker="o", alpha=0.7)
    plt.axhline(y=K, color="r", linestyle="dashed", label="Strike Price")
    plt.xlabel("Time Step")
    plt.ylabel("Stock Price")
    plt.title("Monte Carlo Simulated Stock Price Paths")
    plt.legend()
    plt.show()
    

def cash_flow(stock_paths, K, type_option:str):
    """
    Compute cash flow for American option pricing using Longstaff-Schwartz method.
    
    Args:
        stock_paths (np.array): Simulated stock price paths
        K (float): Strike price
        type_option (str): 'call' or 'put'
    
    Returns:
        np.array: Cash flow matrix
    """
    
    # Initialize cash flow matrix
    cash_flows = np.zeros_like(stock_paths)
    
    # Compute payoffs
    if type_option == 'call':
        for i in range(0, cash_flows.shape[0]):
            cash_flows[i] = np.maximum(stock_paths[i] - K, 0)
    elif type_option == 'put':
        for i in range(0,cash_flows.shape[0]):
            cash_flows[i] = np.maximum(K - stock_paths[i], 0)
    
    discount_cash_flow = np.zeros_like(cash_flows)
    
    return cash_flows, discount_cash_flow

def monte_carlo_lsm_american(S, K, r, sigma, num_simulations, dt, type_option:str, T=None):
    """ Least Squares Monte Carlo simulation for American option pricing using Geometric Brownian Motion.
    Reference: Longstaff, Francis A., and Eduardo S. Schwartz. "Valuing American options by simulation: A simple least-squares approach." The review of financial studies 14.1 (2001):

    Args:
        S (float): Initial stock price
        K (float): strike price
        T (float): time to maturity (in years)
        r (float): risk-free interest rate
        sigma (float): volatility
        num_simulations (int): number of Monte Carlo simulations
        dt (int): time step
        type_option (str): 'call' or 'put'
    
    Returns:
        float: Option prices
    """
    
    paths = simulate_stock_path(S, T, r, sigma, num_simulations, dt)
    
    cash_flows, d_cash_flows = cash_flow(paths, K, type_option)
    
    if not T:
        T = paths.shape[1] - 1
    
    for t in range(1,T):
        in_the_money =paths[1,:] < K

        X = (paths[t,in_the_money])
        X2 = X*X
        Xs = np.column_stack([X,X2])
        Y = cash_flows[t-1,in_the_money]  * np.exp(-r)
        
        continuation_values = regression(Xs,Y,paths,in_the_money)
        
        cash_flows[t,:] = np.where(continuation_values> cash_flows[t,:], 0, cash_flows[t,:])
        
        exercised_early = continuation_values < cash_flows[t, :]
        cash_flows[0:t, :][:, exercised_early] = 0
        d_cash_flows[t-1,:] = cash_flows[t-1,:]* np.exp(-r * 3)
    
    d_cash_flows[T-1:] = cash_flows[T-1,:]* np.exp(-r * 1)
    
    # Return final option price
    final_cfs = np.zeros((d_cash_flows.shape[1], 1), dtype=float)
    for i,_ in enumerate(final_cfs):
        final_cfs[i] = sum(d_cash_flows[:,i])
    option_price = np.mean(final_cfs)
    return option_price
    
    
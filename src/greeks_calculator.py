from math import log, sqrt, exp
from scipy.stats import norm

# cumulative distribution function - used when calculating probabilities
N = norm.cdf
# probability density function - used when measuring rate of change
N_prime = norm.pdf 

def delta(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the Delta of an option.

    Args:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate
        sigma (float): Volatility of the underlying asset
        option_type (str): 'call' or 'put'

    Returns:
        float: Delta of the option
    """

    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))

    if option_type == 'call':
        return N(d1)
    elif option_type == 'put':
        return N(d1) - 1
    else:
        raise ValueError("option_type must be 'call' or 'put'")  


def gamma(S, K, T, r, sigma):
    """
    Calculate the Gamma of an option.

    Args:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate
        sigma (float): Volatility of the underlying asset

    Returns:
        float: Gamma of the option
    """

    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    
    return N(d1) / (S * sigma * sqrt(T))
    
def theta(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the Theta of an option.

    Args:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate
        sigma (float): Volatility of the underlying asset
        option_type (str): 'call' or 'put'

    Returns:
        float: Theta of the option
    """

    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type == 'call':
        theta = (-S * sigma * N_prime(d1) / (2 * sqrt(T)) 
                    - r * K * exp(-r * T) * N(d2)) / 365
    elif option_type == 'put':
        theta = (-S * sigma * N_prime(d1) / (2 * sqrt(T)) 
                    + r * K * exp(-r * T) * N(-d2)) / 365
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return theta
    
def vega(S, K, T, r, sigma):
    """
    Calculate the Vega of an option.

    Args:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate
        sigma (float): Volatility of the underlying asset

    Returns:
        float: Vega of the option
    """

    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    
    return S * sqrt(T) * N_prime(d1) / 100
    
    
def rho(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the Rho of an option.

    Args:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate
        sigma (float): Volatility of the underlying asset
        option_type (str): 'call' or 'put'

    Returns:
        float: Rho of the option
    """

    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type == 'call':
        rho = K * T * exp(-r * T) * N(d2) / 100
    elif option_type == 'put':
        rho = -K * T * exp(-r * T) * N(-d2) / 100
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return rho


def greeks_calculater_funtion(S, K, T, r, sigma, option_type='call'):
    """
    Calculate all the Greeks for an option.

    Args:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate
        sigma (float): Volatility of the underlying asset
        option_type (str): 'call' or 'put'

    Returns:
        dict: A dictionary containing Delta, Gamma, Theta, Vega, and Rho of the option
    """
    greeks = {
        'Delta': delta(S, K, T, r, sigma, option_type),
        'Gamma': gamma(S, K, T, r, sigma),
        'Theta': theta(S, K, T, r, sigma, option_type),
        'Vega': vega(S, K, T, r, sigma),
        'Rho': rho(S, K, T, r, sigma, option_type)
    }
    return greeks
    
    
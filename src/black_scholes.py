import numpy as np
from scipy.stats import norm

N = norm.cdf

def black_scholes_equation(S:float, K:float, T:int, r:float, sigma:float, option_type:str='c'):
    """Function to calculate the price of a European call or put option using the Black-Scholes equation.

    Args:
        S (float): Stock price
        K (float): Strike price
        T (int): Time to maturity
        r (float): Risk-free rate
        sigma (float): Volatility
        option_type (str, optional): Option type. Defaults to 'c'.

    Raises:
        ValueError: Invalid option type. Please enter either "c" for call or "p" for put option.

    Returns:
        float: Price of the option
    """
    
    d1 = ( np.log(S/K) + (r+0.5*sigma**2)*T ) / (sigma*np.sqrt(T))
    d2 = d1 -sigma*np.sqrt(T) 
    
    if option_type == 'c':
        option_price = S * N(d1) - K *np.exp(-r*T)* N(d2)
    elif option_type=='p':
        option_price = N(-d2)*K*np.exp(-r*T) - S*N(-d1)
    else:
        raise ValueError('Invalid option type. Please enter either "c" for call or "p" for put option.')
    return option_price
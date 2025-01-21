import numpy as np
from scipy.stats import norm

N = norm.cdf

def black_scholes_equation(S, K, T, r, sigma, option_type:str='c'):
    
    d1 = ( np.log(S/K) + (r+0.5*sigma**2)*T ) / (sigma*np.sqrt(T))
    d2 = d1 -sigma*np.sqrt(T) 
    
    if option_type == 'c':
        option_price = S * N(d1) - K *np.exp(-r*T)* N(d2)
    elif option_type=='p':
        option_price = N(-d2)*K*np.exp(-r*T) - S*N(-d1)
    else:
        raise ValueError('Invalid option type. Please enter either "c" for call or "p" for put option.')
    return option_price
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def visual_bs(x_values, y_values, x_label):
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label='Call Option Prices', color='blue', marker='o')
    plt.xlabel(x_label.capitalize(), fontsize=12)
    plt.ylabel('Call Option Prices', fontsize=12)
    plt.title(f'Black-Scholes Option Prices vs {x_label.capitalize()}', fontsize=14)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.7)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.show()

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



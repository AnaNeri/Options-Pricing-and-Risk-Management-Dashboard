import matplotlib.pyplot as plt
import numpy as np

def plot_payoff(S_range, K, option_type='call'):
    payoff = np.maximum(S_range - K, 0) if option_type == 'call' else np.maximum(K - S_range, 0)
    plt.plot(S_range, payoff, label=f'{option_type.capitalize()} Payoff')
    plt.xlabel('Stock Price at Expiration')
    plt.ylabel('Payoff')
    plt.legend()
    plt.grid()
    plt.title(f'{option_type.capitalize()} Option Payoff')
    plt.show()

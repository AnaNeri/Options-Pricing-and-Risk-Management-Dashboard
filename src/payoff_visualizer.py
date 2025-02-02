import matplotlib.pyplot as plt
from payoff import payoff_function 

def plot_payoff(S_range, K, option_type='call', option_price=None):
    
    payoff = payoff_function(S_range, K, option_type)
    
    if option_price:
        payoff -= option_price
    
    if option_type == 'put':
        plt.plot(S_range, payoff, label='Put Payoff', color='red')
        plt.plot(S_range, S_range - K, label='Stock Value', color='blue')
        plt.plot(S_range, payoff + S_range - K, label='Protective Put', color='green')
    elif option_type == 'call':
        plt.plot(S_range, payoff, label='Call Payoff', color='red')
        plt.plot(S_range, K - S_range, label='Stock Value', color='blue')
        plt.plot(S_range, payoff + K - S_range, label='Fiduciary Call', color='green')
    
    plt.xlabel('Stock Price at Expiration')
    plt.ylabel('Payoff')
    plt.legend()
    plt.grid()
    plt.title(f'{option_type.capitalize()} Option Payoff')
    plt.show()

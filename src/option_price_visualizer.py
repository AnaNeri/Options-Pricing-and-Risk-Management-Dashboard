import matplotlib.pyplot as plt

def visualizer_bs_call(x_values, y_values, x_label):
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label='Call Option Prices', color='blue', marker='o')
    plt.xlabel(x_label.capitalize(), fontsize=12)
    plt.ylabel('Call Option Prices', fontsize=12)
    plt.title(f'Black-Scholes Option Prices vs {x_label.capitalize()}', fontsize=14)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.7)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.show()
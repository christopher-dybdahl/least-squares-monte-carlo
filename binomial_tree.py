import numpy as np
import matplotlib.pyplot as plt

def option_binomial(S0, K, T, r, q, sigma, n, option_type='call', type='european'):
    dt = T / n
    u = np.exp((r - q) * dt + sigma * np.sqrt(dt))
    d = np.exp((r - q) * dt - sigma * np.sqrt(dt))
    p = (np.exp((r - q) * dt) - d) / (u - d)
    S = np.zeros((n + 1, n + 1))
    S[0, 0] = S0
    for i in range(1, n + 1):
        S[i, 0] = S[i - 1, 0] * u
        for j in range(1, i + 1):
            S[i, j] = S[i - 1, j - 1] * d
    if option_type == 'call':
        option = np.maximum(S - K, 0)
    elif option_type == 'put':
        option = np.maximum(K - S, 0)
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            continuation_val = np.exp(-r * dt) * (p * option[i + 1, j] + (1 - p) * option[i + 1, j + 1])
            option[i, j] = max(
                S[i, j] - K if option_type == 'call' else K - S[i, j],
                continuation_val
            ) if type == 'american' else continuation_val
    return option[0, 0]

def plot_priced_tree(stock_prices, forward_prices, option_prices, exercise_flags, scale=1, loc=0.6):
    """
    Visualizes a binomial tree for option pricing, displaying stock prices, forward prices,
    option prices, and exercise flags.
    """
    n_periods = stock_prices.shape[1]
    plt.figure(figsize=(1.2*n_periods*scale, 0.8*n_periods*scale))
    base_text_kwargs = dict(ha='center', va='center', fontsize=8)
    bbox_props = dict(fc="white", ec="white", lw=0.5)
    for t in range(n_periods):
        for i in range(t+1):
            s = stock_prices[i, t]
            f = forward_prices[i, t]
            op = option_prices[i, t]
            x = t
            y = n_periods/2 - 2 * (i - t / 2) - 3
            sf_text = f"${s:.2f}\n$F^P$={f:.2f}"
            plt.text(x, y, sf_text, bbox=bbox_props, **base_text_kwargs)
            if exercise_flags is not None and exercise_flags[i, t]:
                plt.text(x, y - loc/scale, f"${op:.2f}", fontweight='bold', bbox=bbox_props, **base_text_kwargs)
            else:
                plt.text(x, y - loc/scale, f"${op:.2f}", bbox=bbox_props, **base_text_kwargs)
            if t < n_periods - 1:
                next_y_up = y + 1
                next_y_down = y - 1
                plt.plot([x, x+1], [y, next_y_up], 'k-', lw=0.5)
                plt.plot([x, x+1], [y, next_y_down], 'k-', lw=0.5)
    plt.xlabel('Time Steps')
    plt.ylabel('Nodes')
    plt.xticks(range(n_periods))
    plt.yticks([2 * (i - n_periods / 2) for i in range(n_periods)])
    plt.grid(False)
    plt.axis(False)
    plt.savefig(f'plots/binmoial_tree_{n_periods-1}')
    plt.show()

def option_binomial_pre_paid(S0, K, T, r, q, sigma, n, D, TD, option_type='call', option_style='european', tree=False):
    """
    Calculates the price of an American or European option using a binomial tree model that accounts
    for a pre-paid dividend.
    """
    dt = T / n
    S = np.zeros((n + 1, n + 1))
    FP = np.zeros((n + 1, n + 1))
    option = np.zeros((n + 1, n + 1))
    exercise = np.full((n + 1, n + 1), False)
    S[0, 0] = S0
    FP[0, 0] = S0 - D * np.exp(-r * TD)
    sigma_F = sigma * S[0, 0] / FP[0, 0]
    u = np.exp((r - q) * dt + sigma_F * np.sqrt(dt))
    d = np.exp((r - q) * dt - sigma_F * np.sqrt(dt))
    p = (np.exp((r - q) * dt) - d) / (u - d)
    for i in range(1, n + 1):
        for j in range(i + 1):
            FP[j, i] = FP[0, 0] * (d ** j) * (u ** (i - j))
            S[j, i] = FP[j, i]
            if i * dt <= TD:
                S[j, i] += D * np.exp(-r * (TD - i * dt))
    if option_type == 'call':
        option[:, n] = np.maximum(S[:, n] - K, 0)
    elif option_type == 'put':
        option[:, n] = np.maximum(K - S[:, n], 0)
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            continuation_value = np.exp(-r * dt) * (p * option[j, i + 1] + (1 - p) * option[j + 1, i + 1])
            if option_style == 'american':
                if option_type == 'call':
                    option[j, i] = max(S[j, i] - K, continuation_value)
                else:
                    option[j, i] = max(K - S[j, i], continuation_value)
            else:
                option[j, i] = continuation_value
    for i in range(n, -1, -1):
        for j in range(i + 1):
            exercise_value = (S[j, i] - K if option_type == 'call' else K - S[j, i])
            exercise[j, i] = option[j, i] == exercise_value
    if tree:
        return S, FP, option, exercise
    return option[0, 0]
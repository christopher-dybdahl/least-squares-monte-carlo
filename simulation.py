import numpy as np
from scipy.stats import norm

def simulate_paths(S0, r, q, sigma, T, n_steps, n_paths):
    dt = T / n_steps
    drift = np.exp((r - q - 0.5 * sigma ** 2) * dt)
    shock = np.exp(sigma * np.sqrt(dt) * norm.ppf(np.random.rand(n_paths, n_steps)))
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    for t in range(1, n_steps + 1):
        S[:, t] = S[:, t - 1] * drift * shock[:, t - 1]
    return S

def simulate_paths_antithetic(S0, r, q, sigma, T, n_steps, n_paths):
    dt = T / n_steps
    drift = np.exp((r - q - 0.5 * sigma ** 2) * dt)
    shocks = np.random.normal(0, 1, (int(n_paths/2), n_steps))
    antithetic_shocks = -shocks
    combined_shocks = np.concatenate((shocks, antithetic_shocks))
    shock_multipliers = np.exp(sigma * np.sqrt(dt) * combined_shocks)
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    for t in range(1, n_steps + 1):
        S[:, t] = S[:, t - 1] * drift * shock_multipliers[:, t - 1]
    return S

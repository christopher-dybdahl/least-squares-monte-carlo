from scipy.stats import norm
import numpy as np
from scipy.optimize import brentq

def black_scholes(S, K, T, r, q, sigma, option_type='call'):
    """
    Computes the price of a European call or put option using the Black-Scholes formula.
    """
    d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return price

def implied_volatility_put(S, K, T, r, q, market_price):
    """
    Calculates the implied volatility for a European put option given a market price.
    """
    def error(sigma):
        return black_scholes(S, K, T, r, q, sigma, option_type='put') - market_price
    implied_vol = brentq(error, 1e-6, 1, xtol=1e-6)
    return implied_vol

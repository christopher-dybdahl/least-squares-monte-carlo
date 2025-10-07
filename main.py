import argparse
from config import *
from black_scholes import black_scholes, implied_volatility_put
from binomial_tree import option_binomial, plot_priced_tree, option_binomial_pre_paid
from simulation import simulate_paths, simulate_paths_antithetic
from lsm import longstaff_schwartz_lsm
from analysis import conf_int, simulate_and_analyze_exercise_times, exercise_time_analysis
import numpy as np
import matplotlib.pyplot as plt

def run_binomial():
    print("\nBinomial Tree (American Call):")
    price = option_binomial(S0, K, T, r, q, sigma, n_steps, option_type='call', type='american')
    print(f"American Call (Binomial): {price}")

def run_black_scholes():
    print("\nBlack-Scholes (European Call):")
    price = black_scholes(S0, K, T, r, q, sigma, option_type='call')
    print(f"European Call (Black-Scholes): {price}")

def run_lsm():
    print("\nLSM Monte Carlo (American Call):")
    MC_arr = np.zeros(M)
    for m in range(M):
        paths = simulate_paths(S0, r, q, sigma, T, n_steps, n_paths)
        american_call_price_LSM, _ = longstaff_schwartz_lsm(paths, K, r, T, n_steps, option_type='call')
        MC_arr[m] = american_call_price_LSM
    print(f"LSM American Call (mean): {np.mean(MC_arr)}")
    print(f"LSM American Call (var): {np.var(MC_arr)}")
    print(f"LSM American Call (conf int): {conf_int(MC_arr)}")

def run_regression():
    print("\nDifferent regression approaches (LSM):")
    degrees = [2, None, 4, None, 3, None, 4, 4, 4, 4]
    types = ['polynomial', 'linear', 'polynomial', 'exponential', 'poly_log', 'trigonometric', 'hermite', 'legendre', 'jacobi', 'laguerre']
    M_reg = 100
    for deg, typ in zip(degrees, types):
        american_call_price_LSM = 0
        for m in range(M_reg):
            paths = simulate_paths_antithetic(S0, r, q, sigma, T, n_steps, n_paths)
            american_call_price_LSM += longstaff_schwartz_lsm(paths, K, r, T, n_steps, option_type='call', poly_degree=deg, poly_type=typ)[0]/M_reg
        print(f"{typ:<20} {str(deg):<10} {american_call_price_LSM:<10}")

def run_dividends():
    print("\nBinomial tree plots for discrete dividends:")
    for TD in TDs_plot:
        S, FP, option, exercise = option_binomial_pre_paid(S0_plot, K_plot, T_plot, r_plot, q_plot, sigma_plot, n_plot, D_plot, TD, option_type='call', option_style='american', tree=True)
        plot_priced_tree(S, FP, option, exercise, scale=1, loc=0.8)
        print(f"Option price (TD={TD:.2f}): {option[0, 0]}")
    print("\nOption prices for a range of dividend times:")
    TDs_range = np.linspace(1/12, 11/12, 11)
    for TD in TDs_range:
        option_price = option_binomial_pre_paid(S0_plot, K_plot, T_plot, r_plot, q_plot, sigma_plot, n_plot, D_plot, TD, option_type='call', option_style='american', tree=False)
        print(f"Option price (TD={TD:.2f}): {option_price}")

def run_mcdonald_example():
    print("\nRecreating McDonald (2014) textbook example (page 354):")
    paths = np.array([
        [1.0, 1.09, 1.08, 1.34],
        [1.0, 1.16, 1.26, 1.54],
        [1.0, 1.22, 1.07, 1.03],
        [1.0, 0.93, 0.97, 0.92],
        [1.0, 1.11, 1.56, 1.52],
        [1.0, 0.76, 0.77, 0.90],
        [1.0, 0.92, 0.84, 1.01],
        [1.0, 0.88, 1.22, 1.34]
    ])
    price, exercise_times = longstaff_schwartz_lsm(paths, 1.1, 0.06, 3, 3, option_type='put')
    print(f"LSM price (put, McDonald example): {price}")
    print(f"Exercise times: {exercise_times}")

def run_all():
    run_binomial()
    run_black_scholes()
    run_lsm()
    run_regression()
    run_dividends()
    run_mcdonald_example()

def main():
    parser = argparse.ArgumentParser(description="Run option pricing analyses.")
    parser.add_argument('--analysis', type=str, default='all', choices=['all', 'binomial', 'black-scholes', 'lsm', 'regression', 'dividends', 'mcdonald'], help='Which analysis to run')
    args = parser.parse_args()
    if args.analysis == 'all':
        run_all()
    elif args.analysis == 'binomial':
        run_binomial()
    elif args.analysis == 'black-scholes':
        run_black_scholes()
    elif args.analysis == 'lsm':
        run_lsm()
    elif args.analysis == 'regression':
        run_regression()
    elif args.analysis == 'dividends':
        run_dividends()
    elif args.analysis == 'mcdonald':
        run_mcdonald_example()

if __name__ == "__main__":
    main()

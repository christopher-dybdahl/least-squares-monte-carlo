import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from lsm import longstaff_schwartz_lsm
from simulation import simulate_paths_antithetic

def conf_int(arr):
    return st.t.interval(0.95, len(arr)-1, loc=np.mean(arr), scale=st.sem(arr))

def simulate_and_analyze_exercise_times(S0, r, q, sigma, T, K, n_paths, n_steps, M, option_type='call', poly_degree=2, poly_type='polynomial'):
    all_exercise_times = np.array([])
    for _ in range(M):
        paths = simulate_paths_antithetic(S0, r, q, sigma, T, n_steps, n_paths)
        exercise_times = longstaff_schwartz_lsm(paths, K, r, T, n_steps, option_type=option_type, poly_degree=poly_degree, poly_type=poly_type)[1]
        all_exercise_times = np.concatenate((all_exercise_times, exercise_times[exercise_times < n_steps]))
    plt.hist(all_exercise_times, bins=n_steps//3, weights=np.ones_like(all_exercise_times)/M)
    plt.title('Histogram of Exercise Times for American Call Option')
    plt.xlabel('Time Step')
    plt.ylabel('Frequency of Exercises')
    plt.savefig('plots/exercise_times_histogram.png')
    plt.show()
    percentage_early_exercise = len(all_exercise_times) / (n_paths * M) * 100
    return percentage_early_exercise

def exercise_time_analysis(volatilities, n_paths, n_steps, S0=200, r=0.05, q=0.01, T=3, K=210, B=10, option_type='call', deg=2, typ='polynomial'):
    exercise_results = {}
    variances = {}
    for vol in volatilities:
        all_exercise_times = np.zeros((B, n_paths))
        for b in range(B):
            from simulation import simulate_paths_antithetic
            _, exercise_times = longstaff_schwartz_lsm(
                simulate_paths_antithetic(S0, r, q, vol, T, n_steps, n_paths),
                K, r, T, n_steps, option_type=option_type, poly_degree=deg, poly_type=typ
            )
            all_exercise_times[b] = exercise_times
        percentage_exercised_before_maturity = np.mean(all_exercise_times < n_steps, axis=1) * 100
        exercise_results[vol] = np.mean(percentage_exercised_before_maturity)
        variances[vol] = np.var(percentage_exercised_before_maturity)
    return exercise_results, variances

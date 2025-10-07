import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.special import eval_hermite, eval_legendre, eval_jacobi, genlaguerre

def calculate_continuation_value(stock_prices, discounted_payoff, method='polynomial', degree=2, alpha=0.5, beta=0.5):
    if method in ['hermite', 'legendre', 'jacobi', 'laguerre']:
        if method == 'hermite':
            basis_functions = np.column_stack([eval_hermite(d, stock_prices) for d in range(degree + 1)])
        elif method == 'legendre':
            basis_functions = np.column_stack([eval_legendre(d, stock_prices) for d in range(degree + 1)])
        elif method == 'jacobi':
            basis_functions = np.column_stack([eval_jacobi(d, alpha, beta, stock_prices) for d in range(degree + 1)])
        elif method == 'laguerre':
            basis_functions = np.column_stack([genlaguerre(d, 0)(stock_prices) for d in range(degree + 1)])
        regression_model = LinearRegression().fit(basis_functions, discounted_payoff)
        return regression_model.predict(basis_functions)
    elif method == 'polynomial':
        coeffs = np.polyfit(stock_prices, discounted_payoff, deg=degree)
        return np.polyval(coeffs, stock_prices)
    elif method == 'linear':
        A = np.vstack([stock_prices, np.ones(len(stock_prices))]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, discounted_payoff, rcond=None)
        return A @ coeffs
    elif method == 'exponential':
        log_prices = np.log(stock_prices)
        A = np.vstack([log_prices, np.ones(len(log_prices))]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, discounted_payoff, rcond=None)
        return np.exp(A @ coeffs)
    elif method == 'poly_log':
        log_prices = np.log(stock_prices)
        basis_functions = np.column_stack([stock_prices**i for i in range(1, degree + 1)] + [log_prices])
        coeffs, _, _, _ = np.linalg.lstsq(basis_functions, discounted_payoff, rcond=None)
        return basis_functions @ coeffs
    elif method == 'trigonometric':
        A = np.vstack([np.sin(stock_prices), np.cos(stock_prices), np.ones(len(stock_prices))]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, discounted_payoff, rcond=None)
        return A @ coeffs
    else:
        raise ValueError("Invalid method for continuation value calculation.")

def longstaff_schwartz_lsm(paths, K, r, T, n_steps, option_type, poly_degree=2, poly_type='polynomial'):
    dt = T / n_steps
    discount = np.exp(-r * dt)
    immediate_exercise_value = np.maximum(K - paths, 0) if option_type == 'put' else np.maximum(paths - K, 0)
    discounted_cash_flow = immediate_exercise_value[:, -1]
    exercise_times = np.full(paths.shape[0], n_steps)
    for t in range(n_steps - 1, 0, -1):
        in_the_money = (paths[:, t] < K) if option_type == 'put' else (paths[:, t] > K)
        paths_itm = np.where(in_the_money)[0]
        discounted_cash_flow *= discount
        if len(paths_itm) > 0:
            X = paths[paths_itm, t]
            Y = discounted_cash_flow[paths_itm]
            continuation_values = calculate_continuation_value(X, Y, poly_type, poly_degree)
            exercise = immediate_exercise_value[in_the_money, t] > continuation_values
            discounted_cash_flow[in_the_money] = np.where(exercise, immediate_exercise_value[in_the_money, t], Y)
            exercise_times[in_the_money] = np.where(exercise, t, exercise_times[in_the_money])
    option_price = np.mean(discounted_cash_flow) * discount
    return option_price, exercise_times

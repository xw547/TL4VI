import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import multivariate_normal
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

import traceback
import sys
sys.path.insert(0, '/home/xw547/Working/Ning/Giles1/Code/Python_Implementation/tl4vi_uai/helper/')

# Import your custom functions
from eif import eif_cpi_empirical
from empirical_cdf import empirical_rf_pdf

def simulate_with_seed_and_rho(seed, rho, n=1000, N=500, p=10, num_bootstrap = 100):
    try:
        np.random.seed(seed)

# Define covariance matrix
        sig = np.eye(p)
        sig[0, 1] = rho
        sig[1, 0] = rho

        # Generate data
        X = multivariate_normal.rvs(mean=np.zeros(p), cov=sig, size=n)
        X_reduced = X[:, 1:]
        beta = np.array([3, 0] + [0] * (p - 2))

        Y = X @ beta + np.random.normal(0, 1, size=n)

        # Create dataframes
        full_data = pd.DataFrame(np.column_stack([Y, X]), columns=["output"] + [f"X{i+1}" for i in range(p)])
        reduced_data = pd.DataFrame(np.column_stack([Y, X_reduced]), columns=["output"] + [f"X{i+2}" for i in range(p-1)])

        num_bootstrap = 100

        # Storage for results
        bootstrap_results = []

        for i in range(num_bootstrap):
            # Bootstrap resampling
            X_boot = pd.DataFrame(X).copy()
            X_boot = X_boot.sample(n=len(X_boot), replace=True, random_state=seed + i)
            X_boot = X_boot.values  # convert back to numpy array
            X_boot_reduced = X_boot[:, 1:]
            
            bootstrap_sample = pd.DataFrame(np.column_stack([Y, X_boot]), columns=["output"] + [f"X{i+1}" for i in range(p)])
            boot_reduced_data = pd.DataFrame(np.column_stack([Y, X_boot_reduced]), columns=["output"] + [f"X{i+2}" for i in range(p-1)])
            # Fit the full model
            full_model = XGBRegressor()
            full_model.fit(bootstrap_sample.iloc[:, 1:], bootstrap_sample["output"])
            
            # Fit the reduced model
            reduced_model = XGBRegressor()
            reduced_model.fit(bootstrap_sample.iloc[:, 2:], bootstrap_sample["output"])
            
            # Fit the reduced model for Y
            reduced_model_y = RandomForestRegressor(n_estimators=500, max_features=1/3)
            reduced_model_y.fit(boot_reduced_data.iloc[:, 1:], bootstrap_sample["output"])
            
            # Fit the reduced model for X1
            reduced_model_x = RandomForestRegressor(n_estimators=500, max_features=1/3)
            reduced_model_x.fit(boot_reduced_data.iloc[:, 1:], bootstrap_sample["X1"])
            
            # Compute empirical densities
            reduced_den_y = empirical_rf_pdf(
        forest=reduced_model_y, X_train=bootstrap_sample.iloc[:, 2:], X_test=bootstrap_sample.iloc[:, 2:], seed=seed + 2024
            )
            reduced_den_x = empirical_rf_pdf(
        forest=reduced_model_x, X_train=bootstrap_sample.iloc[:, 2:], X_test=bootstrap_sample.iloc[:, 2:], seed=seed + 2024
            )
            
            # Compute EIF
            eif_now = eif_cpi_empirical(bootstrap_sample, reduced_den_y, reduced_den_x, N, full_model, seed + 2024)
            
            # Store results
            bootstrap_results.append(np.mean(eif_now.iloc[:, 2]))


        # Return results
        return {
            "seed": seed,
            "rho": rho,
            "bootstrap_mean": np.mean(bootstrap_results),
            "bootstrap_var": np.var(bootstrap_results)
        }

    except Exception as e:
        raise e

def simulate_with_error_handling(seed, rho):
    try:
        return {"success": True, "result": simulate_with_seed_and_rho(seed, rho)}
    except Exception:
        return {"success": False, "seed": seed, "rho": rho, "error": traceback.format_exc()}

# Parameters
rho_values = [0.1, 0.2, 0.3, 0.4, 0.5]
num_seeds = 320
np.random.seed(2024)
random_seeds = np.random.choice(range(1, 10000), size=num_seeds, replace=False)

# Parallel execution
for rho in rho_values:
    results = Parallel(n_jobs=-1)(
        delayed(simulate_with_error_handling)(seed, rho) for seed in random_seeds
    )

    # Process results
    successful_results = [res["result"] for res in results if res["success"]]
    failed_results = [res for res in results if not res["success"]]

    # Save successful results
    results_df = pd.DataFrame(successful_results)
    file_name = f"bootstrap_simple_xgb_results_1e_3_rho_{rho}.csv"
    results_df.to_csv(file_name, index=False)
    print(f"Simulation complete for rho={rho}. Results saved to {file_name}.")

    # Log failures so numerically unstable results can be identified without rerunning the simulation.
    if failed_results:
        failed_file_name = f"bootstrap_simple_xgb_failures_rho_{rho}.log"
        with open(failed_file_name, "w") as f:
            for fail in failed_results:
                f.write(f"Seed: {fail['seed']}, Rho: {fail['rho']}\nError:\n{fail['error']}\n\n")
        print(f"Failures logged to {failed_file_name}.")

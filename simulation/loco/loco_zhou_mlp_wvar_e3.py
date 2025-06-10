import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import multivariate_normal
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from pygam import LinearGAM, s
from sklearn.neural_network import MLPRegressor

import traceback
import sys
sys.path.insert(0, '/home/xw547/Working/Ning/Giles1/Code/Python_Implementation/tl4vi_uai/helper/')

def simulate_with_seed_and_rho(seed, rho, n=1000, N=500, p=10):
    try:
        np.random.seed(seed)

        # Define covariance matrix
        indices = np.arange(p)
        sig = rho ** np.abs(indices[:, None] - indices)

        # Generate data
        X = multivariate_normal.rvs(mean=np.zeros(p), cov=sig, size=n)
        X_reduced = X[:, 1:]
        beta = np.array([3, 1, 1, 1, 1, 0, 0.5, 0.8, 1.2, 1.5])


        Y = X @ beta + np.random.normal(0, 1, size=n)

        # Create dataframes
        full_data = pd.DataFrame(np.column_stack([Y, X]), columns=["output"] + [f"X{i+1}" for i in range(p)])
        reduced_data = pd.DataFrame(np.column_stack([Y, X_reduced]), columns=["output"] + [f"X{i+2}" for i in range(p-1)])

        full_model = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),  # Larger network for more complex patterns
            activation='relu',
            solver='adam',
            max_iter=5000,     # Increased iterations for better convergence
            learning_rate='adaptive',  # Adaptive learning rate for better optimization
            early_stopping=True,  # Add early stopping to prevent overfitting
            validation_fraction=0.1,  # Use 10% of training data for validation
            n_iter_no_change=10,  # Number of iterations with no improvement to wait before stopping
            random_state=seed,
            tol=1e-3  # Use the existing tolerance variable
        )
        full_model.fit(full_data.iloc[:, 1:], full_data["output"])
        
        reduced_model = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),  # Larger network for more complex patterns
            activation='relu',
            solver='adam',
            max_iter=5000,     # Increased iterations for better convergence
            learning_rate='adaptive',  # Adaptive learning rate for better optimization
            early_stopping=True,  # Add early stopping to prevent overfitting
            validation_fraction=0.1,  # Use 10% of training data for validation
            n_iter_no_change=10,  # Number of iterations with no improvement to wait before stopping
            random_state=seed,
            tol=1e-3  # Use the existing tolerance variable
        )
        reduced_model.fit(reduced_data.iloc[:, 1:], reduced_data["output"])
        
        full_model_preds = full_model.predict(full_data.iloc[:, 1:])
        reduced_model_preds = reduced_model.predict(reduced_data.iloc[:, 1:])
        mse_full = mean_squared_error(full_data["output"], full_model_preds)
        mse_reduced = mean_squared_error(full_data["output"], reduced_model_preds)
        loco = np.mean(mse_reduced - mse_full)

        # Return results
        return {
            "seed": seed,
            "rho": rho,
            "loco": loco
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
    file_name = f"loco_zhou_mlp_simulation_1e_3_rho_{rho}.csv"
    results_df.to_csv(file_name, index=False)
    print(f"Simulation complete for rho={rho}. Results saved to {file_name}.")

    # Log failures so numerically unstable results can be identified without rerunning the simulation.
    if failed_results:
        failed_file_name = f"loco_zhou_mlp_failures_rho_{rho}.log"
        with open(failed_file_name, "w") as f:
            for fail in failed_results:
                f.write(f"Seed: {fail['seed']}, Rho: {fail['rho']}\nError:\n{fail['error']}\n\n")
        print(f"Failures logged to {failed_file_name}.")

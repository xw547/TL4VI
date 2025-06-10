import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import multivariate_normal
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
from xgboost import XGBRegressor
from pygam import LinearGAM, s
from pygam.terms import Term

import traceback
import sys
sys.path.insert(0, '/home/xw547/Working/Ning/Giles1/Code/Python_Implementation/scripts')

# Import your custom functions
from eif import eif_cpi_empirical
from empirical_cdf import empirical_rf_pdf

def simulate_with_seed_and_rho(seed, rho, n=1000, N=500, p=10):
    try:
        np.random.seed(seed)
        
        # Define covariance matrix
        indices = np.arange(p)
        sig = rho ** np.abs(indices[:, None] - indices)

        # Generate data
        X = multivariate_normal.rvs(mean=np.zeros(p), cov=sig, size=n)
        X_reduced = X[:, 1:]
        #beta = np.array([3, 1, 1, 1, 1, 0, 0.5, 0.8, 1.2, 1.5])
        epsilon = np.random.normal(loc=0, scale=1, size=n)

        # Extract specific X variables for use in the formula
        X1, X2, X3, X6, X7, X10 = X[:, 0], X[:, 1], X[:, 2], X[:, 5], X[:, 6], X[:, 9]

        # Compute Y based on the provided formula
        Y = (
            10*np.sin(X1) +
            10*np.cos(X2) +
            3*X3*X6 +
            3*X10 +
            epsilon
        )

        # Create dataframes
        full_data = pd.DataFrame(np.column_stack([Y, X]), columns=["output"] + [f"X{i+1}" for i in range(p)])
        reduced_data = pd.DataFrame(np.column_stack([Y, X_reduced]), columns=["output"] + [f"X{i+2}" for i in range(p-1)])

        # Fit models
        #full_model = RandomForestRegressor(n_estimators=500, max_features=1/3)
        full_model = LinearGAM(s(0)+s(1)+s(2)+s(3)+s(4)+s(5)+s(6)+s(7)+s(8)+s(9))
        # full_model = XGBRegressor(n_estimators=500)
        full_model.fit(full_data.iloc[:, 1:], full_data["output"])
        
        reduced_model = LinearGAM(s(0)+s(1)+s(2)+s(3)+s(4)+s(5)+s(6)+s(7)+s(8))
        # reduced_model = XGBRegressor(n_estimators=500)
        # reduced_model = RandomForestRegressor(n_estimators=500, max_features=1/3)
        reduced_model.fit(reduced_data.iloc[:, 1:], reduced_data["output"])

        reduced_model_y = RandomForestRegressor(n_estimators=500, max_features=1/3)
        reduced_model_y.fit(reduced_data.iloc[:, 1:], reduced_data["output"])
        

        reduced_model_x = RandomForestRegressor(n_estimators= 500, max_features=1/3)
        reduced_model_x.fit(full_data.iloc[:, 2:], full_data["X1"]) 

        # Empirical densities
        reduced_den_y = empirical_rf_pdf(forest=reduced_model_y, X_train=full_data.iloc[:, 2:], X_test=full_data.iloc[:, 2:], seed=seed+2024)
        reduced_den_x = empirical_rf_pdf(forest=reduced_model_x, X_train=full_data.iloc[:, 2:], X_test=full_data.iloc[:, 2:], seed=seed+2024)

        # Calculate EIF
        eif_now = eif_cpi_empirical(full_data, reduced_den_y, reduced_den_x, N, full_model, seed+2024)

        # Calculate CPI metrics
        naive_CPI = np.mean(eif_now.iloc[:, 2])
        plug_in_CPI = np.mean(eif_now.iloc[:, 0] - eif_now.iloc[:, 1] + eif_now.iloc[:, 2])
        plug_in_var = np.mean((eif_now.iloc[:, 0] - eif_now.iloc[:, 1] + eif_now.iloc[:, 2]-plug_in_CPI)**2)

        # Calculate EIF
        eif = eif_now.iloc[:, 0] - 2 * eif_now.iloc[:, 1] + eif_now.iloc[:, 2]

        # Prepare regress_frame
        curr_offset = reduced_model.predict(full_data.iloc[:, 2:])
        regress_frame = pd.DataFrame({
            'y': full_data.iloc[:, 0],
            'curr_offset': curr_offset,
            'obs': eif
        })

        # Perform linear regression
        lm = LinearRegression(fit_intercept=False)
        lm.fit(regress_frame[['curr_offset', 'obs']], regress_frame['y'])
        epsilon_now = lm.coef_[1]
        next_offset = lm.predict(regress_frame[['curr_offset', 'obs']])

        # Update densities
        eif_scaled = eif * epsilon_now/40
        reduced_den_x_update = reduced_den_x + np.outer(eif_scaled, np.ones(reduced_den_x.shape[1]))
        reduced_den_x_update = np.maximum(reduced_den_x_update, 0)
        reduced_den_x_update = np.maximum(reduced_den_x_update, 0)
        row_sums = reduced_den_x_update.sum(axis=1, keepdims=True)
        reduced_den_x_update = np.divide(reduced_den_x_update, row_sums, where=row_sums!=0)

        reduced_den_y_update = reduced_den_y + np.outer(eif_scaled, np.ones(reduced_den_y.shape[1]))
        reduced_den_y_update = np.maximum(reduced_den_y_update, 0)
        row_sums = reduced_den_y_update.sum(axis=1, keepdims=True)
        reduced_den_y_update = np.divide(reduced_den_y_update, row_sums, where=row_sums!=0)

        # Recalculate EIF
        eif_now = eif_cpi_empirical(full_data, reduced_den_y_update, reduced_den_x_update, N, full_model, 2024)
        tmle_onestep = np.mean(eif_now.iloc[:, 0] - eif_now.iloc[:, 1] + eif_now.iloc[:, 2])

        # Iterative process for TMLE
        tol = 1e-3
        epsilon = 1
        while epsilon > tol:
            eif = eif_now.iloc[:, 0] - 2 * eif_now.iloc[:, 1] + eif_now.iloc[:, 2]
            regress_frame['obs'] = eif
            regress_frame['curr_offset'] = next_offset
            
            lm.fit(regress_frame[['curr_offset', 'obs']], regress_frame['y'])
            epsilon_now = lm.coef_[1]
            next_offset = lm.predict(regress_frame[['curr_offset', 'obs']])
            
            eif_scaled = eif * epsilon_now/20

            reduced_den_x_update = reduced_den_x + np.outer(eif_scaled, np.ones(reduced_den_x.shape[1]))
            reduced_den_x_update = np.maximum(reduced_den_x_update, 0)
            row_sums = reduced_den_x_update.sum(axis=1, keepdims=True)
            reduced_den_x_update = np.divide(reduced_den_x_update, row_sums, where=row_sums!=0)
            
            reduced_den_y_update = reduced_den_y + np.outer(eif_scaled, np.ones(reduced_den_y.shape[1]))
            reduced_den_y_update = np.maximum(reduced_den_y_update, 0)
            row_sums = reduced_den_y_update.sum(axis=1, keepdims=True)
            reduced_den_y_update = np.divide(reduced_den_y_update, row_sums, where=row_sums!=0)
            
            eif_now = eif_cpi_empirical(full_data, reduced_den_y_update, reduced_den_x_update, N, full_model, seed+2024)
            epsilon = abs(epsilon_now)

        tmle_final = np.mean(eif_now.iloc[:, 0] - eif_now.iloc[:, 1] + eif_now.iloc[:, 2])
        tmle_var = np.mean((eif_now.iloc[:, 0] - eif_now.iloc[:, 1] + eif_now.iloc[:, 2] - tmle_final)**2)

        # Return results
        return {
            "seed": seed,
            "rho": rho,
            "naive_CPI": naive_CPI,
            "plug_in_CPI": plug_in_CPI,
            "plug_in_var": plug_in_var,
            "tmle_onestep": tmle_onestep,
            "tmle_final": tmle_final,
            "tmle_var": tmle_var
        }

    except Exception as e:
        raise e

def simulate_with_error_handling(seed, rho):
    try:
        return {"success": True, "result": simulate_with_seed_and_rho(seed, rho)}
    except Exception:
        return {"success": False, "seed": seed, "rho": rho, "error": traceback.format_exc()}

# Parameters
rho_values = [0.2, 0.4]
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
    file_name = f"extra_large_sincos_nosq_gam_wvar_simulation_results_1e_3_rho_{rho}.csv"
    results_df.to_csv(file_name, index=False)
    print(f"Simulation complete for rho={rho}. Results saved to {file_name}.")

    # Log failures
    if failed_results:
        failed_file_name = f"extra_large_sincos_nosq_gam_wvar_failures_rho_{rho}.log"
        with open(failed_file_name, "w") as f:
            for fail in failed_results:
                f.write(f"Seed: {fail['seed']}, Rho: {fail['rho']}\nError:\n{fail['error']}\n\n")
        print(f"Failures logged to {failed_file_name}.")

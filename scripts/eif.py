import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def eif_cpi_empirical(full_data, reduced_den_y, reduced_den_x, N, full_model, seed):
    """
    Compute different terms of Efficient Influence Function (EIF) for Conditional Predictive Importance (CPI).
    
    Parameters:
    - full_data: pd.DataFrame, the data that we use to train the full model. First column as response
      and the rest be covariate/features.
    - reduced_den_y: np.ndarray, conditional density matrix for Y.
    - reduced_den_x: np.ndarray, conditional density matrix for X.
    - N: int, number of samples for bootstrapping.
    - full_model: RandomForestRegressor, the full model trained on the dataset.
    - seed: int, the seed number used to enhance reproducibility.

    Returns:
    - pd.DataFrame: containing first_term, second_term, and third_term arrays.
    """
    
    n = full_data.shape[0]
    first_term = np.zeros(n)
    second_term = np.zeros(n)
    third_term = np.zeros(n)
    full_pred = full_model.predict(full_data.iloc[:, 1:])  # Assuming full_model supports `.predict`
    Y = full_data.iloc[:, 0]
    np.random.seed(seed)
    for i in range(n):
        # Bootstrapped samples for Y and X
        curr_y = np.random.choice(full_data.iloc[:, 0], size=N, replace=True, p=reduced_den_y[i])
        curr_x = np.random.choice(full_data.iloc[:, 1], size=N, replace=True, p=reduced_den_x[i])
        
        # Create new data for predictions
        second_data = pd.DataFrame(np.tile(full_data.iloc[i].values, (N, 1)), columns=full_data.columns)
        second_data.iloc[:, 1] = curr_x
        
        # Obtain predictions
        second_pred = full_model.predict(second_data.iloc[:, 1:])
        
        # Calculating different terms of the efficient influence function.
        first_term[i] = np.mean(np.square(curr_y - full_pred[i]))
        second_term[i] = np.mean(np.square(curr_y - second_pred))
        third_term[i] = np.mean(np.square(Y[i] - second_pred))


    results = pd.DataFrame({
        "first_term": first_term,
        "second_term": second_term,
        "third_term": third_term
    })

    return results

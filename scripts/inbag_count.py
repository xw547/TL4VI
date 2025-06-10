import numpy as np
from sklearn.ensemble._forest import _generate_sample_indices
from sklearn.ensemble._forest import _get_n_samples_bootstrap

def calc_inbag(n_samples, forest):
    """
    Calculate the in-bag matrix for a given random forest.

    Parameters:
    - n_samples: int, the number of samples in the dataset.
    - forest: sklearn.ensemble.RandomForestClassifier or RandomForestRegressor,
      the trained random forest model.

    Returns:
    - inbag: np.ndarray of shape (n_samples, n_trees),
      where each element represents the number of times a sample is included
      in the bootstrap sample for a given tree.
    """
    n_trees = forest.n_estimators
    inbag = np.zeros((n_samples, n_trees), dtype=int)
    
    for t_idx in range(n_trees):
        n_samples_bootstrap = _get_n_samples_bootstrap(n_samples, forest.max_samples)
        sample_idx = _generate_sample_indices(forest.estimators_[t_idx].random_state, n_samples, n_samples_bootstrap)
        inbag[:, t_idx] = np.bincount(sample_idx, minlength=n_samples)
    
    return inbag

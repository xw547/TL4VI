import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from inbag_count import calc_inbag

def empirical_rf_pdf(forest, X_train, X_test, seed):
    """
    Calculate empirical conditional density for X_test

    Parameters:
    - forest: RandomForestRegressor, the trained reduced random forest model (using only covariates from Z),
      note that we need the conditional density so we can only use the covariates that we wish to condition on.
    - X_train: Dataframe, the training data used to train the reduced random forest model.
    - X_test: Dataframe, the test data which we wish to obtain the conditional density from.
    - seed: int, the seed number used to enhance reproducibility.

    Returns:
    - inbag: np.ndarray of shape (n_samples, n_trees),
      where each element represents the number of times a sample is included
      in the bootstrap sample for a given tree.
    """

    # Ensure input arrays are in the correct format
    # X_train = np.array(X_train)
    # X_test = np.array(X_test)

    np.random.seed(seed)

    # Get terminal nodes of training observations
    train_terminal_nodes = forest.apply(X_train)

    # Bagging counts for in-bag samples
    bag_counts = calc_inbag(X_train.shape[0], forest)

    # Setting terminal nodes to NaN for in-bag samples
    train_terminal_nodes = np.where(bag_counts != 0, np.nan, train_terminal_nodes)

    # Create a DataFrame for train terminal nodes
    train_nodes_emp = pd.DataFrame(train_terminal_nodes, columns=[f"tree_{i}" for i in range(train_terminal_nodes.shape[1])])
    train_nodes_emp["obsid_train"] = train_nodes_emp.index

    # Melt train terminal nodes to long format
    train_nodes_emp = train_nodes_emp.melt(id_vars="obsid_train", var_name="tree", value_name="terminal_node").dropna()

    # Collapse to unique tree/node combinations
    train_nodes_emp = train_nodes_emp.groupby(["tree", "terminal_node"]).agg({"obsid_train": list}).reset_index()

    # Get terminal nodes of test observations
    test_terminal_nodes = forest.apply(X_test)

    # Create a DataFrame for test terminal nodes
    # test_preds = forest.predict(X_test)
    test_nodes_emp = pd.DataFrame(test_terminal_nodes, columns=[f"tree_{i}" for i in range(test_terminal_nodes.shape[1])])
    test_nodes_emp["testid_test"] = test_nodes_emp.index
    # test_nodes_emp["pred"] = test_preds

    # Melt test terminal nodes to long format
    test_nodes_emp = test_nodes_emp.melt(id_vars=["testid_test"], var_name="tree", value_name="terminal_node")

    # Merge train and test nodes by tree and terminal node
    merged_data = pd.merge(test_nodes_emp, train_nodes_emp, on=["tree", "terminal_node"], how="left")

    # Initialize result matrix
    result_matrix = np.zeros((X_test.shape[0], X_train.shape[0]))

    # Populate the result matrix
    for i in range(X_test.shape[0]):
        entries = table_to_density(merged_data, i)
        result_matrix[i, entries.index.astype(int)] = entries.values

    return result_matrix

# Helper function to calculate density from table (replace with relevant implementation if needed)
def table_to_density(merged_data, test_id):
    temp_list = merged_data[merged_data["testid_test"] == test_id]["obsid_train"].explode().dropna()
    entries = temp_list.value_counts()
    return entries / len(temp_list)

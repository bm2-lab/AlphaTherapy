from sklearn import metrics
from sklearn.linear_model import LinearRegression

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

def pcc(y_true, y_pred, low_gene_thre, high_gene_thre):
    """
    Calculate Pearson correlation coefficient (PCC) average by sample.

    Parameters
    ----------
    y_true : np.ndarray [n*m]
        True labels
    y_pred : np.ndarray [n*m]
        Predicted labels
    low_gene_thre : float
        Lower threshold for gene significance selection
    high_gene_thre : float
        Higher threshold for gene significance selection

    Returns
    -------
    list
        [original_pcc_arr, precision_pcc_arr]
        - original_pcc_arr: PCC for all samples
        - precision_pcc_arr: PCC after applying precision thresholds
    """

    pcc_value = 0.0
    pcc_value_precision = 0.0
    sample_number = y_pred.shape[0]
    precision_null_sample_number = 0

    original_pcc_arr = np.zeros(sample_number)
    precision_pcc_arr = np.zeros(sample_number)

    # Calculate PCC by gene
    for i in range(sample_number):
        flag = False

        # 1. Standard PCC
        original_pcc_arr[i] = np.corrcoef(y_true[i, :], y_pred[i, :])[0, 1]
        pcc_value += np.corrcoef(y_true[i, :], y_pred[i, :])[0, 1]

        # 2. Precision PCC (focusing on true data)
        a = y_true[i, :].copy()
        b = y_pred[i, :].copy()
        low_gene_thre_val, high_gene_thre_val = np.percentile(np.abs(a), [low_gene_thre, high_gene_thre])
        ind = (np.abs(a) > low_gene_thre_val) & (np.abs(a) <= high_gene_thre_val)

        if np.sum(ind) > 1:
            a = a[ind]
            b = b[ind]
            temp1 = np.corrcoef(a, b)[0, 1]
            pcc_value_precision += temp1
            precision_pcc_arr[i] = temp1
        else:
            precision_null_sample_number += 1
            flag = True

    # Average PCC values
    pcc_value = pcc_value / sample_number
    pcc_value_precision = pcc_value_precision / (sample_number - precision_null_sample_number)

    return [original_pcc_arr, precision_pcc_arr]


# Set the root directory of the project
ROOT_DIR = Path(os.getcwd()).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from path import DATA_DIR  

# Load preprocessed data
input_file = DATA_DIR / "preprocessed_data_2025/state_transition_model_input_data/step2_model_training_dataset_random_split_0.pkl"
with open(input_file, "rb") as f:
    (
        train_drug, train_x, train_y,
        valid_drug, valid_x, valid_y,
        test_drug, test_x, test_y,
        scaler_control, scale_change, y
    ) = pickle.load(f)


# Threshold the input data 
def threshold_data(X, low_thre_percentile, high_thre_percentile):
    """
    Apply thresholds to the input data based on percentiles.

    Parameters
    ----------
    X : np.ndarray
        The input data
    low_thre_percentile : float
        Lower percentile threshold
    high_thre_percentile : float
        Higher percentile threshold

    Returns
    -------
    X_ : np.ndarray
        The thresholded data with values outside the thresholds set to 0
    """
    X_ = X.copy()

    # Iterate through each sample and apply thresholding
    for i in range(X.shape[0]):
        a = np.abs(X[i, :])
        low_gene_thre_val, high_gene_thre_val = np.percentile(a, [low_thre_percentile, high_thre_percentile])
        mask = (a < low_gene_thre_val) | (a > high_gene_thre_val)
        X_[i, mask] = 0
    
    return X_


# Prepare training and testing data
train_x = np.concatenate([train_y, valid_y], axis=0)
train_y = np.concatenate([train_y, valid_y], axis=0)
test_x = test_y
test_y = test_y


# Linear model prediction for different threshold segments
pcc_arr = []
pcc_label = [] 
for segmentation in range(0, 100, 10):
    train_x_ = threshold_data(train_x, segmentation, segmentation + 10)
    test_x_ = threshold_data(test_x, segmentation, segmentation + 10)

    model = LinearRegression()
    model.fit(train_x_, train_y)  # Train the model

    y_pred = model.predict(test_x_)  # Make predictions

    # Calculate PCC for each sample
    pcc_arr.extend([np.corrcoef(test_y[i, :], y_pred[i, :])[0, 1] for i in range(test_x.shape[0])])
    pcc_label.extend([segmentation] * test_x.shape[0])


# Create a DataFrame for plotting
plot_df = pd.DataFrame({"pcc": pcc_arr, "label": pcc_label}) 

# Save the results to a file
output_file = DATA_DIR / "result_2025/state_transition_model/default_model/LinearRegression.pkl"
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, "wb") as f:
    pickle.dump(plot_df, f)


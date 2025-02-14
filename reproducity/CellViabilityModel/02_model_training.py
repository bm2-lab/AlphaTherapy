##############################################
# Script Summary
##############################################
# This script trains a cell viability prediction model using a Ridge regression algorithm. 
# The steps include:
# 1. Data preprocessing: loading x and y data.
# 2. Splitting the dataset into training and testing sets.
# 3. Training the model using 20 iterations of 50% random training/testing splits and calculating the Pearson correlation coefficient (PCC).
# 4. Saving the trained model.
##############################################

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# Set data directory
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.resolve().parent
sys.path.append(str(ROOT_DIR))
from path import DATA_DIR 


def generate_cell_viability_model():
    """
    This function generates a cell viability prediction model using Ridge regression.
    """

    # 1. Load the signatures and viability data

    x_path =DATA_DIR / "preprocessed_data_2025" / "cell_viability_model_data" / "X.csv"
    y_path = DATA_DIR / "preprocessed_data_2025" / "cell_viability_model_data" / "Y.csv"

    signatures_info = pd.read_table(x_path, sep=",", header=0, index_col=0)
    sig_info = pd.read_table(y_path, sep=",", header=0, index_col=0)
    sig_info["pert_dose"] = np.power(10, sig_info["log10_pert_dose"])

    # Prepare the feature matrix (X) and target values (Y)
    X = signatures_info.loc[sig_info.index].values
    Y = sig_info["cpd_avg_pv"].values

     # 2. Split data and use 20 iterations of 50% random split for model validation
    np.random.seed(666)
    sample_number = len(Y)
    print("Sample Size: ", len(Y))

    sample_ind = np.arange(sample_number)
    np.random.shuffle(sample_ind)

    # 2.1 Cross-validation: 20 iterations of 50% random split
    train_number = int(sample_number / 2)
    pcc = np.zeros(20)
    for i in range(20):
        # Randomly choose test and train indices
        test_ind = np.random.choice(sample_number, train_number, replace=False)
        train_ind = np.ma.array(np.arange(sample_number), mask=True)
        train_ind[test_ind] = False
        train_ind = np.arange(sample_number)[train_ind.mask]

        # Data preparation
        train_x = X[train_ind, :]
        train_y = Y[train_ind]
        test_x = X[test_ind, :]
        test_y = Y[test_ind]

        # Model fitting
        model = Ridge()
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)

        # Calculate Pearson correlation coefficient (PCC) for each iteration
        pcc[i] = np.corrcoef(test_y, pred_y)[0, 1]

    print("20-fold Cross Validation PCC: ", pcc)

    # Final model fitting with the entire dataset
    model = Ridge()
    model.fit(X[sample_ind, :], Y[sample_ind])

    # Save the model using pickle

    model_file = DATA_DIR / "working_log_2025" / "cell_viability_model" / "cv_model.pkl"
    model_file.parent.mkdir(parents=True, exist_ok=True)    
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    print("Successful save cell viability model in %s!" % model_file)

    result_file = DATA_DIR / "result_2025" / "cell_viability_model" / "cv_result.pkl"   
    result_file.parent.mkdir(parents=True, exist_ok=True)    
    with open(result_file, "wb") as f:
        pickle.dump(pcc, f)
    print("Successful save cell viability pcc in %s!" % result_file)


if __name__ == "__main__":
    generate_cell_viability_model()


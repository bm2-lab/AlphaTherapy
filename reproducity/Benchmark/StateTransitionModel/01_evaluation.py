import os
import sys
from pathlib import Path

# Set root and data directories
ROOT_DIR = Path(os.getcwd()).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from path import DATA_DIR

print("ROOT_DIR:", ROOT_DIR)
print("DATA_DIR:", DATA_DIR)

import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
import numpy as np
from scipy.stats import pearsonr


def r2_scores_sample(y_true, y_pred):
    r2_scores = np.array([r2_score(y_true[i], y_pred[i]) for i in range(len(y_true))])
    return r2_scores


def pcc_sample(y_true, y_pred):
    pcc_scores = []
    for i in range(len(y_true)):
        pcc, _ = pearsonr(y_true[i], y_pred[i])
        pcc_scores.append(pcc)
    return np.array(pcc_scores)


def rmse_sample(y_true, y_pred):
    rmse_scores = []
    for i in range(len(y_true)):
        rmse = np.sqrt(np.mean((y_true[i] - y_pred[i]) ** 2))
        rmse_scores.append(rmse)
    return np.array(rmse_scores)


with open(DATA_DIR / "preprocessed_data_2025/state_transition_model_benchmark/chemCPA/scaler.pkl", "rb") as f:
    scaler_ctrl, scaler_treat = pickle.load(f)


def performance(data_file, predict_file):
    with open(data_file, "rb") as f:
        train_drug, train_x, train_y, valid_drug, valid_x, valid_y, test_drug, test_x, test_y, scaler_control, scaler_change, y = pickle.load(f)

    with open(predict_file, "rb") as f:
        true_change_scale, pred_change_scale = pickle.load(f)

    true_control = scaler_control.inverse_transform(test_x)
    pred_change = scaler_change.inverse_transform(pred_change_scale)
    true_change = scaler_change.inverse_transform(true_change_scale)
    pred_treat = true_control + pred_change
    true_treat = true_control + true_change

    pred_treat_scale = scaler_treat.transform(pred_treat)
    true_treat_scale = scaler_treat.transform(true_treat)

    r2_scores = r2_scores_sample(true_treat_scale, pred_treat_scale)
    pcc_scores = pcc_sample(true_treat_scale, pred_treat_scale)
    rmse_scores = rmse_sample(true_treat_scale, pred_treat_scale)

    return r2_scores, pcc_scores, rmse_scores

# random split 1
data_file = DATA_DIR / "preprocessed_data_2025/state_transition_model_input_data/step2_model_training_dataset_random_split_1.pkl"
predict_file = DATA_DIR / "working_log_2025/state_transition_model/default_model/random_predict_1.pkl"
random_split_r2_arr, random_split_pcc_arr, random_split_rmse_arr = performance(data_file, predict_file)

# cell blind split 0
data_file = DATA_DIR / "preprocessed_data_2025/state_transition_model_input_data/step2_model_training_dataset_cell_blind_0.pkl"
predict_file = DATA_DIR / "working_log_2025/state_transition_model/default_model/cell_blind_0.pkl"
cell_split_r2_arr, cell_split_pcc_arr, cell_split_rmse_arr = performance(data_file, predict_file)

# compound blind split 0
data_file = DATA_DIR / "preprocessed_data_2025/state_transition_model_input_data/step2_model_training_dataset_chemical_blind_0.pkl"
predict_file = DATA_DIR / "working_log_2025/state_transition_model/default_model/chemical_blind_0.pkl"
chemical_split_r2_arr, chemical_split_pcc_arr, chemical_split_rmse_arr = performance(data_file, predict_file)

output_file = DATA_DIR / "result_2025/Benchmark/state_transition_model/performance.pkl"
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, "wb") as f:
    pickle.dump([random_split_r2_arr, random_split_pcc_arr, random_split_rmse_arr, cell_split_r2_arr, cell_split_pcc_arr, cell_split_rmse_arr, chemical_split_r2_arr, chemical_split_pcc_arr, chemical_split_rmse_arr], f)
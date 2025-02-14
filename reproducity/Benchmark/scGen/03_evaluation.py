import pickle
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.resolve().parent.resolve().parent
sys.path.append(str(ROOT_DIR))
from path import DATA_DIR  

with open("./results/scgen_predition_result.h5ad", "rb") as f:
    y_true_result, y_pred_result, error_log = pickle.load(f)

y_true_result = np.vstack(y_true_result)
y_pred_result = np.vstack(y_pred_result)


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

scgen_r2_cell_blind = r2_scores_sample(y_true_result, y_pred_result)
scgen_pcc_cell_blind = pcc_sample(y_true_result, y_pred_result)
scgen_rmse_cell_blind = rmse_sample(y_true_result, y_pred_result)

out_file_name = DATA_DIR / f"result_2025/Benchmark/scGen/scgen.pkl"
out_file_name.parent.mkdir(parents=True, exist_ok=True)
with open(out_file_name, "wb") as f:
    pickle.dump([scgen_r2_cell_blind, scgen_pcc_cell_blind, scgen_rmse_cell_blind], f)

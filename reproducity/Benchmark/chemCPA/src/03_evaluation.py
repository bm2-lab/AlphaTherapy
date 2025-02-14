import pickle
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.resolve().parent.resolve().parent.parent
sys.path.append(str(ROOT_DIR))
print(ROOT_DIR)
from path import DATA_DIR  


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
        rmse = np.sqrt(np.mean((np.array(y_true[i]) - np.array(y_pred[i])) ** 2))
        rmse_scores.append(rmse)
    return np.array(rmse_scores)

for dataset_type in ["random_split", "cell_blind", "chemical_blind"]:

    result_file_name = ROOT_DIR / f"Benchmark/chemCPA/chemCPA-main/testing_result_{dataset_type}.pkl"
    with open(result_file_name, "rb") as f:
        y_true_dict, y_pred_dict = pickle.load(f)

    r2_all = []
    pcc_all = []
    rmse_all = []
    for k in y_true_dict:
        r2_all.append(r2_scores_sample(y_true_dict[k].cpu(), y_pred_dict[k].cpu()))
        pcc_all.append(pcc_sample(y_true_dict[k].cpu(), y_pred_dict[k].cpu()))
        rmse_all.append(rmse_sample(y_true_dict[k].cpu(), y_pred_dict[k].cpu()))

    r2_all = np.concatenate(r2_all)
    pcc_all = np.concatenate(pcc_all)
    rmse_all = np.concatenate(rmse_all)

    out_file_name = DATA_DIR / f"result_2025/Benchmark/chemCPA/performance_result_{dataset_type}.pkl"
    out_file_name.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file_name, "wb") as f:
        pickle.dump([r2_all, pcc_all, rmse_all], f)
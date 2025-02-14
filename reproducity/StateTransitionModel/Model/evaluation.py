import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from utils import sample_based_pcc  

# Load data directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))
from path import DATA_DIR  

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

def calculate_and_save_pcc(pred_file, output_file1, output_file2, segmentation_num=10):
    """
    Calculate PCC based on prediction file and save the results.

    Parameters:
    - pred_file (str or Path): Path to the prediction file (pickle format).
    - output_file (str or Path): Path to save the PCC results (CSV format).
    - segmentation_num (int): Number of segments to divide for PCC calculation.

    Returns:
    - None
    """
    pred_file = Path(pred_file)
    output_file1 = Path(output_file1)
    output_file2 = Path(output_file2)



    # Load true and predicted values from the file
    with open(pred_file, "rb") as f:
        y_true, y_pred = pickle.load(f)

    all_precision_pcc_arr = []

    # Compute PCC for each threshold range
    for thre in range(0, 100, segmentation_num):
        original_pcc_arr, precision_pcc_arr = sample_based_pcc(y_true, y_pred, thre, (thre + segmentation_num))
        all_precision_pcc_arr.append(precision_pcc_arr)

    # Reshape and create a DataFrame
    original_pcc_arr = np.array(original_pcc_arr).reshape(-1)
    state_transition_original_pcc_df = pd.DataFrame({
        "pcc": original_pcc_arr,
    })

    # Reshape and create a DataFrame
    all_precision_pcc_arr = np.array(all_precision_pcc_arr).reshape(-1)
    state_transition_precision_pcc_df = pd.DataFrame({
        "pcc": all_precision_pcc_arr,
        "label": np.repeat(np.arange(1, segmentation_num + 1), y_true.shape[0])
    })

    # Ensure output directory exists
    output_file1.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    state_transition_original_pcc_df.to_csv(output_file1)
    state_transition_precision_pcc_df.to_csv(output_file2)


# 1. Evaluation for random, chemical blind, cell blind, dataset
for data_prefix in ["random_predict", "chemical_blind", "cell_blind"]:
    for split_index in range(10):
        calculate_and_save_pcc(
        pred_file = DATA_DIR / f"working_log_2025/state_transition_model/default_model/{data_prefix}_{split_index}.pkl",
        output_file1 = DATA_DIR / f"result_2025/state_transition_model/default_model/state_transition_original_pcc_{data_prefix}_{split_index}.csv",
        output_file2 = DATA_DIR / f"result_2025/state_transition_model/default_model/state_transition_precision_pcc_{data_prefix}_{split_index}.csv",
        )



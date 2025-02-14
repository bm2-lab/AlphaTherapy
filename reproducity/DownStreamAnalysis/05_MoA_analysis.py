import os
import pickle
import numpy as np
import pandas as pd

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.resolve().parent
sys.path.append(str(ROOT_DIR))
from path import DATA_DIR

from scipy.stats import binom
from statsmodels.stats.multitest import multipletests

def binomial_test(k, M, n, N):
    """
    Calculate the probability of i >= k in a binomial distribution.

    Parameters
    ----------
    k : int
        Number of successful events in the selected sample.
    M : int
        Total size of the population.
    n : int
        Number of successful events in the population.
    N : int
        Size of the selected sample.

    Returns
    -------
    float
        Probability of i >= k in the binomial distribution.
    """
    prob = binom.cdf(k - 1, N, n / M)
    return 1 - prob

def calculate_fold_change(selected_cnt, total_selected_cnt, bg_cnt, total_bg_cnt):
    """
    Calculate the Fold Change value.

    Parameters
    ----------
    selected_cnt : int
        Number of successful events in the selected sample.
    total_selected_cnt : int
        Total number of events in the selected sample.
    bg_cnt : int
        Number of successful events in the population.
    total_bg_cnt : int
        Total number of events in the population.

    Returns
    -------
    float
        Fold Change value.
    """
    return (selected_cnt / total_selected_cnt) / (bg_cnt / total_bg_cnt)

def moa_binomial_test(background_moa_cnt_df, selected_moa_cnt_df):
    """
    Perform a binomial test for each MoA (Mechanism of Action) relative to the background,
    and save the Fold Change values and p-values corrected using the Benjamini-Hochberg (BH) method.

    Parameters
    ----------
    background_moa_cnt_df : dataframe
        Background data containing two columns: MoA and count.
    selected_moa_cnt_df : dataframe
        Specific scenario data containing two columns: MoA and count.

    Returns
    -------
    dataframe
        Statistical results for each MoA, including Fold Change values and adjusted p-values.
    """
    cnt_info = pd.merge(selected_moa_cnt_df, background_moa_cnt_df, on="MoA", how="left")
    cnt_info.columns = ["MoA", "selected_cnt", "bg_cnt"]
    cnt_info.index = cnt_info["MoA"]

    # Replace NaN values for MoA not present in the background set with 1
    cnt_info = cnt_info.fillna(1)

    total_bg_cnt = np.sum(background_moa_cnt_df["count"])
    total_selected_cnt = np.sum(selected_moa_cnt_df["count"])

    p_values = []
    fold_changes = []
    for moa in cnt_info["MoA"]:
        bg_cnt = cnt_info.loc[moa, "bg_cnt"]
        selected_cnt = cnt_info.loc[moa, "selected_cnt"]

        # Calculate p-value using binomial test and Fold Change value
        p_value = binomial_test(k=selected_cnt, M=total_bg_cnt, n=bg_cnt, N=total_selected_cnt)
        fold_change = calculate_fold_change(selected_cnt, total_selected_cnt, bg_cnt, total_bg_cnt)

        p_values.append(p_value)
        fold_changes.append(fold_change)

    # Construct the result DataFrame and save Fold Change and corrected p-values
    binomial_res = cnt_info
    binomial_res["fold_change"] = fold_changes
    binomial_res["p_value"] = p_values

    # Filter using fold_change > 2.0 before p-value correction
    binomial_res = binomial_res.loc[binomial_res["fold_change"] > 2.0, :]
    if binomial_res.shape[0] == 0:
        return binomial_res
    binomial_res["adjusted_p_value"] = multipletests(binomial_res["p_value"], method="bonferroni")[1]

    return binomial_res

def generate_MoA_analysis_result(cell_line, cluster):
    """
    Generate MoA analysis results for a specific cell line and cluster.

    Parameters
    ----------
    cell_line : str
        Cell line identifier.
    cluster : int
        Cluster number.

    Returns
    -------
    dataframe
        Analysis results for the given cell line and cluster.
    """
    # Load drug information and establish background MoA counts
    drug_info_path = ROOT_DIR.parent / "data/400_drugs_MoA.xlsx"
    drug_info = pd.read_excel(drug_info_path, engine="openpyxl")
    drug_info = drug_info.loc[:, ["Name", "MoA"]]
    background_MoA_cnt_df = drug_info["MoA"].value_counts().reset_index()
    background_MoA_cnt_df.columns = ["MoA", "count"]

    cluster_label_file = DATA_DIR / f"result_2025/down_stream_analysis/cluster_label_res/{cell_line}.pkl"
    with open(cluster_label_file, "rb") as f:
        cluster_labels, _ = pickle.load(f)

    result_data = pd.read_csv(DATA_DIR / f"preprocessed_data_2025/down_stream_analysis/plans/{cell_line}.csv")
    cluster_data = result_data.loc[cluster_labels == cluster]
    drug_info = cluster_data.loc[:, ["first_drug", "first_moa"]]
    ccl_MoA_cnt_df = drug_info["first_moa"].value_counts().reset_index()
    ccl_MoA_cnt_df.columns = ["MoA", "count"]

    # Perform the binomial test
    ccl_test_res = moa_binomial_test(background_MoA_cnt_df, ccl_MoA_cnt_df)
    ccl_test_res = ccl_test_res.sort_values("fold_change", ascending=False)
    if ccl_test_res.shape[0] > 0:
        ccl_test_res = ccl_test_res.loc[ccl_test_res["adjusted_p_value"] < 0.05, :]

    return ccl_test_res

all_ccls = [sys.argv[1]]
for cell_line in all_ccls:
    res_dict = dict()
    for cluster in range(1, 8):
        res_dict[(cell_line, cluster)] = generate_MoA_analysis_result(cell_line, cluster)

    output_path = DATA_DIR / f"result_2025/down_stream_analysis/MoA_res/{cell_line}.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(res_dict, f)

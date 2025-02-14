import gym
import pandas as pd
from downstream_utils import *

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.resolve().parent
sys.path.append(str(ROOT_DIR))
from path import DATA_DIR 


all_ccls = [sys.argv[1]]

for cell_line in all_ccls:

    # Define output paths for storing results
    synergy_vector_path = DATA_DIR / f"result_2025/down_stream_analysis/synergy_vector_res/{cell_line}.pkl"
    synergy_vector_preprocessed_path = DATA_DIR / f"result_2025/down_stream_analysis/synergy_vector_preprocessed_res/{cell_line}.pkl"
    cluster_label_path = DATA_DIR / f"result_2025/down_stream_analysis/cluster_label_res/{cell_line}.pkl"
    max_step_path = DATA_DIR / f"result_2025/down_stream_analysis/max_step_res/{cell_line}.pkl"
    kay_pathway_path = DATA_DIR / f"result_2025/down_stream_analysis/key_pathway_res/{cell_line}.pkl"

    paths = [synergy_vector_path, synergy_vector_preprocessed_path, cluster_label_path, max_step_path, kay_pathway_path]

    # 检查并创建文件夹
    for path in paths:
        folder = path.parent  # 获取文件的父文件夹
        if not folder.exists():  # 检查文件夹是否存在
            folder.mkdir(parents=True, exist_ok=True)  # 创建文件夹（包括中间文件夹）
            print(f"Created folder: {folder}")
        else:
            print(f"Folder already exists: {folder}")


    # Step 1: Generate synergy vectors
    max_a1_step = 10  # Maximum number of steps for analysis

    total_res = pd.read_csv(DATA_DIR / f"preprocessed_data_2025/down_stream_analysis/plans/{cell_line}.csv")
    drug_combo_number = total_res.shape[0]  # Total number of drug combinations

    env_name = f"ENV_FDA_{cell_line}_STEP9"
    env = gym.make('gym_cell_model:ccl-env-cpd-v1', env_name=env_name)

    # Initialize matrices and lists to store analysis results
    drugA_synergy_mat = np.zeros([drug_combo_number, max_a1_step])
    drugB_synergy_mat = np.zeros([drug_combo_number, max_a1_step])
    drugA_synergy_mat_preprocessed = np.zeros([drug_combo_number, max_a1_step])
    drugB_synergy_mat_preprocessed = np.zeros([drug_combo_number, max_a1_step])
    drugA_cluster_labels = np.zeros(drug_combo_number, dtype=np.int16)
    drugB_cluster_labels = np.zeros(drug_combo_number, dtype=np.int16)
    drugA_max_steps = np.zeros(drug_combo_number, dtype=np.int16)
    drugB_max_steps = np.zeros(drug_combo_number, dtype=np.int16)
    match_key_pathway_res = []

    print(cell_line)
    for i in range(drug_combo_number):
        print(i, drug_combo_number)

        # Select the drug combination data for the current iteration
        selected_info = total_res.iloc[i, :]
        drugA_index = selected_info["first_ind"]

        # Simulate SDME (Synergy Determination for Multi-step Experiments)
        drugA_synergy_vec, drugB_synergy_vec = simulate_SDME(selected_info, env)
        drugA_synergy_mat[i, :] = drugA_synergy_vec
        drugB_synergy_mat[i, :] = drugB_synergy_vec

        # Preprocess the SDME vectors
        drugA_synergy_vec = synergy_data_preprocessing(drugA_synergy_vec)
        drugB_synergy_vec = synergy_data_preprocessing(drugB_synergy_vec)
        drugA_synergy_mat_preprocessed[i, :] = drugA_synergy_vec
        drugB_synergy_mat_preprocessed[i, :] = drugB_synergy_vec

        # Fit and classify the drugA SDME vector
        X = np.arange(10)
        Y = drugA_synergy_vec
        final_count, slope_arr, px, py = segments_fit(X, Y, 3)
        drugA_cluster_labels[i] = self_designed_cluster_func(slope_arr, thre=0.01)  # Classify using a custom clustering function
        drugA_max_steps[i] = np.argmax(drugA_synergy_vec) + 1  # Identify the maximum synergy step

        # Fit and classify the drugB SDME vector
        X = np.arange(10)
        Y = drugB_synergy_vec
        final_count, slope_arr, px, py = segments_fit(X, Y, 3)
        drugB_cluster_labels[i] = self_designed_cluster_func(slope_arr, thre=0.01)  # Classify using a custom clustering function
        drugB_max_steps[i] = np.argmax(drugB_synergy_vec) + 1  # Identify the maximum synergy step

        # Retrieve enriched pathway data for drugA
        file_name = DATA_DIR / f"preprocessed_data_2025/down_stream_analysis/enrich_res/{cell_line}_drugA_{drugA_index}.csv"
        enrich_pathway_df = pd.read_table(file_name, index_col=0)
        if drugA_cluster_labels[i] == 1:
            key_pathways = []
        else:
            key_pathways = match_key_pathway(drugA_synergy_vec, drugA_cluster_labels[i], enrich_pathway_df)
        match_key_pathway_res.append(key_pathways)

    # Save results to the output paths
    with open(synergy_vector_path, "wb") as f:
        pickle.dump([drugA_synergy_mat, drugB_synergy_mat], f)

    with open(synergy_vector_preprocessed_path, "wb") as f:
        pickle.dump([drugA_synergy_mat_preprocessed, drugB_synergy_mat_preprocessed], f)

    with open(cluster_label_path, "wb") as f:
        pickle.dump([drugA_cluster_labels, drugB_cluster_labels], f)

    with open(max_step_path, "wb") as f:
        pickle.dump([drugA_max_steps, drugB_max_steps], f)

    with open(kay_pathway_path, "wb") as f:
        pickle.dump(match_key_pathway_res, f)

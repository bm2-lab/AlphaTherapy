import gym
import pickle
import numpy as np
import pandas as pd
from scipy import optimize

import sys
from pathlib import Path

import gym
import pickle
import numpy as np
import pandas as pd
from scipy import optimize
import sys
from pathlib import Path

# Set the root directory and include it in the system path
ROOT_DIR = Path(__file__).parent.resolve().parent
sys.path.append(str(ROOT_DIR))
from path import DATA_DIR  # Import data directory from the path module


def cv_to_eff(cv_ls):
    """Convert a list of coefficients of variation (CV) to efficiencies."""
    cv_ls = np.array(cv_ls)
    return 1 - cv_ls[1:] / cv_ls[:-1]


def smooth(data):
    """Apply smoothing to the input 2D array."""
    n, m = data.shape
    smooth_data = np.zeros([n, m])
    smooth_data[:, 0] = (data[:, 0] + data[:, 1]) / 2
    for i in range(1, m - 1):
        smooth_data[:, i] = (data[:, i - 1] + data[:, i] + data[:, i + 1]) / 3
    smooth_data[:, m - 1] = (data[:, m - 2] + data[:, m - 1]) / 2
    return smooth_data


def synergy_data_preprocessing(synergy_vec):
    """Preprocess the synergy vector to normalize and smooth the data."""
    synergy_mat = synergy_vec.reshape([1, -1])
    synergy_mat = smooth(synergy_mat)
    min_vals = np.min(synergy_mat, axis=1, keepdims=True)
    max_vals = np.max(synergy_mat, axis=1, keepdims=True)
    synergy_mat = (synergy_mat - min_vals) / (max_vals - min_vals)
    return synergy_mat.reshape(-1)


def simulate_SDME(drug_combos_info, env):
    """
    Simulate the Sequential Drug Mechanism Experiment (SDME) for two drug combinations.

    Parameters:
        drug_combos_info (dict): Dictionary containing indices of the two drugs.
        env (gym.Env): OpenAI Gym environment.

    Returns:
        tuple: Synergy vectors for the first and second drugs.
    """
    max_a1_step = 10
    env.reset()
    max_step = max_a1_step * 2
    drug_number = env.action_space.n

    single_regimen_cv_mat = np.zeros([drug_number, max_step + 1])
    single_regimen_eff_mat = np.zeros([drug_number, max_step])

    for a in range(drug_number):
        env.reset()
        for s in range(max_step):
            env.step(a)
        single_regimen_cv_mat[a, :] = env.cv_ls
        single_regimen_eff_mat[a, :] = cv_to_eff(single_regimen_cv_mat[a, :])

    average_eff = np.mean(single_regimen_eff_mat, axis=0)
    drugA_index = drug_combos_info["first_ind"]
    drugB_index = drug_combos_info["second_ind"]

    drugB_increase_effs = []

    for drugA_step in range(1, max_a1_step + 1):
        for drugB_step in range(1, max_a1_step + 1):
            env.reset()
            for i in range(drugA_step):
                env.step(drugA_index)
            env.step(drugB_index)
            for j in range(drugB_step - 1):
                env.step(0)

            eff_arr = cv_to_eff(env.cv_ls)

            drugB_eff_fc_arr = np.zeros(drugB_step)
            for b in range(drugA_step, drugA_step + drugB_step):
                drugB_eff_fc_arr[b - drugA_step] = eff_arr[b] / average_eff[b]

            drugB_eff_fc = np.mean(drugB_eff_fc_arr)
            drugB_increase_effs.append(drugB_eff_fc)

    synergy_mat = np.array(drugB_increase_effs).reshape(max_a1_step, max_a1_step)
    first_drug_synergy_vec = np.mean(synergy_mat, axis=1)
    second_drug_synergy_vec = np.mean(synergy_mat, axis=0)

    return first_drug_synergy_vec, second_drug_synergy_vec


def simulate_SDME_mat(drug_combos_info, env):
    """
    Simulate SDME for a specific drug combination, returning the synergy matrix.

    Parameters:
        drug_combos_info (dict): Dictionary containing indices of the two drugs.
        env (gym.Env): OpenAI Gym environment.

    Returns:
        np.ndarray: Synergy matrix.
    """
    max_a1_step = 10
    max_step = max_a1_step * 2
    drug_number = env.action_space.n

    single_regimen_cv_mat = np.zeros([drug_number, max_step + 1])
    single_regimen_eff_mat = np.zeros([drug_number, max_step])

    for a in range(drug_number):
        env.reset()
        for s in range(max_step):
            env.step(a)
        single_regimen_cv_mat[a, :] = env.cv_ls
        single_regimen_eff_mat[a, :] = cv_to_eff(single_regimen_cv_mat[a, :])

    average_eff = np.mean(single_regimen_eff_mat, axis=0)
    drugA_index = drug_combos_info["first_ind"]
    drugB_index = drug_combos_info["second_ind"]

    drugB_increase_effs = []

    for drugA_step in range(1, max_a1_step + 1):
        for drugB_step in range(1, max_a1_step + 1):
            env.reset()
            for i in range(drugA_step):
                env.step(drugA_index)
            env.step(drugB_index)
            for j in range(drugB_step - 1):
                env.step(0)

            eff_arr = cv_to_eff(env.cv_ls)

            drugB_eff_fc_arr = np.zeros(drugB_step)
            for b in range(drugA_step, drugA_step + drugB_step):
                drugB_eff_fc_arr[b - drugA_step] = eff_arr[b] / average_eff[b]

            drugB_eff_fc = np.mean(drugB_eff_fc_arr)
            drugB_increase_effs.append(drugB_eff_fc)

    synergy_mat = np.array(drugB_increase_effs).reshape(max_a1_step, max_a1_step)
    return synergy_mat


def segments_fit(X, Y, maxcount):
    """
    Perform segmentation fitting to divide data into piecewise linear segments.

    Parameters:
        X (array-like): The x-coordinates of the data points.
        Y (array-like): The y-coordinates of the data points.
        maxcount (int): The maximum number of segments to consider.

    Returns:
        tuple: Contains the final number of segments, slope array, x-coordinates of segment boundaries (px), and y-coordinates of segment boundaries (py).
    """
    xmin = X.min()
    xmax = X.max()
    n = len(X)

    AIC_arr = np.zeros(maxcount)
    BIC_arr = np.zeros(maxcount)
    px_ls = []
    py_ls = []

    for count in range(1, maxcount + 1):
        # Initialize segment positions and segment means
        seg = np.full(count - 1, (xmax - xmin) / count)
        px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
        py_init = np.array([Y[np.abs(X - x) < (xmax - xmin) * 0.1].mean() for x in px_init])

        def func(p):
            """Update segment boundaries and means based on parameters."""
            seg = p[:count - 1]
            py = p[count - 1:]
            px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
            return px, py

        def err(p):
            """Calculate the mean squared error (RSS/n)."""
            px, py = func(p)
            Y2 = np.interp(X, px, py)
            return np.mean((Y - Y2) ** 2)

        # Optimize segment boundaries and means
        r = optimize.minimize(err, x0=np.r_[seg, py_init], method='Nelder-Mead')

        # Compute AIC and BIC
        AIC = 2 * np.log(err(r.x)) + 2 * count
        BIC = 2 * np.log(err(r.x)) + np.log(n) * count

        AIC_arr[count - 1] = AIC
        BIC_arr[count - 1] = BIC
        px_, py_ = func(r.x)
        px_ls.append(px_)
        py_ls.append(py_)

    ind = np.argmin(AIC_arr)
    px = px_ls[ind]
    py = py_ls[ind]

    final_count = ind + 1
    slope_arr = (py[1:] - py[:-1]) / (px[1:] - px[:-1])

    return final_count, slope_arr, px, py


def self_designed_cluster_func(slope_arr, thre):
    """
    Cluster data based on slope array and a given threshold.

    Parameters:
        slope_arr (array-like): Array of slopes between segments.
        thre (float): Threshold to classify slopes as significant.

    Returns:
        int: Cluster type based on slope patterns.
    """
    slope_arr[np.abs(slope_arr) <= thre] = 0.0

    if np.all(slope_arr == 0.0):
        return 1
    if np.all(slope_arr >= 0.0):
        return 2
    elif np.all(slope_arr <= 0.0):
        return 3

    if len(slope_arr) == 2:
        # Patterns for two segments
        if slope_arr[0] > 0 and slope_arr[1] < 0:
            return 4
        elif slope_arr[0] < 0 and slope_arr[1] > 0:
            return 5

    elif len(slope_arr) == 3:
        # Patterns for three segments
        if slope_arr[0] >= 0 and slope_arr[1] >= 0 and slope_arr[2] <= 0:
            return 4
        elif slope_arr[0] >= 0 and slope_arr[1] <= 0 and slope_arr[2] <= 0:
            return 4
        elif slope_arr[0] <= 0 and slope_arr[1] <= 0 and slope_arr[2] >= 0:
            return 5
        elif slope_arr[0] <= 0 and slope_arr[1] >= 0 and slope_arr[2] >= 0:
            return 5
        elif slope_arr[0] > 0 and slope_arr[1] < 0 and slope_arr[2] > 0:
            return 6
        elif slope_arr[0] < 0 and slope_arr[1] > 0 and slope_arr[2] < 0:
            return 7


def match_key_pathway(synergy_vector, cluster, enrich_pathway_df):
    # Increment cluster: find the peak, with an increasing trend on the left and decreasing trend on the right
    if cluster == 2:

        # (1) Find the peak index
        key_point = np.argmax(synergy_vector)
        # Add tolerance: the peak can be at key_point or one index to the left/right
        key_point_arrs = np.array([key_point - 1, key_point, key_point + 1])

        # (2) Match pathways with the same peak index
        max_ind = np.argmax(enrich_pathway_df.values, axis=1)
        match_pathway_df = enrich_pathway_df.loc[np.in1d(max_ind, key_point_arrs), :]

        # (3) Match pathways with the same trend (increasing then decreasing)
        slope_test = []
        for i in range(match_pathway_df.shape[0]):
            p_fc = match_pathway_df.iloc[i, :].values
            p_max_ind = np.argmax(p_fc)

            # If the peak index is 0, it is strictly decreasing
            if p_max_ind == 0:
                slope_test.append(False)
                continue

            # Fit the first segment (increasing)
            segment_data = p_fc[:p_max_ind + 1]
            x = np.arange(len(segment_data))
            slope1, _ = np.polyfit(x, segment_data, 1)

            # Fit the second segment (decreasing)
            if p_max_ind == 9:
                slope2 = 0.0  # Edge case for peak at the last index
            else:
                segment_data = p_fc[p_max_ind:]
                x = np.arange(len(segment_data))
                slope2, _ = np.polyfit(x, segment_data, 1)

            if slope1 >= 0 and slope2 <= 0:
                slope_test.append(True)
            else:
                slope_test.append(False)

        match_pathway_df = match_pathway_df.loc[slope_test]

    # Decrement cluster: find the peak, with an increasing trend on the left and decreasing trend on the right
    if cluster == 3:

        # (1) Find the peak index
        key_point = np.argmax(synergy_vector)
        # Add tolerance: the peak can be at key_point or one index to the left/right
        key_point_arrs = np.array([key_point - 1, key_point, key_point + 1])

        # (2) Match pathways with the same peak index
        max_ind = np.argmax(enrich_pathway_df.values, axis=1)
        match_pathway_df = enrich_pathway_df.loc[np.in1d(max_ind, key_point_arrs), :]

        # (3) Match pathways with the same trend (increasing then decreasing)
        slope_test = []
        for i in range(match_pathway_df.shape[0]):
            p_fc = match_pathway_df.iloc[i, :].values
            p_max_ind = np.argmax(p_fc)

            # If the peak index is 9, it is strictly increasing
            if p_max_ind == 9:
                slope_test.append(False)
                continue

            # Fit the first segment (increasing)
            if p_max_ind == 0:
                slope1 = 0.0  # Edge case for peak at the first index
            else:
                segment_data = p_fc[:p_max_ind + 1]
                x = np.arange(len(segment_data))
                slope1, _ = np.polyfit(x, segment_data, 1)

            # Fit the second segment (decreasing)
            segment_data = p_fc[p_max_ind:]
            x = np.arange(len(segment_data))
            slope2, _ = np.polyfit(x, segment_data, 1)

            if slope1 >= 0 and slope2 <= 0:
                slope_test.append(True)
            else:
                slope_test.append(False)

        match_pathway_df = match_pathway_df.loc[slope_test]

    # Increasing-decreasing cluster: find the peak, increasing on the left and decreasing on the right
    if cluster == 4:

        # (1) Find the peak index
        key_point = np.argmax(synergy_vector)
        # Add tolerance: the peak can be at key_point or one index to the left/right
        key_point_arrs = np.array([key_point - 1, key_point, key_point + 1])

        # (2) Match pathways with the same peak index
        max_ind = np.argmax(enrich_pathway_df.values, axis=1)
        match_pathway_df = enrich_pathway_df.loc[np.in1d(max_ind, key_point_arrs), :]

        # (3) Match pathways with the same trend (increasing then decreasing)
        slope_test = []
        for i in range(match_pathway_df.shape[0]):
            p_fc = match_pathway_df.iloc[i, :].values
            p_max_ind = np.argmax(p_fc)

            # Skip pathways where the peak is at the edges
            if p_max_ind == 0 or p_max_ind == 9:
                slope_test.append(False)
                continue

            # Fit the first segment (increasing)
            segment_data = p_fc[:p_max_ind + 1]
            x = np.arange(len(segment_data))
            slope1, _ = np.polyfit(x, segment_data, 1)

            # Fit the second segment (decreasing)
            segment_data = p_fc[p_max_ind:]
            x = np.arange(len(segment_data))
            slope2, _ = np.polyfit(x, segment_data, 1)

            if slope1 >= 0 and slope2 <= 0:
                slope_test.append(True)
            else:
                slope_test.append(False)

        match_pathway_df = match_pathway_df.loc[slope_test]

    # Decreasing-increasing cluster: find the trough, decreasing on the left and increasing on the right
    if cluster == 5:

        # (1) Find the trough index
        key_point = np.argmin(synergy_vector)
        # Add tolerance: the trough can be at key_point or one index to the left/right
        key_point_arrs = np.array([key_point - 1, key_point, key_point + 1])

        # (2) Match pathways with the same trough index
        min_ind = np.argmin(enrich_pathway_df.values, axis=1)
        match_pathway_df = enrich_pathway_df.loc[np.in1d(min_ind, key_point_arrs), :]

        # (3) Match pathways with the same trend (decreasing then increasing)
        slope_test = []
        for i in range(match_pathway_df.shape[0]):
            p_fc = match_pathway_df.iloc[i, :].values
            p_min_ind = np.argmin(p_fc)

            # Skip pathways where the trough is at the edges
            if p_min_ind == 0 or p_min_ind == 9:
                slope_test.append(False)
                continue

            # Fit the first segment (decreasing)
            segment_data = p_fc[:p_min_ind + 1]
            x = np.arange(len(segment_data))
            slope1, _ = np.polyfit(x, segment_data, 1)

            # Fit the second segment (increasing)
            segment_data = p_fc[p_min_ind:]
            x = np.arange(len(segment_data))
            slope2, _ = np.polyfit(x, segment_data, 1)

            if slope1 <= 0 and slope2 >= 0:
                slope_test.append(True)
            else:
                slope_test.append(False)

        match_pathway_df = match_pathway_df.loc[slope_test]

    # Complex clusters: increasing-decreasing-increasing or decreasing-increasing-decreasing
    if cluster == 6 or cluster == 7:
        # Identify key points
        final_count, slope_arr, px, py = segments_fit(np.arange(10), synergy_vector, 3)
        key_points = np.round(px[1:3])
        key_point_arr1 = np.array([key_points[0] - 1, key_points[0], key_points[0] + 1])
        key_point_arr2 = np.array([key_points[1] - 1, key_points[1], key_points[1] + 1])

        # Match pathways with the same number of segments and slope trends
        slope_test = []
        for i in range(enrich_pathway_df.shape[0]):
            final_count, p_slope_arr, p_px, p_py = segments_fit(np.arange(10), enrich_pathway_df.iloc[i, :], 3)
            if final_count == 3:
                x_ind = np.round(p_px[1:3])
                p_slope_arr[np.abs(p_slope_arr) < 0.01] = 0.0
                if np.in1d(x_ind[0], key_point_arr1)[0] and np.in1d(x_ind[1], key_point_arr2)[0]:
                    if p_slope_arr[0] * slope_arr[0] > 0 and p_slope_arr[1] * slope_arr[1] > 0 and p_slope_arr[2] * slope_arr[2] > 0:
                        slope_test.append(True)
                        continue
            slope_test.append(False)

        match_pathway_df = enrich_pathway_df.loc[slope_test]

    return match_pathway_df



def expre_ls_to_change_df(expression_arr):
    """
    Convert expression levels to change matrix and normalize.

    Parameters:
        expression_arr (array-like): Array of expression levels.

    Returns:
        DataFrame: Normalized expression change matrix.
    """
    weights = pd.read_table(
        ROOT_DIR / "../data/LINCS_weights.csv",
        sep="\t",
        header=0,
        index_col=0,
        low_memory=False
    )
    weights_mat = weights.T.values

    adjusted_order = pd.read_table(
        ROOT_DIR / '../data/adjusted_order.txt', 
        sep="\t", header=None, index_col=0
    ).iloc[:, 0].values

    with open(ROOT_DIR / '../data/long_expre_change_scaler.pkl', "rb") as f:
        long_expre_chang_scaler = pickle.load(f)

    expre_978_pre = expression_arr[:, adjusted_order]
    expre_979_pre = np.insert(expre_978_pre, 0, 1, axis=1)
    expre_11350_pre = np.matmul(expre_979_pre, weights_mat)
    expression_arr = np.concatenate([expre_978_pre, expre_11350_pre], axis=1)

    expression_change_mat = np.zeros([10, expression_arr.shape[1]])
    for i in range(1, expression_arr.shape[0]):
        expression_change_mat[i - 1, :] = expression_arr[i, :] - expression_arr[0, :]

    expression_change_mat = long_expre_chang_scaler.transform(expression_change_mat)
    expression_change_df = pd.DataFrame(expression_change_mat)

    weights_ind = list(weights.columns) + list(weights.index)
    weights_ind = [str(int(i)) for i in weights_ind[1:]]
    expression_change_df.columns = weights_ind

    return expression_change_df


def expression_simulate(cell_line, drugA_index, env):
    """
    Simulate expression changes for a given cell line and drug.

    Parameters:
        cell_line (str): Cell line identifier.
        drugA_index (int): Index of the drug to simulate.
        env: Environment object for simulation.

    Returns:
        None
    """
    max_a1_step = 10

    env.reset()
    for i in range(max_a1_step):
        env.step(drugA_index)
    
    drugA_expression_arr = np.concatenate(env.expression_ls)
    drugA_expression_change_df = expre_ls_to_change_df(drugA_expression_arr)

    outfile_path = DATA_DIR / f"preprocessed_data_2025/down_stream_analysis/expression_res/{cell_line}_drugA_{drugA_index}.csv"
    outfile_path.parent.mkdir(parents=True, exist_ok=True)
    drugA_expression_change_df.to_csv(outfile_path)

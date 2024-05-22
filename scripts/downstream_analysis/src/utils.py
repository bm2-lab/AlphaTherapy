import os
import gym
import pickle
import numpy as np
import pandas as pd
from scipy import optimize

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def expre_ls_to_change_df(expression_arr):
    # generate long expression change data
    weights = pd.read_table(
        project_dir + "/data/LINCS_weights.csv",
        sep="\t",
        header=0,
        index_col=0,
        low_memory=False)
    weights_mat = weights.T.values

    adjusted_order = pd.read_table(project_dir + '/data/adjusted_order.txt', sep="\t", header=None, index_col=0).iloc[:, 0].values

    with open(project_dir + "/data/long_expre_change_scaler.pkl", "rb") as f:
        long_expre_chang_scaler = pickle.load(f)

    expre_978_pre = expression_arr[:, adjusted_order]
    expre_979_pre = np.insert(expre_978_pre, 0, 1, axis=1)
    expre_11350_pre = np.matmul(expre_979_pre, weights_mat)
    expression_arr = np.concatenate([expre_978_pre, expre_11350_pre], axis=1)

    expression_change_mat = np.zeros([10, expression_arr.shape[1]])
    for i in range(1, expression_arr.shape[0]):
        expression_change_mat[i-1, :] = expression_arr[i, :] - expression_arr[0, :]

    expression_change_mat = long_expre_chang_scaler.transform(expression_change_mat)
    expression_change_df = pd.DataFrame(expression_change_mat)
    weights_ind = list(weights.columns) + list(weights.index)
    weights_ind = [str(int(i)) for i in weights_ind[1:]]
    expression_change_df.columns = weights_ind

    return expression_change_df


def expression_simulate(env_name, drugA_index, env):
    # Simulate the recommended sequential drug combinations in a specific cell line with a fixed drugB step, iterate over the drugA step's synergy reward, and save the analysis results to a CSV file.

    max_a1_step = 10

    env.reset()
    for i in range(max_a1_step):
        env.step(drugA_index)
    drugA_expression_arr = np.concatenate(env.expression_ls)
    drugA_expression_change_df = expre_ls_to_change_df(drugA_expression_arr)
    drugA_expression_change_df.to_csv(project_dir + "/scripts/downstream_analysis/working_log/%s_drugA_%d_expression.csv" % (env_name, drugA_index))


def cv_to_eff(cv_ls):

    return (1 - np.array(cv_ls[1:]) / np.array(cv_ls[0:-1]))


def smooth(data):
    n = data.shape[0]
    m = data.shape[1]
    smooth_data = np.zeros([n, m])
    smooth_data[:, 0] = (data[:, 0] + data[:, 1])/2
    for i in range(1, m-1):
        smooth_data[:, i] = (data[:, i] + data[:, i+1] + data[:, i-1])/3
    smooth_data[:, m-1] = (data[:, m-2] + data[:, m-1])/2
    return smooth_data


def synergy_data_preprocessing(synergy_vec):
    synergy_mat = synergy_vec.reshape([1, -1])
    synergy_mat = smooth(synergy_mat)
    min_vals = np.min(synergy_mat, axis=1, keepdims=True) 
    max_vals = np.max(synergy_mat, axis=1, keepdims=True) 
    synergy_mat = (synergy_mat - min_vals) / (max_vals - min_vals)
    return synergy_mat.reshape(-1)


def simulate_SDME(drug_combos_info, env):
    # generate SDE vectors
    max_a1_step = 10
    env.reset()

    max_step = max_a1_step+max_a1_step
    drug_number = env.action_space.n
    single_regimen_cv_mat = np.zeros([drug_number, max_step+1])
    single_regimen_eff_mat = np.zeros([single_regimen_cv_mat.shape[0], max_step])

    for a in range(drug_number):
        env.reset()
        for s in range(max_step):
            env.step(a)
        single_regimen_cv_mat[a, :] = env.cv_ls
        single_regimen_eff_mat[a, :] = cv_to_eff(single_regimen_cv_mat[a, :])

    average_eff = np.mean(single_regimen_eff_mat, axis=0) # shape: max_step, average_eff[i] means the eff of step(i+1)

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
            for b in range(drugA_step, drugA_step+drugB_step):
                drugB_eff_fc_arr[b-(drugA_step)] = eff_arr[b] / average_eff[b]

                # drugB_eff_fc_arr[b-(drugA_step)] = eff_arr[b] / single_regimen_eff_mat[drugA_index, b]
            drugB_eff_fc = np.mean(drugB_eff_fc_arr)
            drugB_increase_effs.append(drugB_eff_fc)

    synergy_mat = np.array(drugB_increase_effs).reshape(max_a1_step, max_a1_step)

    first_drug_synergy_vec = np.mean(synergy_mat, axis=1)
    second_drug_synergy_vec = np.mean(synergy_mat, axis=0)

    return first_drug_synergy_vec, second_drug_synergy_vec


def segments_fit(X, Y, maxcount):
    # fit SDE vector
    xmin = X.min()
    xmax = X.max()
    
    n = len(X)
    
    AIC_arr = np.zeros(maxcount)
    BIC_arr = np.zeros(maxcount)
    px_ls = []
    py_ls = []
    
    for count in range(1, maxcount+1):
        
        seg = np.full(count - 1, (xmax - xmin) / count)

        px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
        py_init = np.array([Y[np.abs(X - x) < (xmax - xmin) * 0.1].mean() for x in px_init])

        def func(p):
            seg = p[:count - 1]
            py = p[count - 1:]
            px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
            return px, py

        def err(p): # This is RSS / n
            px, py = func(p)
            Y2 = np.interp(X, px, py)
            return np.mean((Y - Y2)**2)

        r = optimize.minimize(err, x0=np.r_[seg, py_init], method='Nelder-Mead')
    
        # Compute AIC/ BIC. 
        AIC = 2 * np.log(err(r.x)) + 2 * count
        BIC = 2 * np.log(err(r.x)) + np.log(n) * count

        AIC_arr[count-1] = AIC
        BIC_arr[count-1] = BIC
        px_, py_ = func(r.x)
        px_ls.append(px_)
        py_ls.append(py_)

    ind = np.argmin(AIC_arr)
    px = px_ls[ind]
    py = py_ls[ind]
    
    final_count = ind + 1
    slope_arr = (py[1:] - py[0:-1])/(px[1:] - px[0:-1])
        
    return final_count, slope_arr, px, py ## Return the last (n-1)


def self_designed_cluster_func(slope_arr, thre):
    # classify SDE vector to 7 different trend classes

    slope_arr[np.abs(slope_arr) <= thre] = 0.0

    if np.all(slope_arr == 0.0):
        return 1
    if np.all(slope_arr >= 0.0):
        return 2
    elif np.all(slope_arr <= 0.0):
        return 3

    if len(slope_arr) == 2:
        
        if slope_arr[0]>0 and slope_arr[1]<0:
            return 4
        elif slope_arr[0]<0 and slope_arr[1]>0:
            return 5
    
    elif len(slope_arr) == 3:

        if slope_arr[0]>=0 and slope_arr[1]>=0 and slope_arr[2]<=0:
            return 4
        elif slope_arr[0]>=0 and slope_arr[1]<=0 and slope_arr[2]<=0:
            return 4
        elif slope_arr[0]<=0 and slope_arr[1]<=0 and slope_arr[2]>=0:
            return 5
        elif slope_arr[0]<=0 and slope_arr[1]>=0 and slope_arr[2]>=0:
            return 5
        elif slope_arr[0]>0 and slope_arr[1]<0 and slope_arr[2]>0:
            return 6
        elif slope_arr[0]<0 and slope_arr[1]>0 and slope_arr[2]<0:
            return 7


def match_key_pathway(synergy_vector, cluster, enrich_pathway_df):
    # match the SDE vector with the pathway vectors
    if cluster == 2:

        key_point = np.argmax(synergy_vector)
        key_point_arrs = np.array([key_point-1, key_point, key_point+1])

        max_ind = np.argmax(enrich_pathway_df.values, axis=1)
        match_pathway_df = enrich_pathway_df.loc[np.in1d(max_ind, key_point_arrs),:]


        slope_test = []
        for i in range(match_pathway_df.shape[0]):
            p_fc = match_pathway_df.iloc[i, :].values
            p_max_ind = np.argmax(p_fc)
            
            if p_max_ind == 0:
                slope_test.append(False)
                continue
            
            segment_data = p_fc[:p_max_ind+1]
            x = np.arange(len(segment_data))
            slope1, _ = np.polyfit(x, segment_data, 1)

            if p_max_ind == 9:
                slope2 = 0.0
            else:
                segment_data = p_fc[p_max_ind:]
                x = np.arange(len(segment_data))
                slope2, _ = np.polyfit(x, segment_data, 1)

            if slope1 >= 0 and slope2 <= 0:
                slope_test.append(True)
            else:
                slope_test.append(False)
        match_pathway_df = match_pathway_df.loc[slope_test]

    if cluster == 3:
        
        key_point = np.argmax(synergy_vector)
        key_point_arrs = np.array([key_point-1, key_point, key_point+1])

        max_ind = np.argmax(enrich_pathway_df.values, axis=1)
        match_pathway_df = enrich_pathway_df.loc[np.in1d(max_ind, key_point_arrs), :]



        slope_test = []
        for i in range(match_pathway_df.shape[0]):
            p_fc = match_pathway_df.iloc[i, :].values
            p_max_ind = np.argmax(p_fc)
            
            if p_max_ind == 9:
                slope_test.append(False)
                continue

            if p_max_ind == 0:
                slope1 = 0.0
            else:
                segment_data = p_fc[:p_max_ind+1]
                x = np.arange(len(segment_data))
                slope1, _ = np.polyfit(x, segment_data, 1)

            segment_data = p_fc[p_max_ind:]
            x = np.arange(len(segment_data))
            slope2, _ = np.polyfit(x, segment_data, 1)

            if slope1 >= 0 and slope2 <= 0:
                slope_test.append(True)
            else:
                slope_test.append(False)
        
        match_pathway_df = match_pathway_df.loc[slope_test]

    if cluster == 4:

        key_point = np.argmax(synergy_vector)
        key_point_arrs = np.array([key_point-1, key_point, key_point+1])

        max_ind = np.argmax(enrich_pathway_df.values, axis=1)
        match_pathway_df = enrich_pathway_df.loc[np.in1d(max_ind, key_point_arrs), :]


        slope_test = []
        for i in range(match_pathway_df.shape[0]):
            p_fc = match_pathway_df.iloc[i, :].values
            p_max_ind = np.argmax(p_fc)
            
            if p_max_ind == 0 or p_max_ind == 9:
                slope_test.append(False)
                continue

            segment_data = p_fc[:p_max_ind+1]
            x = np.arange(len(segment_data))
            slope1, _ = np.polyfit(x, segment_data, 1)

            segment_data = p_fc[p_max_ind:]
            x = np.arange(len(segment_data))
            slope2, _ = np.polyfit(x, segment_data, 1)

            if slope1 >= 0 and slope2 <= 0:
                slope_test.append(True)
            else:
                slope_test.append(False)
        
        match_pathway_df = match_pathway_df.loc[slope_test]

    if cluster == 5:

        key_point = np.argmin(synergy_vector)
        key_point_arrs = np.array([key_point-1, key_point, key_point+1])

        min_ind = np.argmin(enrich_pathway_df.values, axis=1)
        match_pathway_df = enrich_pathway_df.loc[np.in1d(min_ind, key_point_arrs), :]

        slope_test = []
        for i in range(match_pathway_df.shape[0]):
            p_fc = match_pathway_df.iloc[i, :].values
            p_min_ind = np.argmin(p_fc)
            
            if p_min_ind == 0 or p_min_ind == 9:
                slope_test.append(False)
                continue
            
            segment_data = p_fc[:p_min_ind+1]
            x = np.arange(len(segment_data))
            slope1, _ = np.polyfit(x, segment_data, 1)

            segment_data = p_fc[p_min_ind:]
            x = np.arange(len(segment_data))
            slope2, _ = np.polyfit(x, segment_data, 1)

            if slope1 <= 0 and slope2 >= 0:
                slope_test.append(True)
            else:
                slope_test.append(False)
        
        match_pathway_df = match_pathway_df.loc[slope_test]

    if cluster == 6 or cluster == 7:
        final_count, slope_arr, px, py = segments_fit(np.arange(10), synergy_vector, 3)
        key_points = np.round(px[1:3])
        key_point_arr1 = np.array([key_points[0]-1, key_points[0], key_points[0]+1])
        key_point_arr2 = np.array([key_points[1]-1, key_points[1], key_points[1]+1])

        slope_test = []
        for i in range(enrich_pathway_df.shape[0]):
            final_count, p_slope_arr, p_px, p_py = segments_fit(np.arange(10), enrich_pathway_df.iloc[0, :], 3)
            if final_count == 3:
                x_ind = np.round(p_px[1:3])
                p_slope_arr[np.abs(p_slope_arr) < 0.01] = 0.0
                if np.in1d(x_ind[0], key_point_arr1)[0] and np.in1d(x_ind[1], key_point_arr2)[0]:
                    if p_slope_arr[0]*slope_arr[0] > 0 and p_slope_arr[1]*slope_arr[1] > 0 and p_slope_arr[2]*slope_arr[2] > 0:
                        slope_test.append(True)
                        continue
            slope_test.append(False)

        match_pathway_df = enrich_pathway_df.loc[slope_test]

    return match_pathway_df

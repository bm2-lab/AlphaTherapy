import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.resolve().parent.resolve().parent
sys.path.append(str(ROOT_DIR))
from path import DATA_DIR  

print("ROOT_DIR:", ROOT_DIR)
print("DATA_DIR:", DATA_DIR)

# Change working directory to data input path
os.chdir(DATA_DIR / "preprocessed_data_2025" / "state_transition_model_input_data")

#####################################################################
# Step 1: Load data
#####################################################################

# Set random seed for reproducibility
np.random.seed(111)

# Load preprocessed dataset from step 1
file_name = "step1_LINCS_model_dataset.pkl"
with open(file_name, "rb") as f:
    drug, x1, x2, ccl, _, _, drug_names, _ = pickle.load(f)

# Define x (control expression) and y (perturbation-induced expression change)
x = x1
y = x2 - x1

#####################################################################
# Step 2: Filter data based on sample perturbation degree (sig_ss)
#####################################################################

def signature_strength(mat: np.ndarray, low_thre: np.ndarray, high_thre: np.ndarray) -> list:
    return [
        np.sum((mat[i, :] < low_thre) | (mat[i, :] > high_thre))
        for i in range(len(mat))
    ]

low_thre = np.percentile(y, 10, axis=0)
high_thre = np.percentile(y, 90, axis=0)
instances_ss = []
resorted_ind = []

count = 0
total_number = len(np.unique(drug_names)) * len(np.unique(ccl))
for drug_i in np.unique(drug_names):
    for ccl_j in np.unique(ccl):
        print(total_number, count)
        count += 1
        if np.sum(np.in1d(drug_names, [drug_i]) & np.in1d(ccl, [ccl_j])) > 0:   # 如果存在该细胞系和药物的实验数据
            ind = np.arange(len(drug_names))[np.in1d(drug_names, [drug_i])  # 找到instances的index
                                             & np.in1d(ccl, [ccl_j])]
            ss = signature_strength(y[ind, :], low_thre, high_thre)
            instances_ss.append(ss)
            resorted_ind.append(ind)

resorted_ind = np.array(resorted_ind, dtype=object)
ss_ = np.array([sum(item) / (len(item) * 978) for item in instances_ss])    # 计算(drug cell)平均的signature strength

# Filter samples with perturbation scores above threshold
ss_thre = 0.1
filter_ind = np.concatenate(resorted_ind[ss_ >= ss_thre])

# Apply filter to dataset
drug = drug[filter_ind]
x = x[filter_ind]
y = y[filter_ind]
ccl = ccl[filter_ind]
drug_names = drug_names[filter_ind]

# Normalize features
ss = StandardScaler()
x = ss.fit_transform(x)
ss2 = StandardScaler()
y = ss2.fit_transform(y)

#####################################################################
# Step 3: Split data into training, validation, and test sets
#####################################################################

# Define split ratios (80% training, 10% validation, 10% testing)
sample_number = drug_names.shape[0]
init_one_fold_number = int(sample_number * 0.1)

unique_drug_names = np.unique(drug_names)
np.random.shuffle(unique_drug_names)

one_fold_number = init_one_fold_number
one_fold_ind_ls = []
k_folds_ind_ls = []

for d in unique_drug_names:
    temp = np.where([drug_names == d])[1]
    
    one_fold_number -= len(temp)

    if one_fold_number > 0:
        one_fold_ind_ls.extend(temp)
    else:
        k_folds_ind_ls.append(one_fold_ind_ls)
        one_fold_number = init_one_fold_number
        one_fold_ind_ls = []
        one_fold_number -= len(temp)
        one_fold_ind_ls.extend(temp)

k_folds_ind_arr = np.array(k_folds_ind_ls, dtype=object)

for i in range(10):

    test_fold_ind = i
    train_fold_ind = np.setdiff1d(np.arange(10), [test_fold_ind])

    test_ind = k_folds_ind_arr[test_fold_ind]
    train_ind = np.concatenate(k_folds_ind_arr[train_fold_ind])

    # all ccls are in both train ccls and test ccls
    # train_ccls = np.unique(ccl[train_ind])
    # test_ccls = np.unique(ccl[test_ind])
    # not_seen_ccls = test_ccls[~np.in1d(test_ccls, train_ccls)]
    # print(not_seen_ccls)

    valid_ind = np.random.choice(train_ind, size=one_fold_number, replace=False)
    train_ind = np.setdiff1d(train_ind, valid_ind)

    # Separate data into training, validation, and test sets
    train_drug, train_x, train_y = drug[train_ind], x[train_ind], y[train_ind]
    valid_drug, valid_x, valid_y = drug[valid_ind], x[valid_ind], y[valid_ind]
    test_drug, test_x, test_y = drug[test_ind], x[test_ind], y[test_ind]

    # Save split data into pickle file
    out_file = "step2_model_training_dataset_chemical_blind_%d.pkl" % i
    with open(out_file, "wb") as f:
        pickle.dump([train_drug, train_x, train_y, valid_drug, valid_x, valid_y, test_drug, test_x, test_y, ss, ss2, y], f)

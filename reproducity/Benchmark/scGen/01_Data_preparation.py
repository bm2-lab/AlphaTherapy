import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.resolve().parent.resolve().parent
sys.path.insert(0, str(ROOT_DIR))
from path import DATA_DIR  


import os
import pickle
import numpy as np
import scanpy as sc
import pandas as pd
from sklearn.preprocessing import StandardScaler

np.random.seed(111)

# 1. read data
file_name = DATA_DIR / "preprocessed_data_2025/state_transition_model_input_data/step1_LINCS_model_dataset.pkl"
with open(file_name, "rb") as f:
    drug_smiles, x1, x2, cell_id, concentraion, duration, drug_name, rna_plate = pickle.load(f)
x = x1  
y = x2 - x1

# We have done this process before which is also time-costing, so we just load the result here.
intermediate_data_path = DATA_DIR / "preprocessed_data_2025" / "state_transition_model_input_data" / "step2_intermediate_data.pkl"
with open(intermediate_data_path, "rb") as f:
    resorted_ind, ss_ = pickle.load(f)
ss_thre = 0.1
filter_ind = np.concatenate(resorted_ind[ss_ >= ss_thre])

drug_smiles = drug_smiles[filter_ind]
x = x[filter_ind] 
y = y[filter_ind] 
x1 = x1[filter_ind] 
x2 = x2[filter_ind] 
cell_id = cell_id[filter_ind]
concentraion = concentraion[filter_ind]
duration = duration[filter_ind]
drug_name = drug_name[filter_ind]
rna_plate = rna_plate[filter_ind]

# normalization
ss = StandardScaler()
x = ss.fit_transform(x) 
ss2 = StandardScaler()
y = ss2.fit_transform(y)  
ss3 = StandardScaler()
x1 = ss3.fit_transform(x1) 
ss4 = StandardScaler()
x2 = ss4.fit_transform(x2) 

# Data split
sample_number = cell_id.shape[0]
init_one_fold_number = int(sample_number * 0.1)

unique_ccl_names = np.unique(cell_id)
np.random.shuffle(unique_ccl_names)

one_fold_number = init_one_fold_number
one_fold_ind_ls = []
k_folds_ind_ls = []

for c in unique_ccl_names:
    temp = np.where([cell_id == c])[1]
    
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

i = 0
test_fold_ind = i
train_fold_ind = np.setdiff1d(np.arange(10), [test_fold_ind])

test_ind = k_folds_ind_arr[test_fold_ind]
train_ind = np.concatenate(k_folds_ind_arr[train_fold_ind])

train_drugs = np.unique(drug_name[train_ind])
test_drugs = np.unique(drug_name[test_ind])
not_seen_drugs = test_drugs[~np.in1d(test_drugs, train_drugs)]
not_seen_drugs_index = np.where(np.in1d(drug_name, not_seen_drugs))[0]
test_ind = np.setdiff1d(test_ind, not_seen_drugs_index)
print(i, test_ind.shape)

valid_ind = np.random.choice(train_ind, size=one_fold_number, replace=False)
train_ind = np.setdiff1d(train_ind, valid_ind)


# 2. Create anndata
split_ind_arr = split_ind_arr = np.array(["invalid_data"]*(len(x1)))
split_ind_arr[train_ind] = "train"
split_ind_arr[valid_ind] = "valid"
split_ind_arr[test_ind] = "test"

obs1 = pd.DataFrame({
    'cell_type': cell_id,
    'condition': drug_name,
    'AlphaTherapy_split': split_ind_arr
})
adata = sc.AnnData(X=x2, obs=obs1)

adata.layers["control_expression"] =  x1

output_path = ROOT_DIR / "Benchmark/scGen/datasets/lincs_dataset.h5ad"
output_path.parent.mkdir(parents=True, exist_ok=True)
adata.write(output_path)




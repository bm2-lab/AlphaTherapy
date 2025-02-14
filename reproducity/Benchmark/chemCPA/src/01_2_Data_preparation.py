# Dataset for cell blind scenario for chemCPA

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.resolve().parent.resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from path import DATA_DIR  

import os
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
from rdkit import Chem
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
ss4 = StandardScaler()
x2 = ss4.fit_transform(x2) 

# data split
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

# Remove test set samples that have been unseen by both drug and CCL
train_drugs = np.unique(drug_name[train_ind])
test_drugs = np.unique(drug_name[test_ind])
not_seen_drugs = test_drugs[~np.in1d(test_drugs, train_drugs)]
not_seen_drugs_index = np.where(np.in1d(drug_name, not_seen_drugs))[0]
test_ind = np.setdiff1d(test_ind, not_seen_drugs_index)

valid_ind = np.random.choice(train_ind, size=one_fold_number, replace=False)
train_ind = np.setdiff1d(train_ind, valid_ind)

split_arr = np.array(sample_number*["train"])
split_arr[train_ind] = "train"
split_arr[valid_ind] = "test"
split_arr[test_ind] = "ood"
split_arr[not_seen_drugs_index] = "invalid_index"

control_train_data = x[split_arr=="train", :]
control_train_data, control_train_index = np.unique(control_train_data, axis=0, return_index=True)
control_train_cell_id = cell_id[control_train_index]

control_test_data = x[split_arr=="test", :]
control_test_data, control_test_index = np.unique(control_test_data, axis=0, return_index=True)
control_test_cell_id = cell_id[control_test_index]

# 2. create AnnData object

# treatment
obs1 = pd.DataFrame({
    'cell_type': cell_id,
    'condition': drug_name,
    'dose_val': [1.0]*len(cell_id),
    'pert_id': drug_name,
})

# train control
obs2 = pd.DataFrame({
    'cell_type': control_train_cell_id,
    'condition': ["DMSO"]*len(control_train_cell_id),
    'dose_val': [1.0]*len(control_train_cell_id),
    'pert_id': ["DMSO"]*len(control_train_cell_id),
})

# test control
obs3 = pd.DataFrame({
    'cell_type': control_test_cell_id,
    'condition': ["DMSO"]*len(control_test_cell_id),
    'dose_val': [1.0]*len(control_test_cell_id),
    'pert_id': ["DMSO"]*len(control_test_cell_id),
})

X_combined = np.concatenate([x2, control_train_data, control_test_data], axis=0) 
obs_combined = pd.concat([obs1, obs2, obs3], axis=0).reset_index(drop=True)  

adata = sc.AnnData(X=X_combined, obs=obs_combined)

X_layer_combined = np.concatenate([x, control_train_data, control_test_data], axis=0) 
adata.layers["control_gene_expression"] = X_layer_combined

split_info = list(split_arr) + ["train"]*control_train_data.shape[0] + ["test"]*control_test_data.shape[0]
adata.obs["AlphaTherapy_split"] = split_info
adata.obs["AlphaTherapy_split"].value_counts()

del x, x1, x2, y

# data preprocessing following the original chemCPA

import re

def remove_non_alphanumeric(input_string):
    return re.sub(r'[^a-zA-Z0-9]', '', input_string)

adata.obs['condition'] = adata.obs['condition'].apply(remove_non_alphanumeric)

adata.obs['cov_drug_dose_name'] = adata.obs.cell_type.astype(str) + '_' + adata.obs.condition.astype(str) + '_' + adata.obs.dose_val.astype(str)
adata.obs['cov_drug_name'] = adata.obs.cell_type.astype(str) + '_' + adata.obs.condition.astype(str)
adata.obs['eval_category'] = adata.obs['cov_drug_dose_name']
adata.obs['control'] = (adata.obs['condition'] == 'DMSO').astype(int)

# Differential expression analysis
de_genes = {}
de_genes_quick = {}

adata_df = adata.to_df()
adata_df['condition'] = adata.obs.condition
numeric_cols = adata_df.select_dtypes(include=np.number).columns 
dmso = adata_df[adata_df.condition == "DMSO"][numeric_cols].mean()

for cond, df in adata_df.groupby('condition'): 
    if cond != 'DMSO':
        drug_mean = df[numeric_cols].mean()
        de_50_idx = np.argsort(abs(drug_mean-dmso))[-50:]
        de_genes_quick[cond] = drug_mean.index[de_50_idx].values

de_genes = de_genes_quick

def extract_drug(cond): 
    split = cond.split('_')
    return split[1]
temp = {cat: de_genes_quick[extract_drug(cat)] for cat in adata.obs.eval_category.unique() if extract_drug(cat) != 'DMSO'}
adata.uns['rank_genes_groups_cov'] = temp

# Compound information
adata.obs["pert_iname"] = adata.obs["pert_id"]
pert_iname_unique = pd.Series(np.unique(adata.obs.pert_iname))
print(f"# of unique perturbations: {len(pert_iname_unique)}")

# get smiles from LINCS
reference_df = pd.read_csv(DATA_DIR / "raw_data/LINCS_GSE92742/GSE92742_Broad_LINCS_pert_info.txt", delimiter = "\t")
reference_df = reference_df.loc[reference_df.pert_iname.isin(pert_iname_unique), ['pert_iname', 'canonical_smiles']]
reference_df = reference_df.drop_duplicates("pert_iname")

cond = ~pert_iname_unique.isin(reference_df.pert_iname)
print(f"From {len(pert_iname_unique)} total drugs, {cond.sum()} were not part of the reference dataframe.")

adata.obs = adata.obs.reset_index().merge(reference_df, how="left").set_index('index')
adata.obs.loc[adata.obs["pert_iname"] == "DMSO", "canonical_smiles"] = "CS(=O)C" 
adata.obs.loc[:, 'canonical_smiles'] = adata.obs.canonical_smiles.astype('str')
invalid_smiles = adata.obs.canonical_smiles.isin(['-666', 'restricted', 'nan'])
print(f'Among {len(adata)} observations, {100*invalid_smiles.sum()/len(adata):.2f}% ({invalid_smiles.sum()}) have an invalid SMILES string')
adata = adata[~invalid_smiles]

# canonicalize_smiles
def canonicalize_smiles(smiles):
    if smiles:
        return Chem.CanonSmiles(smiles)
    else:
        return None

adata.obs["canonical_smiles"] = [canonicalize_smiles(s) for s in adata.obs["canonical_smiles"]]

def check_smiles(smiles):
    m = Chem.MolFromSmiles(smiles,sanitize=False)
    if m is None:
        print('invalid SMILES')
        return False
    else:
        try:
            Chem.SanitizeMol(m)
        except:
            print('invalid chemistry')
            return False
    return True

def remove_invalid_smiles(dataframe, smiles_key: str = 'SMILES', return_condition: bool = False):
    unique_drugs = pd.Series(np.unique(dataframe[smiles_key]))
    valid_drugs = unique_drugs.apply(check_smiles)
    print(f"A total of {(~valid_drugs).sum()} have invalid SMILES strings")
    _validation_map = dict(zip(unique_drugs, valid_drugs))
    cond = dataframe[smiles_key].apply(lambda x: _validation_map[x])
    if return_condition: 
        return cond
    dataframe = dataframe[cond].copy()
    return dataframe

cond = remove_invalid_smiles(adata.obs, smiles_key='canonical_smiles', return_condition=True)
adata = adata[cond]

# 3. save data

adata_out = DATA_DIR / "preprocessed_data_2025/state_transition_model_benchmark/chemCPA/cell_blind.h5ad"  
adata.write(adata_out)







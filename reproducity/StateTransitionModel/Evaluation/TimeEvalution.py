import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Define the root directory of the project
ROOT_DIR = Path(os.getcwd()).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from path import DATA_DIR  

import os
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys

def intermediate_data():
    inst_info = pd.read_table(DATA_DIR / 'raw_data/LINCS_GSE92742/GSE92742_Broad_LINCS_inst_info.txt',
                            sep='\t',
                            header=0,
                            index_col=None,
                            low_memory=False)
    inst_info_control = inst_info[inst_info["pert_id"] == "DMSO"]
    inst_info_trt = inst_info[inst_info["pert_id"] != "DMSO"]

    instance_rna_plates = np.unique(np.array(inst_info_trt["rna_plate"]))
    inst_info_trt_new = pd.DataFrame()

    for rna_plate in instance_rna_plates:
        inst_info_control_plate = inst_info_control[inst_info_control["rna_plate"] == rna_plate]
        if inst_info_control_plate.empty:
            continue
        inst_info_trt_plate = inst_info_trt[inst_info_trt["rna_plate"] == rna_plate]
        inst_info_trt_plate.loc[:, "ctl_inst_id"] = np.random.choice(
            inst_info_control_plate.loc[:, "inst_id"], len(inst_info_trt_plate)) 
        inst_info_trt_new = pd.concat((inst_info_trt_new.loc[:, :],
                                    inst_info_trt_plate.loc[:, :])) 
        
    pert_info = pd.read_table(DATA_DIR / 'raw_data/LINCS_GSE92742/GSE92742_Broad_LINCS_pert_info.txt',
                            sep="\t",
                            header=0,
                            index_col=None)
    pert_info.index = pert_info["pert_id"].values
    inst_info_trt_new = pd.merge(inst_info_trt_new, pert_info, on="pert_id")

    drug_table = []
    drugs_pert_id = np.unique(np.array(inst_info_trt_new["pert_id"]))
    count = 0
    del_drug_name = []
    for drug_pert_id in drugs_pert_id:
        canonical_smiles = pert_info.loc[drug_pert_id, "canonical_smiles"]
        if type(canonical_smiles) == type(np.nan) or canonical_smiles == "-666" or canonical_smiles == "restricted":
            del_drug_name.append(drug_pert_id)
            continue
        try:
            m = Chem.MolFromSmiles(canonical_smiles)
            MACCS_str = MACCSkeys.GenMACCSKeys(m).ToBitString()
            MACCS_code = np.array([int(i) for i in list(MACCS_str)])
        except:
            del_drug_name.append(drug_pert_id)
            continue
        drug_fingerprint = MACCS_code[1:]
        drug_name = pert_info.loc[drug_pert_id, "pert_iname"]
        drug_table.append([count, drug_pert_id, drug_name, drug_fingerprint])
        count += 1
        print(count, "drugs have(s) been convert to fingerprint!")

    drug_df = pd.DataFrame(drug_table)
    drug_df.columns = ["index", "pert_id", "pert_iname", "pert_code"]
    drug_df.index = drug_df["pert_id"]
    inst_info_trt_new = inst_info_trt_new[np.in1d(inst_info_trt_new["pert_id"],
                                                np.unique(drug_df["pert_id"]))]   # 根据具有fingerprint的cpd来筛选instances
    return inst_info_trt_new, drug_df
inst_info_trt_new, drug_df = intermediate_data()


def get_pert_time_ls(x):
    times = np.unique(x["pert_time"])
    return times

inst_info = pd.read_table(DATA_DIR / 'raw_data/LINCS_GSE92742/GSE92742_Broad_LINCS_inst_info.txt',
                          sep='\t',
                          header=0,
                          index_col=None,
                          low_memory=False)
inst_info = inst_info[inst_info["pert_id"] != "DMSO"]
inst_info = inst_info[inst_info["pert_type"] == "trt_cp"]
inst_info = inst_info[inst_info["pert_dose_unit"] != "-666"]
inst_info.loc[inst_info["pert_dose_unit"]=="mm", "pert_dose"] = inst_info.loc[inst_info["pert_dose_unit"]=="mm", "pert_dose"]/1000

res_time = inst_info.groupby(["pert_id", "pert_dose", "cell_id"]).apply(get_pert_time_ls)

temp = [tuple(i[:]) for i in res_time.values]    
temp = np.array(temp, dtype=object)
key, val = np.unique(temp, return_counts=True)
expmt_stats = pd.DataFrame({"Time(h)":key, "experiment_number":val})
print(expmt_stats)


def contain_more_than_24h(x):
    if (24 in x) and (48 in x):
        return True
    else:
        return False
res_time_boolen = res_time.apply(contain_more_than_24h)
filter_24h_df = pd.DataFrame({"pert_id": res_time_boolen.index.get_level_values('pert_id'), "pert_dose": res_time_boolen.index.get_level_values('pert_dose'), "cell_id": res_time_boolen.index.get_level_values('cell_id'), "pert_time_bool": res_time_boolen.values, "pert_time": res_time.values})
filter_24h_df = filter_24h_df.loc[filter_24h_df["pert_time_bool"]]

selected_inst_df = filter_24h_df

match_data = pd.merge(selected_inst_df, inst_info_trt_new, on=["pert_id", "pert_dose", "cell_id"])
match_data = match_data[match_data["pert_dose"]==5.0]
match_data_48h = match_data[match_data["pert_time_y"]==48]
match_data_24h = match_data[match_data["pert_time_y"]==24]


from cmapPy.pandasGEXpress.parse import parse

def get_data_from_instance_ids(instance_id_df):

    gene_table = pd.read_table(
        ROOT_DIR / "../data/cmap_978_genes_order.txt",
                            sep='\t',
                            header=None)
    genes = [str(i) for i in gene_table.iloc[:, 0]]
    all_inst_id = np.unique(np.concatenate((instance_id_df["inst_id"], instance_id_df["ctl_inst_id"])))

    expre_data = parse(
        str(DATA_DIR / "raw_data/LINCS_GSE92742/GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx"),    # time-costly
        cid=all_inst_id,
        rid=genes)
    expre_data = expre_data.data_df

    instance_id_df.index = range(len(instance_id_df))
    sample_num = instance_id_df.shape[0]

    drug_code = np.zeros((sample_num, 166))
    x1 = np.zeros((sample_num, expre_data.shape[0]))
    x2 = np.zeros((sample_num, expre_data.shape[0]))
    cid = []
    conc = []
    dur = []
    drug = []
    rna_plates = []
    for i in range(sample_num):
        ctl_inst_id = instance_id_df.loc[i, "ctl_inst_id"]
        trt_inst_id = instance_id_df.loc[i, "inst_id"]

        x1[i, :] = expre_data.loc[:, ctl_inst_id].values    # ctl expression
        x2[i, :] = expre_data.loc[:, trt_inst_id].values    # trt expression
        pert_id = instance_id_df.loc[i, "pert_id"]   # pert id(BRD)
        drug_code[i, :] = drug_df.loc[pert_id, "pert_code"]  # pert code(Fingerprint)
        cid.append(instance_id_df.loc[i, "cell_id"])  # cell line
        conc.append(instance_id_df.loc[i, "pert_dose"])  # concentration
        drug.append(instance_id_df.loc[i, "pert_iname_x"])   # drug name
        rna_plates.append(instance_id_df.loc[i, "rna_plate"])    # rna plate
        dur.append(24)  # duration
    
    return x1, x2, drug_code


# for pytorch model loading
sys.path.append(str(ROOT_DIR / "StateTransitionModel/Model/"))
from model import STATE_TRANSITION


import torch
model_file = DATA_DIR / "working_log_2025/state_transition_model/default_model/random_model_5uM.pth"
StateTransition = torch.load(model_file, map_location="cuda:0")
out_file = DATA_DIR / "preprocessed_data_2025/state_transition_model_input_data/step2_model_training_dataset_random_split_5uM.pkl"

with open(out_file, "rb") as f:
    _, _, _, _, _, _, _, _, _, x_scaler, y_scaler, _ = pickle.load(f)

def one_step_predict(cur_state, action):

    drug = action.reshape(-1, 166)  # (1, 166)
    cur_state_scale = x_scaler.transform(cur_state.reshape(-1, 978))  # (1, 978)
    delta_state_scale = StateTransition.predict(drug, cur_state_scale)
    delta_state = y_scaler.inverse_transform(delta_state_scale)
    next_state = cur_state + delta_state
    next_state = next_state.reshape(-1, 978)  # (1, 978)

    return next_state

# 24h
x1, x2, drug_code = get_data_from_instance_ids(match_data_24h)

x_24h = x1
d_24h = drug_code
y_24h_real = x2 - x1
predict_x2 = one_step_predict(x_24h, d_24h)
y_24h_predict = predict_x2 - x1

# 48h
x1, x2, drug_code = get_data_from_instance_ids(match_data_48h)

x_48h = x1
d_48h = drug_code
y_48h_real = x2 - x1
temp = one_step_predict(x_48h, d_48h)
predict_x2 = one_step_predict(temp, d_48h)
y_48h_predict = predict_x2 - x1
y_24h_predict_testing = temp - x1

import numpy as np
from sklearn import metrics

def sample_based_pcc(y_true, y_pred, low_gene_thre, high_gene_thre):
    """pcc : calculate pearson correlation coefficient average by gene

    Parameters
    ----------
    y_true : np.ndarray [n*m]
        true label
    y_pred : np.ndarray [n*m]
        predict label
    gene_thre : np.ndarray [m]
        label thresholds
        which we use to select the significant expression values

    Returns
    -------
    list
        [pcc_value, pcc_value_precision]
    """

    pcc_value = 0.0
    pcc_value_precision = 0.0

    sample_number = y_pred.shape[0]
    precision_null_sample_number = 0

    original_pcc_arr = np.zeros(sample_number)
    precision_pcc_arr = np.zeros(sample_number)

    # calculate PCC by sample
    for i in range(sample_number):

        flag = False

        # 1. standard pcc
        original_pcc_arr[i] = np.corrcoef(y_true[i, :], y_pred[i, :])[0, 1]
        pcc_value += np.corrcoef(y_true[i, :], y_pred[i, :])[0, 1]

        # 2. precision pcc(focusing on true data)
        a = y_true[i, :].copy()
        b = y_pred[i, :].copy()
        low_gene_thre_val, high_gene_thre_val = np.percentile(np.abs(a), [low_gene_thre, high_gene_thre])
        ind = (np.abs(a) > low_gene_thre_val) & (np.abs(a) <= high_gene_thre_val)
        if np.sum(ind) > 1:
            a = a[ind]
            b = b[ind]
            temp1 = np.corrcoef(a, b)[0, 1]
            pcc_value_precision += temp1
            precision_pcc_arr[i] = temp1
        else:
            precision_null_sample_number += 1
            flag = True

    return [original_pcc_arr, precision_pcc_arr]

# PCC
segmentation_num = 10
all_precision_pcc_arr = []
for thre in range(0,100,segmentation_num):
    original_pcc_arr, precision_pcc_arr = sample_based_pcc(y_24h_real, y_24h_predict, thre, (thre+10))
    all_precision_pcc_arr.append(precision_pcc_arr)
    # print(np.mean(precision_pcc_arr))
    
all_precision_pcc_arr = np.array(all_precision_pcc_arr).reshape(-1)
plot_df1 = pd.DataFrame({"pcc": all_precision_pcc_arr, "label": np.repeat(np.arange(1, segmentation_num+1), y_24h_real.shape[0])})

segmentation_num = 10
all_precision_pcc_arr = []
for thre in range(0,100,segmentation_num):
    original_pcc_arr, precision_pcc_arr = sample_based_pcc(y_48h_real, y_48h_predict, thre, (thre+10))
    all_precision_pcc_arr.append(precision_pcc_arr)
all_precision_pcc_arr = np.array(all_precision_pcc_arr).reshape(-1)
plot_df2 = pd.DataFrame({"pcc": all_precision_pcc_arr, "label": np.repeat(np.arange(1, segmentation_num+1), y_48h_real.shape[0])})

# save
with open(DATA_DIR / "result_2025/state_transition_model/default_model/24vs48_plot_data.pkl", "wb") as f:
    pickle.dump([plot_df1, plot_df2], f)
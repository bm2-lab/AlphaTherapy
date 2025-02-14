# Extract usable data from the LINCS database

#####################################################################
# Step 1: Filter suitable experiments (instances) 【24h & ~10uM/5uM】
# Step 2: Match treatment groups with corresponding control groups 
#         (based on RNA plate sequence)
# Step 3: Retrieve compound SMILES strings and generate fingerprints
# Step 4: Retrieve expression profiles by instance ID and construct 
#         training data for the state transition model
# Step 5: Save the data
#####################################################################

import os
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from cmapPy.pandasGEXpress.parse import parse

# Set working directory
# users should download the raw data from the GSE92742
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.resolve().parent.resolve().parent
sys.path.append(str(ROOT_DIR))
from path import DATA_DIR  

print("ROOT_DIR:", ROOT_DIR)
print("DATA_DIR:", DATA_DIR)


os.chdir(DATA_DIR / "raw_data" / "LINCS_GSE92742")
np.random.seed(2222)

#####################################################################
# Step 1: Filter suitable experiments (instances)
inst_info = pd.read_table('GSE92742_Broad_LINCS_inst_info.txt', sep='\t', header=0, low_memory=False)

# Filter by experimental duration (24 hours)
inst_info = inst_info[inst_info["pert_time"] == 24]
inst_info_control = inst_info[inst_info["pert_id"] == "DMSO"]
inst_info_trt = inst_info[inst_info["pert_id"] != "DMSO"]

# Filter by experimental concentration (~10uM)
concentration = 10.0
inst_info_trt = inst_info_trt[
    (inst_info_trt["pert_dose"] < concentration + 1) & 
    (inst_info_trt["pert_dose"] > concentration - 1)
]

#####################################################################
# Step 2: Match treatment groups with corresponding control groups
instance_rna_plates = np.unique(inst_info_trt["rna_plate"])
inst_info_trt_new = pd.DataFrame()

for rna_plate in instance_rna_plates:
    inst_info_control_plate = inst_info_control[inst_info_control["rna_plate"] == rna_plate]
    if inst_info_control_plate.empty:
        continue
    inst_info_trt_plate = inst_info_trt[inst_info_trt["rna_plate"] == rna_plate]

    inst_info_trt_plate.loc[:, "ctl_inst_id"] = np.random.choice(
        inst_info_control_plate["inst_id"], len(inst_info_trt_plate)
    )  # Randomly select with replacement
    inst_info_trt_new = pd.concat([inst_info_trt_new, inst_info_trt_plate])

#####################################################################
# Step 3: Retrieve compound fingerprints
pert_info = pd.read_table("GSE92742_Broad_LINCS_pert_info.txt", sep="\t", header=0)
pert_info.index = np.array(pert_info["pert_id"])
inst_info_trt_new = pd.merge(inst_info_trt_new, pert_info, on="pert_id")

drug_table = []
drugs_pert_id = np.unique(inst_info_trt_new["pert_id"])
count = 0
del_drug_name = []

for drug_pert_id in drugs_pert_id:
    canonical_smiles = pert_info.loc[drug_pert_id, "canonical_smiles"]
    if pd.isna(canonical_smiles) or canonical_smiles in ["-666", "restricted"]:
        del_drug_name.append(drug_pert_id)
        continue
    try:
        m = Chem.MolFromSmiles(canonical_smiles)
        MACCS_str = MACCSkeys.GenMACCSKeys(m).ToBitString()
        MACCS_code = np.array([int(i) for i in MACCS_str])
    except:
        del_drug_name.append(drug_pert_id)
        continue

    drug_fingerprint = MACCS_code[1:]
    drug_name = pert_info.loc[drug_pert_id, "pert_iname"]
    drug_table.append([count, drug_pert_id, drug_name, drug_fingerprint])
    count += 1
    print(f"{count} drugs have been converted to fingerprints!")

drug_df = pd.DataFrame(drug_table, columns=["index", "pert_id", "pert_iname", "pert_code"])
drug_df.index = drug_df["pert_id"]
inst_info_trt_new = inst_info_trt_new[inst_info_trt_new["pert_id"].isin(drug_df["pert_id"])]

#####################################################################
# Step 4: Retrieve expression profiles and construct training data
gene_table = pd.read_table(ROOT_DIR / "../data/cmap_978_genes_order.txt", header=None)
genes = gene_table.iloc[:, 0].astype(str).tolist()
all_inst_id = np.concatenate([inst_info_trt_new["inst_id"], inst_info_control["inst_id"]])

print("all_inst_id shape", all_inst_id.shape)
expre_data = parse(
    "GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx", 
    cid=all_inst_id, 
    rid=genes
).data_df

inst_info_trt_new.reset_index(drop=True, inplace=True)
sample_num = len(inst_info_trt_new)

# Initialize arrays to store data
drug_code = np.zeros((sample_num, 166))
x1 = np.zeros((sample_num, expre_data.shape[0]))
x2 = np.zeros((sample_num, expre_data.shape[0]))
cid, conc, dur, drug, rna_plates = [], [], [], [], []

for i in range(sample_num):
    print(f"{i} samples have been converted!")
    ctl_inst_id = inst_info_trt_new.loc[i, "ctl_inst_id"]
    trt_inst_id = inst_info_trt_new.loc[i, "inst_id"]

    x1[i, :] = expre_data.loc[:, ctl_inst_id].values  # Control expression
    x2[i, :] = expre_data.loc[:, trt_inst_id].values  # Treatment expression
    pert_id = inst_info_trt_new.loc[i, "pert_id"]
    drug_code[i, :] = drug_df.loc[pert_id, "pert_code"]  # Drug fingerprint
    cid.append(inst_info_trt_new.loc[i, "cell_id"])
    conc.append(inst_info_trt_new.loc[i, "pert_dose"])
    drug.append(inst_info_trt_new.loc[i, "pert_iname_x"])
    rna_plates.append(inst_info_trt_new.loc[i, "rna_plate"])
    dur.append(24)  # Duration

#####################################################################
# Step 5: Save the data
data_path = DATA_DIR / "preprocessed_data_2025" / "state_transition_model_input_data" / "step1_LINCS_model_dataset.pkl"
data_path.parent.mkdir(parents=True, exist_ok=True) 

# Save step1 preprocessed dataset
with open(data_path, "wb") as f:
    pickle.dump(
        [drug_code, x1, x2, np.array(cid), np.array(conc), np.array(dur), np.array(drug), np.array(rna_plates)], 
        f
    )

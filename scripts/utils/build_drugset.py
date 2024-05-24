import os
import gym
import pickle
import argparse 
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys

# The parameters of input
argparser = argparse.ArgumentParser()
argparser.add_argument('--input', type=str, help='the input data file name (*.xlsx)', required=True)
argparser.add_argument('--output', type=str, help='the output data file (*.pickle)', required=True)
args = argparser.parse_args()

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# read drug information list
drug_info = pd.read_excel("./data/" + args.input, engine="openpyxl")

# convert drug smiles to fingerprint
valid_drug_ind = []
drug_fingerprint_dict = {}
count=0
for i in range(drug_info.shape[0]):
    drug = drug_info.loc[i, "Name"]
    canonical_smiles = drug_info.loc[i, "SMILES"]
    try:
        m = Chem.MolFromSmiles(canonical_smiles)
        MACCS_str = MACCSkeys.GenMACCSKeys(m).ToBitString()
        MACCS_code = np.array([int(i) for i in list(MACCS_str)])

        valid_drug_ind.append(i)
        drug_fingerprint_dict[drug] = MACCS_code[1:]

    except :
        print(drug)
        print(canonical_smiles)

# 2. Construct a drugset.
drugset = []
new_ind = 0

for i in valid_drug_ind:
    drug = drug_info.loc[i,"Name"]
    target = str(drug_info.loc[i,"Target"]).split(",")
    pathway = drug_info.loc[i,"Pathway"]
    drugset.append([new_ind, "_", drug, drug_fingerprint_dict[drug], target, pathway])
    new_ind+=1
drugset = np.array(drugset, dtype=object)

save_file = "./scripts/gym_cell_model/gym_cell_model/data/" + args.output
with open(save_file, "wb") as f:
    pickle.dump(drugset, f)

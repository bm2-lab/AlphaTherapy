#############################################
# Script Summary
#############################################
# This script processes CTRP and LINCS datasets to generate training data (X and Y)
# for a cell viability regression model. The steps include:
# 1. Processing CTRP data to extract and organize relevant information.
# 2. Filtering experiments to match cell lines and compounds between CTRP and LINCS datasets.
# 3. Extracting expression profiles and calculating delta expression profiles.
# 4. Saving the processed data to CSV files.


import os
import numpy as np
import pandas as pd
from cmapPy.pandasGEXpress.parse import parse

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.resolve().parent
sys.path.append(str(ROOT_DIR))
from path import DATA_DIR  

# Set the project directory
os.chdir(DATA_DIR / "raw_data" / "CTRP")

print("ROOT_DIR:", ROOT_DIR)
print("DATA_DIR:", DATA_DIR)


#####################################################################
# 1. Process CTRP data
#####################################################################

# Read the raw CTRP cell viability data and metadata
ctrp_raw = pd.read_table('./v20.data.per_cpd_post_qc.txt', sep='\t', header=0, index_col=None)

# Read CTRP metadata
cell_info = pd.read_table('./v20.meta.per_cell_line.txt', sep='\t', header=0, index_col=None)
compound_info = pd.read_table('./v20.meta.per_compound.txt', sep='\t', header=0, index_col=None)
experiment_info = pd.read_table('./v20.meta.per_experiment.txt', sep='\t', header=0, index_col=None)

# Add CTRP metadata (ccl_name, broad_cpd_id) to the raw data
ctrp_proc = ctrp_raw.loc[:, ['experiment_id', 'master_cpd_id', 'cpd_conc_umol', 'cpd_avg_pv']].copy()

# Remove duplicate experiments and add relevant metadata to processed data
experiment_info = experiment_info.drop_duplicates(['experiment_id', 'master_ccl_id'])
experiment_info.index = experiment_info['experiment_id']
ctrp_proc['master_ccl_id'] = experiment_info.loc[ctrp_proc['experiment_id'].values, 'master_ccl_id'].values
cell_info.index = cell_info['master_ccl_id']
ctrp_proc['ccl_name'] = cell_info.loc[ctrp_proc['master_ccl_id'].values, 'ccl_name'].values
compound_info.index = compound_info['master_cpd_id']
ctrp_proc['broad_cpd_id'] = compound_info.loc[ctrp_proc['master_cpd_id'].values, 'broad_cpd_id'].values

# Keep relevant columns for further processing
ctrp_proc = ctrp_proc.loc[:, ['ccl_name', 'broad_cpd_id', 'cpd_conc_umol', 'cpd_avg_pv']]

#####################################################################
# 2. Filter experiments from CTRP and L1000 with matching cell lines, compounds, and similar concentrations
#####################################################################

# 2.1 Identify common cell lines between L1000 and CTRP
gse92742_cell = pd.read_table('../LINCS_GSE92742/GSE92742_Broad_LINCS_cell_info.txt', sep='\t', header=0, index_col=None)
lincs_cells = list(set(gse92742_cell['cell_id']) & set(ctrp_proc['ccl_name']))
print(f"L1000 and CTRP have {len(lincs_cells)} common cell lines")

# 2.2 Identify common compounds between L1000 and CTRP
gse92742_comp = pd.read_table('../LINCS_GSE92742/GSE92742_Broad_LINCS_pert_info.txt', sep='\t', header=0, index_col=None)
lincs_compounds = list(set(gse92742_comp['pert_id']) & set(ctrp_proc['broad_cpd_id']))
print(f"L1000 and CTRP have {len(lincs_compounds)} common compounds")

# 2.3 Filter CTRP data for experiments with matching cell lines and compounds
fil = np.in1d(ctrp_proc['ccl_name'], lincs_cells) & np.in1d(ctrp_proc['broad_cpd_id'], lincs_compounds)
ctrp_proc = ctrp_proc[fil]
print(f'Filtered CTRP data with matching experiments from L1000: {ctrp_proc.shape}')

# For the same ('ccl_name', 'broad_cpd_id', 'cpd_conc_umol'), take the average
ctrp_proc = ctrp_proc.groupby(['ccl_name', 'broad_cpd_id', 'cpd_conc_umol']).mean()
ctrp_proc.reset_index(inplace=True)
print(f'Data size after averaging experiments with identical conditions: {ctrp_proc.shape}')

CTRP = ctrp_proc

def get_closest_cc_ctrp(l1000):
    """
    Given an L1000 sample, find the closest matching data in CTRP with the same cell line and compound,
    and the most similar concentration.
    Returns the cell viability (cpd_avg_pv) and log10 of concentration (log10_cpd_conc_umol) from CTRP.
    """
    fil = (CTRP['broad_cpd_id'] == l1000['pert_id']) & (CTRP['ccl_name'] == l1000['cell_id'])
    if np.sum(fil) > 0:
        ctrp_temp = CTRP[fil].copy()  # CTRP data for the given drug in the cell line
        ctrp_temp['delta_cc'] = np.abs(ctrp_temp['log10_cpd_conc_umol'] - l1000['log10_pert_dose'])
        j = ctrp_temp.sort_values('delta_cc').index[0]
        return ctrp_temp.loc[j, ['cpd_avg_pv', 'log10_cpd_conc_umol']]
    else:
        return np.nan

# 2.4 Filter L1000 data for experiments with matching cell lines and compounds from CTRP
CTRP['log10_cpd_conc_umol'] = np.log10(CTRP['cpd_conc_umol'])
cell_lines = list(set(CTRP['ccl_name']))
compounds = list(set(CTRP['broad_cpd_id']))

inst_info_gse92742 = pd.read_table('../LINCS_GSE92742/GSE92742_Broad_LINCS_inst_info.txt', sep='\t', header=0, index_col=None, low_memory=False)
fil = np.in1d(inst_info_gse92742['pert_id'], compounds) & np.in1d(inst_info_gse92742['cell_id'], cell_lines)
inst_info_gse92742 = inst_info_gse92742[fil]

# For each L1000 data point, find the closest matching CTRP data based on concentration (log10)
inst_info_gse92742['pert_dose'] = inst_info_gse92742['pert_dose'].astype(float)
fil = inst_info_gse92742['pert_dose'] != 0.0  # Remove 0 concentration instances
inst_info_gse92742 = inst_info_gse92742[fil]
inst_info_gse92742['log10_pert_dose'] = np.log10(inst_info_gse92742['pert_dose'].astype(float))

inst_info_gse92742_nearest = inst_info_gse92742.apply(get_closest_cc_ctrp, axis=1)
inst_info_gse92742 = pd.concat([inst_info_gse92742, inst_info_gse92742_nearest], axis=1)  # Add y values to L1000 data
fil = ~pd.isnull(inst_info_gse92742['cpd_avg_pv'])  # Remove L1000 data without matching concentrations
inst_info_gse92742 = inst_info_gse92742[fil]  # (14147, 15)

#####################################################################
# 3. Extract expression profiles
# 3.1 Process treatment expression profiles
#####################################################################

inst_ids_gse92742 = list(inst_info_gse92742['inst_id'])
gene_table = pd.read_table(ROOT_DIR / "../data/cmap_978_genes_order.txt", sep='\t', header=None)
genes = [str(i) for i in gene_table.iloc[:, 0]]
expre_gse92742 = parse('../LINCS_GSE92742/GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx', cid=inst_ids_gse92742, rid=genes)
expre_gse92742 = expre_gse92742.data_df.T

inst_info_gse92742.index = inst_info_gse92742['inst_id']
inst_info_gse92742 = inst_info_gse92742.loc[:, ['pert_id', 'cell_id', 'pert_iname', 'log10_pert_dose', 'pert_time',
                                                'log10_cpd_conc_umol', 'cpd_avg_pv']]

# 3.2 Filter L1000 and CTRP data with concentration log difference less than 0.2
fil = np.abs(inst_info_gse92742['log10_pert_dose'] - inst_info_gse92742['log10_cpd_conc_umol']) < 0.2
inst_info_gse92742 = inst_info_gse92742[fil]
inst_info_l1000_ctrp = inst_info_gse92742
expre_l1000_ctrp = expre_gse92742.loc[inst_info_l1000_ctrp.index, genes]

# 3.3 Extract control group expression profiles
inst_info = pd.read_table('../LINCS_GSE92742/GSE92742_Broad_LINCS_inst_info.txt', sep='\t', header=0, index_col=None, low_memory=False)
inst_info_control = inst_info[inst_info["pert_id"] == "DMSO"]
inst_info_trt = inst_info[inst_info["pert_id"] != "DMSO"]
drugs = np.unique(np.array(inst_info_trt["pert_id"]))
rna_plates = np.unique(np.array(inst_info_trt["rna_plate"]))

# Match control groups based on the rna_plate order
np.random.seed(123)
inst_info_trt_new = pd.DataFrame()
for rna_plate in rna_plates:
    inst_info_control_plate = inst_info_control[inst_info_control["rna_plate"] == rna_plate]
    if inst_info_control_plate.empty:
        continue
    inst_info_trt_plate = inst_info_trt[inst_info_trt["rna_plate"] == rna_plate]
    inst_info_trt_plate["ctl_inst_id"] = np.random.choice(inst_info_control_plate.loc[:, "inst_id"], len(inst_info_trt_plate))  # Randomly select control group
    inst_info_trt_new = pd.concat((inst_info_trt_new.loc[:, :], inst_info_trt_plate.loc[:, :]))  # Concatenate

inst_info_trt_new.index = inst_info_trt_new["inst_id"]
ctl_inst_id = inst_info_trt_new.loc[inst_info_l1000_ctrp.index, "ctl_inst_id"]
expr_cont = parse('../LINCS_GSE92742/GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx', cid=np.unique(ctl_inst_id), rid=genes)
expr_cont = expr_cont.data_df.T
expr_cont = expr_cont.loc[ctl_inst_id, genes]

# 3.4 Calculate delta expression profiles
expre_l1000_ctrp_delta = expre_l1000_ctrp.values - expr_cont.values
expre_l1000_ctrp_delta = pd.DataFrame(expre_l1000_ctrp_delta)
expre_l1000_ctrp_delta.index = expre_l1000_ctrp.index
expre_l1000_ctrp_delta.columns = expre_l1000_ctrp.columns

#####################################################################
# 4. Save Data
#####################################################################

x_path =DATA_DIR / "preprocessed_data_2025" / "cell_viability_model_data" / "X.csv"
y_path = DATA_DIR / "preprocessed_data_2025" / "cell_viability_model_data" / "Y.csv"

x_path.parent.mkdir(parents=True, exist_ok=True)
y_path.parent.mkdir(parents=True, exist_ok=True)

expre_l1000_ctrp_delta.to_csv(
    x_path,
    sep=',')

inst_info_l1000_ctrp.to_csv(
    y_path,
    sep=',')
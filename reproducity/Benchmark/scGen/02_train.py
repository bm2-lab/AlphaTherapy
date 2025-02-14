import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.resolve().parent.resolve().parent
sys.path.insert(0, str(ROOT_DIR))
from path import DATA_DIR  

import os
import pickle
import logging
import numpy as np
import scanpy as sc
import pandas as pd

import scgen

# 1. read data
adata = sc.read(ROOT_DIR / "Benchmark/scGen/datasets/lincs_dataset.h5ad")
adata = adata[~np.isin(adata.obs["AlphaTherapy_split"], "invalid_data")]
adata.obs["record_index"] = np.arange(len(adata))

error_log = []
test_idices = []
y_pred_result = []
y_true_result = []
unique_perturbation = np.unique(adata.obs.condition)

for i in range(len(unique_perturbation)):
    try:

        # 1. Data preparation
        print("###############")
        print(f"Processing perturbation index: {i}")
        p = unique_perturbation[i]
        treat_data = adata[adata.obs["condition"] == p].copy()
        treat_train_data = treat_data[
            (treat_data.obs["AlphaTherapy_split"] == "train") | 
            (treat_data.obs["AlphaTherapy_split"] == "valid")
        ].copy()
        control_data = sc.AnnData(X=treat_data.layers["control_expression"], obs=treat_data.obs)
        control_data.obs["condition"] = "DMSO"
        train_data = sc.concat([treat_train_data, control_data])
        unseen_cells = np.unique(treat_data[treat_data.obs["AlphaTherapy_split"] == "test"].obs["cell_type"])
        if len(unseen_cells) == 0:
            print(f"No unseen cells for perturbation: {p}, skipping.")
            continue
        
        # 2. Model training
        scgen.SCGEN.setup_anndata(train_data, batch_key="condition", labels_key="cell_type")
        model = scgen.SCGEN(train_data)
        model.save(ROOT_DIR / "Benchmark/scGen/model/",overwrite=True)
        model.train(
            max_epochs=100,
            batch_size=32,
            early_stopping=True,
            early_stopping_patience=25
        )
        
        # 3. Model prediction
        all_pred = []
        for unseen_cell in unseen_cells:
            pred, _ = model.predict(ctrl_key='DMSO', stim_key=p, celltype_to_predict=unseen_cell)
            all_pred.append(pred)
            y_pred_result.extend(pred.X)
        all_true = []
        
        for y_pred in all_pred:
            sample_record_index = np.array(y_pred.obs.record_index)
            record_to_row = {record: idx for idx, record in enumerate(treat_data.obs['record_index'])}
            sorted_indices = [record_to_row[record] for record in sample_record_index if record in record_to_row]
            y_true = treat_data.X[sorted_indices, :]
            extracted_record_indices = treat_data.obs.iloc[sorted_indices]['record_index'].to_numpy()
            assert np.array_equal(extracted_record_indices, sample_record_index), "Data order mismatch!"
            test_idices.append(extracted_record_indices)
            all_true.append(y_true)
            y_true_result.extend(y_true)
    
    except Exception as e:
        error_message = f"Error at index {i} with perturbation {p}: {str(e)}"
        error_log.append({"index": i, "perturbation": p, "error": str(e)})
        
        model.train(
            max_epochs=100,
            batch_size=32,
            early_stopping=True,
            early_stopping_patience=25,
            train_size = 0.5
        )
        # 3. Model prediction
        all_pred = []
        for unseen_cell in unseen_cells:
            pred, _ = model.predict(ctrl_key='DMSO', stim_key=p, celltype_to_predict=unseen_cell)
            all_pred.append(pred)
            y_pred_result.extend(pred.X)
        all_true = []
        for y_pred in all_pred:
            sample_record_index = np.array(y_pred.obs.record_index)
            record_to_row = {record: idx for idx, record in enumerate(treat_data.obs['record_index'])}
            sorted_indices = [record_to_row[record] for record in sample_record_index if record in record_to_row]
            y_true = treat_data.X[sorted_indices, :]
            extracted_record_indices = treat_data.obs.iloc[sorted_indices]['record_index'].to_numpy()
            assert np.array_equal(extracted_record_indices, sample_record_index), "Data order mismatch!"
            test_idices.append(extracted_record_indices)
            all_true.append(y_true)
            y_true_result.extend(y_true)

output_path = ROOT_DIR / "Benchmark/scGen/results/scgen_predition_result.h5ad"
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "wb") as f:
    pickle.dump([y_true_result, y_pred_result, error_log], f)
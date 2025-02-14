import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.resolve().parent.resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from path import DATA_DIR  

import os
import sys
import json
import yaml
import torch
import pickle

from tqdm import tqdm
from sklearn.metrics import r2_score
from seml.config import generate_configs, read_config
from chemCPA.data import load_dataset_splits #type: ignore
PROJECT_DIR = ROOT_DIR / "Benchmark/chemCPA/chemCPA-main/project_folder"
from chemCPA.experiments_run import ExperimentWrapper

import numpy as np
import pandas as pd
import scanpy as sc

script_path = ROOT_DIR / "Benchmark/chemCPA/chemCPA-main"
os.chdir(str(script_path))
sys.path.append(str(script_path))
sys.path.append(ROOT_DIR / "Benchmark/chemCPA/src")
from MyUtils import *  #type: ignore

################# Training ###################
def TrainChemCPA(dataset_path, model_file_name="output_01.pt", result_file_name="training_result_01.pkl", args_file_name="args_01.yaml"):
    
    exp = ExperimentWrapper(init_all=False)
    exp.seed = 1337

    _, _, experiment_config = read_config("lincs_01.yaml")
    configs = generate_configs(experiment_config)
    args = configs[0]

    args["dataset"]["data_params"]["dataset_path"] = dataset_path
    args["training"]["file_name"] = model_file_name

    # loads the dataset splits
    exp.init_dataset(**args["dataset"])

    exp.init_drug_embedding(embedding=args["model"]["embedding"])

    exp.init_model(
        hparams=args["model"]["hparams"],
        additional_params=args["model"]["additional_params"],
        load_pretrained=args["model"]["load_pretrained"],
        append_ae_layer=args["model"]["append_ae_layer"],
        enable_cpa_mode=args["model"]["enable_cpa_mode"],
        pretrained_model_path = args["model"]["pretrained_model_path"],
        pretrained_model_hashes = args["model"]["pretrained_model_hashes"],
    )

    # setup the torch DataLoader
    exp.update_datasets()

    results = exp.train(**args["training"])

    with open(result_file_name, "wb") as f:
        pickle.dump(results, f)

    with open(args_file_name, "w") as yaml_file:
        yaml.dump(args, yaml_file, default_flow_style=False, sort_keys=False)

def load_config(seml_collection, model_hash):
    file_path = '{}/{}.json'.format(PROJECT_DIR, seml_collection)  # Provide path to json

    with open(file_path) as f:
        file_data = json.load(f)

    for _config in tqdm(file_data):
        if _config["config_hash"] == model_hash:
            config = _config["config"]
            config["config_hash"] = _config["config_hash"]
    return config

def r2_scores_sample(y_true, y_pred):
    r2_scores = np.array([r2_score(y_true[i], y_pred[i]) for i in range(y_true.shape[0])])
    return r2_scores


import numpy as np
from scipy.stats import pearsonr

def pcc_sample(y_true, y_pred):

    pcc_scores = []
    for i in range(len(y_true)):
        pcc, _ = pearsonr(y_true[i], y_pred[i])
        pcc_scores.append(pcc)
    return np.array(pcc_scores)

def rmse_sample(y_true, y_pred):

    rmse_scores = []
    for i in range(len(y_true)):
        rmse = np.sqrt(np.mean((y_true[i] - y_pred[i]) ** 2))
        rmse_scores.append(rmse)
    return np.array(rmse_scores)


##################Predicting#####################
def PredictChemCPA(dataset_path, config_hash="output_01", result_file_name="testing_result_01.pkl"):

    config = load_config("chemCPA_configs", "c824e42f7ce751cf9a8ed26f0d9e0af7")
    config["dataset"]["data_params"]["dataset_path"] = dataset_path
    config['dataset']['data_params']['degs_key'] = 'rank_genes_groups_cov'
    config['dataset']['data_params']['split_key'] = 'AlphaTherapy_split'
    config['dataset']['data_params']['dose_key'] = 'dose_val'
    config['dataset']['data_params']['smiles_key'] = 'canonical_smiles'

    config['config_hash']= config_hash
    config["model"]["load_pretrained"] = True
    config["model"]["append_ae_layer"] = False
    config['model']['pretrained_model_hashes']['rdkit'] = config_hash 
    config["model"]["embedding"]["directory"] = ROOT_DIR / "Benchmark/chemCPA/chemCPA-main/project_folder/embeddings"


    dataset, key_dict = load_dataset(config)  #type: ignore
    config['dataset']['n_vars'] = dataset.n_vars

    canon_smiles_unique_sorted, _ = load_smiles(config, dataset, key_dict, False)  #type: ignore

    data_params = config['dataset']['data_params']
    datasets, _ = load_dataset_splits(**data_params, return_dataset=True)

    dosages = list(np.unique(datasets['training'].dosages))
    dosages = [i for i in dosages if i !=0]
    cell_lines = list(np.unique(datasets['training'].covariate_names['cell_type']))

    model_pretrained, _ = load_model(config, canon_smiles_unique_sorted)  #type: ignore

    data = sc.read(dataset_path)
    ood_data = data[data.obs['AlphaTherapy_split'] == 'ood']
    control_gene_expression = ood_data.layers["control_gene_expression"]
    control_gene_expression = torch.tensor(control_gene_expression, dtype=torch.float)

    del data

    _, y_true_dict, y_pred_dict = compute_pred1(model = model_pretrained,     #type: ignore
                            dataset = datasets['ood'], # 
                            genes_control = control_gene_expression,  
                            dosages=dosages,
                            cell_lines=cell_lines,
                            use_DEGs=False,
                            verbose=False,
                        )

    with open(result_file_name, "wb") as f:
        pickle.dump([y_true_dict, y_pred_dict], f)


if __name__ == '__main__':

    for dataset_type in ["random_split", "cell_blind", "chemical_blind"]:

        dataset_path = DATA_DIR / f'preprocessed_data_2025/state_transition_model_benchmark/chemCPA/{dataset_type}.h5ad'
        print(type(dataset_type))
        model_hash = f"output_{dataset_type}"
        model_file_name = f"output_{dataset_type}.pt"
        training_result_file_name = f"training_result_{dataset_type}.pkl"
        testing_result_file_name = f"testing_result_{dataset_type}.pkl"
        args_file_name = f"args_{dataset_type}.yaml"

        TrainChemCPA(dataset_path, model_file_name, training_result_file_name, args_file_name)
        PredictChemCPA(dataset_path, model_hash, testing_result_file_name)
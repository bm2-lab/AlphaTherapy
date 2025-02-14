import scanpy as sc
import numpy as np, pandas as pd
from chemCPA.data import drug_names_to_once_canon_smiles
from chemCPA.helper import canonicalize_smiles

import numpy as np
import pandas as pd
import scanpy as sc
import seml
import torch
from tqdm.auto import tqdm

# from chemCPA.data import (
#     SubDataset,
#     canonicalize_smiles,
#     drug_names_to_once_canon_smiles,
# )
from chemCPA.embedding import get_chemical_representation
from chemCPA.model import ComPert
from chemCPA.paths import CHECKPOINT_DIR
from chemCPA.train import bool2idx, compute_prediction, compute_r2, repeat_n


def load_dataset(config):
    perturbation_key = config["dataset"]["data_params"]["perturbation_key"]
    smiles_key = config["dataset"]["data_params"]["smiles_key"]
    dataset = sc.read(config["dataset"]["data_params"]["dataset_path"])
    key_dict = {
        "perturbation_key": perturbation_key,
        "smiles_key": smiles_key,
    }
    return dataset, key_dict




def load_smiles(config, dataset, key_dict, return_pathway_map=False):
    perturbation_key = key_dict["perturbation_key"]
    smiles_key = key_dict["smiles_key"]

    # this is how the `canon_smiles_unique_sorted` is generated inside chemCPA.data.Dataset
    # we need to have the same ordering of SMILES, else the mapping to pathways will be off
    # when we load the Vanilla embedding. For the other embeddings it's not as important.
    drugs_names = np.array(dataset.obs[perturbation_key].values)
    drugs_names_unique = set()
    for d in drugs_names:
        [drugs_names_unique.add(i) for i in d.split("+")]
    drugs_names_unique_sorted = np.array(sorted(drugs_names_unique))
    canon_smiles_unique_sorted = drug_names_to_once_canon_smiles(
        list(drugs_names_unique_sorted), dataset, perturbation_key, smiles_key
    )

    smiles_to_drug_map = {
        canonicalize_smiles(smiles): drug
        for smiles, drug in dataset.obs.groupby(
            [config["dataset"]["data_params"]["smiles_key"], perturbation_key]
        ).groups.keys()
    }
    if return_pathway_map:
        smiles_to_pathway_map = {
            canonicalize_smiles(smiles): pathway
            for smiles, pathway in dataset.obs.groupby(
                [config["dataset"]["data_params"]["smiles_key"], "pathway_level_1"]
            ).groups.keys()
        }
        return canon_smiles_unique_sorted, smiles_to_pathway_map, smiles_to_drug_map
    return canon_smiles_unique_sorted, smiles_to_drug_map


def load_model(config, canon_smiles_unique_sorted):
    model_hash = config["config_hash"]
    #model_checkp = CHECKPOINT_DIR / (model_hash + ".pt")
    model_checkp = config['model']['pretrained_model_path'] + '/' + (model_hash + ".pt")

    embedding_model = config["model"]["embedding"]["model"]
    if embedding_model == "vanilla":
        embedding = None
    else:
        embedding = get_chemical_representation(
            smiles=canon_smiles_unique_sorted,
            embedding_model=config["model"]["embedding"]["model"],
            data_dir=config["model"]["embedding"]["directory"],
            device="cuda",
        )
    dumped_model = torch.load(model_checkp)
    if len(dumped_model) == 3:
        print("This model does not contain the covariate embeddings or adversaries.")
        state_dict, init_args, history = dumped_model
        COV_EMB_AVAILABLE = False
    elif len(dumped_model) == 4:
        print("This model does not contain the covariate embeddings.")
        state_dict, cov_adv_state_dicts, init_args, history = dumped_model
        COV_EMB_AVAILABLE = False
    elif len(dumped_model) == 5:
        (
            state_dict,
            cov_adv_state_dicts,
            cov_emb_state_dicts,
            init_args,
            history,
        ) = dumped_model
        COV_EMB_AVAILABLE = True
        assert len(cov_emb_state_dicts) == 1
    append_layer_width = (
        config["dataset"]["n_vars"]
        if (config["model"]["append_ae_layer"] and config["model"]["load_pretrained"])
        else None
    )

    if embedding_model != "vanilla":
        state_dict.pop("drug_embeddings.weight")
    model = ComPert(
        **init_args, drug_embeddings=embedding, append_layer_width=append_layer_width
    )
    model = model.eval()
    if COV_EMB_AVAILABLE:
        for embedding_cov, state_dict_cov in zip(
            model.covariates_embeddings, cov_emb_state_dicts
        ):
            embedding_cov.load_state_dict(state_dict_cov)

    incomp_keys = model.load_state_dict(state_dict, strict=False)
    if embedding_model == "vanilla":
        assert (
            len(incomp_keys.unexpected_keys) == 0 and len(incomp_keys.missing_keys) == 0
        )
    else:
        # make sure we didn't accidentally load the embedding from the state_dict
        torch.testing.assert_allclose(model.drug_embeddings.weight, embedding.weight)
        assert (
            len(incomp_keys.missing_keys) == 1
            and "drug_embeddings.weight" in incomp_keys.missing_keys
        ), incomp_keys.missing_keys
        # assert len(incomp_keys.unexpected_keys) == 0, incomp_keys.unexpected_keys

    return model, embedding



def compute_drug_embeddings(model, embedding, dosage=1e4):
    all_drugs_idx = torch.tensor(list(range(len(embedding.weight))))
    dosages = dosage * torch.ones((len(embedding.weight),))
    # dosages = torch.ones((len(embedding.weight),))
    with torch.no_grad():
        # scaled the drug embeddings using the doser
        transf_embeddings = model.compute_drug_embeddings_(
            drugs_idx=all_drugs_idx, dosages=dosages
        )
        # apply drug embedder
        # transf_embeddings = model.drug_embedding_encoder(transf_embeddings)
    return transf_embeddings


def compute_pred(
    model,
    dataset,
    dosages=[1e4],
    cell_lines=None,
    genes_control=None,
    use_DEGs=True,
    verbose=True,
):
    # dataset.pert_categories contains: 'celltype_perturbation_dose' info
    pert_categories_index = pd.Index(dataset.pert_categories, dtype="category")

    allowed_cell_lines = []

    cl_dict = {
        torch.Tensor([1, 0, 0]): "A549",
        torch.Tensor([0, 1, 0]): "K562",
        torch.Tensor([0, 0, 1]): "MCF7",
    }

    if cell_lines is None:
        cell_lines = ["A549", "K562", "MCF7"]

    print(cell_lines)

    predictions_dict = {}
    drug_r2 = {}
    for cell_drug_dose_comb, category_count in tqdm(
        zip(*np.unique(dataset.pert_categories, return_counts=True))
    ):
        if dataset.perturbation_key is None:
            break

        # estimate metrics only for reasonably-sized drug/cell-type combos
        if category_count <= 5:
            continue

        # doesn't make sense to evaluate DMSO (=control) as a perturbation
        if (
            "dmso" in cell_drug_dose_comb.lower()
            or "control" in cell_drug_dose_comb.lower()
        ):
            continue

        # dataset.var_names is the list of gene names
        # dataset.de_genes is a dict, containing a list of all differentiably-expressed
        # genes for every cell_drug_dose combination.
        bool_de = dataset.var_names.isin(
            np.array(dataset.de_genes[cell_drug_dose_comb])
        )
        idx_de = bool2idx(bool_de)

        # need at least two genes to be able to calc r2 score
        if len(idx_de) < 2:
            continue

        bool_category = pert_categories_index.get_loc(cell_drug_dose_comb)
        idx_all = bool2idx(bool_category)
        idx = idx_all[0]
        y_true = dataset.genes[idx_all, :].to(device="cuda")


        if genes_control is None:
            n_obs = y_true.size(0)
        else:
            assert isinstance(genes_control, torch.Tensor)
            n_obs = genes_control.size(0)

        emb_covs = [repeat_n(cov[idx], n_obs) for cov in dataset.covariates]

        if dataset.dosages[idx] not in dosages:
            continue

        stop = False
        for tensor, cl in cl_dict.items():
            if (tensor == dataset.covariates[0][idx]).all():
                if cl not in cell_lines:
                    stop = True
        if stop:
            continue

        if dataset.use_drugs_idx:
            emb_drugs = (
                repeat_n(dataset.drugs_idx[idx], n_obs).squeeze(),
                repeat_n(dataset.dosages[idx], n_obs).squeeze(),
            )
        else:
            emb_drugs = repeat_n(dataset.drugs[idx], n_obs)

        # copies just the needed genes to GPU
        # Could try moving the whole genes tensor to GPU once for further speedups (but more memory problems)

        if genes_control is None:
            # print("Predicting AE alike.")
            mean_pred, _ = compute_prediction(
                model,
                y_true,
                emb_drugs,
                emb_covs,
            )
        else:
            # print("Predicting counterfactuals.")
            mean_pred, _ = compute_prediction(
                model,
                genes_control,
                emb_drugs,
                emb_covs,
            )

        y_pred = mean_pred.mean(0)
        y_true = y_true.mean(0)
        if use_DEGs:
            r2_m_de = compute_r2(y_true[idx_de].cuda(), y_pred[idx_de].cuda())
            print(f"{cell_drug_dose_comb}: {r2_m_de:.2f}") if verbose else None
            drug_r2[cell_drug_dose_comb] = max(r2_m_de, 0.0)
        else:
            r2_m = compute_r2(y_true.cuda(), y_pred.cuda())
            print(f"{cell_drug_dose_comb}: {r2_m:.2f}") if verbose else None
            drug_r2[cell_drug_dose_comb] = max(r2_m, 0.0)

        predictions_dict[cell_drug_dose_comb] = [y_true, y_pred, idx_de]
    return drug_r2, predictions_dict


def compute_pred_ctrl(
    dataset,
    dosages=[1e4],
    cell_lines=None,
    dataset_ctrl=None,
    use_DEGs=True,
    verbose=True,
):
    # dataset.pert_categories contains: 'celltype_perturbation_dose' info
    pert_categories_index = pd.Index(dataset.pert_categories, dtype="category")

    print("compute_pred_ctrl")

    allowed_cell_lines = []

    cl_dict = {
        torch.Tensor([1, 0, 0]): "A549",
        torch.Tensor([0, 1, 0]): "K562",
        torch.Tensor([0, 0, 1]): "MCF7",
    }

    if cell_lines is None:
        cell_lines = ["A549", "K562", "MCF7"]

    print(cell_lines)

    predictions_dict = {}
    drug_r2 = {}
    for cell_drug_dose_comb, category_count in tqdm(
        zip(*np.unique(dataset.pert_categories, return_counts=True))
    ):
        if dataset.perturbation_key is None:
            break

        # estimate metrics only for reasonably-sized drug/cell-type combos
        if category_count <= 5:
            continue

        # doesn't make sense to evaluate DMSO (=control) as a perturbation
        if (
            "dmso" in cell_drug_dose_comb.lower()
            or "control" in cell_drug_dose_comb.lower()
        ):
            continue

        # dataset.var_names is the list of gene names
        # dataset.de_genes is a dict, containing a list of all differentiably-expressed
        # genes for every cell_drug_dose combination.
        bool_de = dataset.var_names.isin(
            np.array(dataset.de_genes[cell_drug_dose_comb])
        )
        idx_de = bool2idx(bool_de)

        # need at least two genes to be able to calc r2 score
        if len(idx_de) < 2:
            continue

        bool_category = pert_categories_index.get_loc(cell_drug_dose_comb)
        idx_all = bool2idx(bool_category)
        idx = idx_all[0]
        y_true = dataset.genes[idx_all, :].to(device="cuda")

        cov_name = cell_drug_dose_comb.split("_")[0]
        cond = dataset_ctrl.covariate_names["cell_type"] == cov_name
        genes_control = dataset_ctrl.genes[cond]

        if genes_control is None:
            n_obs = y_true.size(0)
        else:
            assert isinstance(genes_control, torch.Tensor)
            n_obs = genes_control.size(0)

        emb_covs = [repeat_n(cov[idx], n_obs) for cov in dataset.covariates]

        if dataset.dosages[idx] not in dosages:
            continue

        stop = False
        for tensor, cl in cl_dict.items():
            if (tensor == dataset.covariates[0][idx]).all():
                if cl not in cell_lines:
                    stop = True
        if stop:
            continue

        if dataset.use_drugs_idx:
            emb_drugs = (
                repeat_n(dataset.drugs_idx[idx], n_obs).squeeze(),
                repeat_n(dataset.dosages[idx], n_obs).squeeze(),
            )
        else:
            emb_drugs = repeat_n(dataset.drugs[idx], n_obs)

        y_pred = genes_control.mean(0)
        y_true = y_true.mean(0)
        if use_DEGs:
            r2_m_de = compute_r2(y_true[idx_de].cuda(), y_pred[idx_de].cuda())
            print(f"{cell_drug_dose_comb}: {r2_m_de:.2f}") if verbose else None
            drug_r2[cell_drug_dose_comb] = max(r2_m_de, 0.0)
        else:
            r2_m = compute_r2(y_true.cuda(), y_pred.cuda())
            print(f"{cell_drug_dose_comb}: {r2_m:.2f}") if verbose else None
            drug_r2[cell_drug_dose_comb] = max(r2_m, 0.0)

        predictions_dict[cell_drug_dose_comb] = [y_true, y_pred, idx_de]
    return drug_r2, predictions_dict

def compute_pred1(
    model,
    dataset,
    dosages=[1e4],
    cell_lines=None,
    genes_control=None,
    use_DEGs=True,
    verbose=True,
):
    # print("verion1")
    # dataset.pert_categories contains: 'celltype_perturbation_dose' info
    pert_categories_index = pd.Index(dataset.pert_categories, dtype="category")
    
    # cl_dict = {
    #     torch.Tensor([1, 0, 0]): "A549",
    #     torch.Tensor([0, 1, 0]): "K562",
    #     torch.Tensor([0, 0, 1]): "MCF7",
    # }

    if cell_lines is None:
        cell_lines = ["A549", "K562", "MCF7"]

    print(cell_lines)

    predictions_dict = {}
    realdata_dict = {}
    ctrldata_dict = {}
    drug_r2 = {}

    for cell_drug_dose_comb, category_count in tqdm(
        zip(*np.unique(dataset.pert_categories, return_counts=True))
    ):

        if dataset.perturbation_key is None:
            break
        # estimate metrics only for reasonably-sized drug/cell-type combos
        # if category_count <= 5:
            # continue
        print(cell_drug_dose_comb, category_count)
        # doesn't make sense to evaluate DMSO (=control) as a perturbation
        if (
            "dmso" in cell_drug_dose_comb.lower()
            or "control" in cell_drug_dose_comb.lower()
        ):
            continue

        # dataset.var_names is the list of gene names
        # dataset.de_genes is a dict, containing a list of all differentiably-expressed
        # genes for every cell_drug_dose combination.
        
        bool_de = dataset.var_names.isin(
            np.array(dataset.de_genes[cell_drug_dose_comb])
        )
        idx_de = bool2idx(bool_de)

        # need at least two genes to be able to calc r2 score
        # if len(idx_de) < 2:
            # continue

        bool_category = pert_categories_index.get_loc(cell_drug_dose_comb)
        if type(bool_category) == type(0):
            idx_all = np.array([bool_category])
        else:
            idx_all = bool2idx(bool_category)

        idx = idx_all[0]
        y_true = dataset.genes[idx_all, :].to(device="cuda")
        ctrl_expre = genes_control[idx_all, :]


        # cov_name = cell_drug_dose_comb.split("_")[0]
        # cond = dataset_ctrl.covariate_names["cell_type"] == cov_name
        # genes_control = dataset_ctrl.genes[cond]

        # if genes_control is None:
        #     n_obs = y_true.size(0)
        # else:
        #     assert isinstance(genes_control, torch.Tensor)
        #     n_obs = genes_control.size(0)

        if ctrl_expre is None:
            n_obs = y_true.size(0)
        else:
            assert isinstance(ctrl_expre, torch.Tensor)
            n_obs = ctrl_expre.size(0)

        emb_covs = [repeat_n(cov[idx], n_obs) for cov in dataset.covariates]

        if dataset.dosages[idx] not in dosages:
            continue

        # stop = False
        # for tensor, cl in cl_dict.items():
        #     if (tensor == dataset.covariates[0][idx]).all():
        #         if cl not in cell_lines:
        #             stop = True
        # if stop:
        #     continue

        if dataset.use_drugs_idx:
            emb_drugs = (
                repeat_n(dataset.drugs_idx[idx], n_obs).squeeze(),
                repeat_n(dataset.dosages[idx], n_obs).squeeze(),
            )
        else:
            emb_drugs = repeat_n(dataset.drugs[idx], n_obs)

        # copies just the needed genes to GPU
        # Could try moving the whole genes tensor to GPU once for further speedups (but more memory problems)

        if ctrl_expre is None:
            # print("Predicting AE alike.")
            mean_pred, _ = compute_prediction(
                model,
                y_true,
                emb_drugs,
                emb_covs,
            )
        else:
            # print("Predicting counterfactuals.")
            mean_pred, variance = compute_prediction(
                model,
                ctrl_expre,
                emb_drugs,
                emb_covs,
            )

        if use_DEGs:
            mean_pred = mean_pred[:, idx_de]
        predictions_dict[cell_drug_dose_comb] = mean_pred
        realdata_dict[cell_drug_dose_comb] = y_true
        ctrldata_dict[cell_drug_dose_comb] = ctrl_expre

    return ctrldata_dict, realdata_dict, predictions_dict




def compute_pred_version1(
    model,
    dataset,
    dosages=[1e4],
    cell_lines=None,
    genes_control=None,
    use_DEGs=True,
    verbose=True,
):
    print("verion1")
    # dataset.pert_categories contains: 'celltype_perturbation_dose' info
    pert_categories_index = pd.Index(dataset.pert_categories, dtype="category")
    
    if cell_lines is None:
        cell_lines = ["A549", "K562", "MCF7"]

    print(cell_lines)

    predictions_dict = {}
    realdata_dict = {}
    ctrldata_dict = {}
    drug_r2 = {}

    for cell_drug_dose_comb, category_count in tqdm(
        zip(*np.unique(dataset.pert_categories, return_counts=True))
    ):

        if dataset.perturbation_key is None:
            break
        # estimate metrics only for reasonably-sized drug/cell-type combos
        if category_count <= 5:
            continue
        print(cell_drug_dose_comb, category_count)
        # doesn't make sense to evaluate DMSO (=control) as a perturbation
        if (
            "dmso" in cell_drug_dose_comb.lower()
            or "control" in cell_drug_dose_comb.lower()
        ):
            continue

        # dataset.var_names is the list of gene names
        # dataset.de_genes is a dict, containing a list of all differentiably-expressed
        # genes for every cell_drug_dose combination.
        
        bool_de = dataset.var_names.isin(
            np.array(dataset.de_genes[cell_drug_dose_comb])
        )
        idx_de = bool2idx(bool_de)

        # need at least two genes to be able to calc r2 score
        # if len(idx_de) < 2:
            # continue

        bool_category = pert_categories_index.get_loc(cell_drug_dose_comb)
        if type(bool_category) == type(0):
            idx_all = np.array([bool_category])
        else:
            idx_all = bool2idx(bool_category)

        idx = idx_all[0]
        y_true = dataset.genes[idx_all, :].to(device="cuda")

        if genes_control is None:
            n_obs = y_true.size(0)
        else:
            assert isinstance(genes_control, torch.Tensor)
            n_obs = genes_control.size(0)

        emb_covs = [repeat_n(cov[idx], n_obs) for cov in dataset.covariates]

        if dataset.dosages[idx] not in dosages:
            continue

        if dataset.use_drugs_idx:
            emb_drugs = (
                repeat_n(dataset.drugs_idx[idx], n_obs).squeeze(),
                repeat_n(dataset.dosages[idx], n_obs).squeeze(),
            )
        else:
            emb_drugs = repeat_n(dataset.drugs[idx], n_obs)

        # copies just the needed genes to GPU
        # Could try moving the whole genes tensor to GPU once for further speedups (but more memory problems)

        if genes_control is None:
            # print("Predicting AE alike.")
            mean_pred, _ = compute_prediction(
                model,
                y_true,
                emb_drugs,
                emb_covs,
            )
        else:
            # print("Predicting counterfactuals.")
            mean_pred, variance = compute_prediction(
                model,
                genes_control,
                emb_drugs,
                emb_covs,
            )

        if use_DEGs:
            mean_pred = mean_pred[:, idx_de]
       
        predictions_dict[cell_drug_dose_comb] = mean_pred
        realdata_dict[cell_drug_dose_comb] = y_true
        # ctrldata_dict[cell_drug_dose_comb] = ctrl_expre

    return realdata_dict, predictions_dict
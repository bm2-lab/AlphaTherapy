# AlphaTherapy

## Introduction

AlphaTherapy is a theoretical proof-of-concept AI framework for the rational design of sequential drug treatments for tumors across diverse drugs and tumor types.

Users can create a scenario by customizing input parameters tailored to their specific problem related to sequential therapy (Methods). Specifically, AlphaTherapy is designed to investigate sequential drug treatments applicable for different scenarios, including for a specified cell line (parameter: cell line), for a specified treatment duration (parameter: termination step) and for a constrained sequential drug combination space (parameter: drug pool). 

## Table of contents

## Requirements
* Python == 3.6.13
* torch == 1.10.2
* gym == 0.15.4
* rdkit == 2021.03.5
* openpyxl, numpy, pandas, scipy, matplotlib, tensorboard, scikit-learn Python modules are also needed
*  R == 4.3.1
*  clusterProfiler, org.Hs.eg.db, usefun R packages are also needed

## Installation from github

**1. Downloading AlphaTherapy**
```bash
git clone https://github.com/bm2-lab/AlphaTherapy.git
```
**2. Installing some packages**
```python
cd ./scripts/gym_cell_model
pip install -e .
cd ../tianshou-master
pip install -e .
```

## Installation from Docker image

```bash
docker pull xhchen/alphatherapy:latest
```

## Usage

### 1. Rational design of sequential drug treatments for tumors by AlphaTherapy

**This section will generate a list of effective sequential drug combinations for user-specified scenarios, sorted by episode reward. For illustration, we use a specific scenario with the following three key parameters:**

1. Drug Pool: A drug set containing 387 anticancer drugs approved by the United States Food and Drug Administration (FDA).
2. Cell Line: MCF7, a commonly used breast cancer cell line.
3. Terminal Step: Step 3, indicating the drug duration time.

#### 1.1 Drug pool construction

```python
python ./scripts/utils/build_drugset.py --input FDA_drugset.xlsx --output FDA_drugset.pickle
```

**The command facilitates the construction of a drug set by processing an input Excel file and outputing a pickle file for use in cell model environments.**

**Parameters**
* input:  the file name of an Excel file containing the compound information of the drug set. If users wish to create a custom drug set, they must store thier input file in ./data/ and ensure that the input Excel file contains the following five required columns.
  1. Index: The index of the compound.
  2. Name: The name of the compound.
  3. Target: The target(s) of the compound. Multiple targets should be separated by commas. If the target is unknown, use "Unknown".
  4. Pathway: The pathway associated with the compound. If the pathway is unknown, use "Unknown".
  5. SMILES: The SMILES (Simplified Molecular Input Line Entry System) notation of the compound. Ensure that the correct SMILES notation is provided.
* output: The output is the file name of a pickle file containing the processed compound information of the drug set in ./scripts/gym_cell_model/gym_cell_model/data directory.

#### 1.2 Write configurations

```
python ./scripts/utils/write_scenario_configs.py --drugset_file_name FDA_drugset.pickle --drugset_short_name FDA --cell_line MCF7 --terminal_step 3
```

**This command prepares configurations prior to running the RL agents by modifying two config files:**
  * env_cpd.config: This file contains configurations for the cell model environment located in /scripts/gym_cell_model/gym_cell_model/config/. The command will append configurations for ENV_[drugset_short_name]_[cell_line]_[terminal_step], like ENV_FDA_MCF7_3.
  * RL_agent.config: This file contains configurations for the RL agents located in /scripts/AlphaTherapy/config/. The command will append configurations for ENV_[drugset_short_name]_[cell_line]_[terminal_step]_SEED[i], where i ranges from 1 to 10, like ENV_FDA_MCF7_3_SEED1.

**Parameters**
  * drugset_file_name: the pickle file name of drug set which saved in the above command.
  * drugset_short_name: a phrase or a short name for easy reference of the drug set.
  * cell_line: the cell line you want to research.
  * terminal_step: a number of the terminal step you want to reearch.

Notably, we only add the information of the scenario to RL_agents.config. You also can modify the model training parameters in RL_agents.config.

#### 1.3 Run RL agents
```python
nohup python ./scripts/AlphaTherapy/model/train_RL_agents.py --drugset_short_name FDA --cell_line MCF7 --terminal_step 3 --max_workers 10 &
```

**This command trains RL agents through interaction with the cell model. It will generate "policy.pth" files and "plot_test_data.csv" files within new directories like "./scripts/AlphaTherapy/working_log/ENV_FDA_MCF7_STEP2_SEED1". These files serve as records for model training, capturing all generated .pth files during runtime and the fluctuation of episodes throughout the process.
**

**Parameters**
  * drugset_short_name, cell_line, terminal_step is the same as above.
  * max_workers: Maximum number of available CPUs.

#### 1.4 Merge results
```python
python ./scripts/AlphaTherapy/model/output_result.py --drugset_short_name FDA --cell_line MCF7 --terminal_step 3
```

**This command will organize the sequential drug combination results for each env_name like ENV_FDA_A375_STEP3 scenario and finally output a file named ENV_FDA_A375_STEP3.csv, stored in /output/AlphaTherapy/.**

**Parameters**
  * drugset_short_name, cell_line, terminal_step is the same as above.

### 2. Downstream analysis
```python
python ./scripts/downstream_analysis/src/run_downstram_analysis.py --combos_file_name ENV_FDA_A375_STEP3.csv --env_name ENV_FDA_MCF7_STEP3 --drugA_name Hydroxyurea --drugB_name "ixantrone Maleate"
```
**This command will run downstream analysis and save the results in file like in /output/downstream_analysis/."**


**Parameters**
  * combos_file_name: the file name of env_name.csv file generated in the previous command.
  * env_name: the config name of cell model, like ENV_FDA_A375_STEP3.
  * drugA_name: the name of the first frug in drug combos in the combos_file_name.csv.
  * drugB_name: the name of the first frug in drug combos in the combos_file_name.csv.

## Citation

## Contacts
bm2-lab@tongji.edu.cn

















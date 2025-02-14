import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Load data directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))
from path import DATA_DIR  

import configparser
import os

config = configparser.ConfigParser()

config['DEFAULT'] = {
    'seed': '3333',
    'hidden_size': '512',
    'device': 'cuda:0',
    'epoch': '500',
    'batch_size': '128',
    'learning_rate': '0.001',
    'input_file': '/the/training/data/path',
    'model_file': '/the/model/save/path',
    'predict_file': '/the/predict/save/path'
}

for i in range(10):
    section_name = f'random_{i}'
    config[section_name] = {
        'input_file': DATA_DIR / f'preprocessed_data_2025/state_transition_model_input_data/step2_model_training_dataset_random_split_{i}.pkl',
        'model_file': DATA_DIR / f'working_log_2025/state_transition_model/default_model/random_model_{i}.pth',
        'predict_file': DATA_DIR / f'working_log_2025/state_transition_model/default_model/random_predict_{i}.pkl'
    }

for i in range(10):
    section_name = f'chemical_{i}'
    config[section_name] = {
        'input_file': DATA_DIR / f'preprocessed_data_2025/state_transition_model_input_data/step2_model_training_dataset_chemical_blind_{i}.pkl',
        'model_file': DATA_DIR / f'working_log_2025/state_transition_model/default_model/chemical_blind_{i}.pth',
        'predict_file': DATA_DIR / f'working_log_2025/state_transition_model/default_model/chemical_blind_{i}.pkl'
    }

for i in range(10):
    section_name = f'cell_{i}'
    config[section_name] = {
        'input_file': DATA_DIR / f'preprocessed_data_2025/state_transition_model_input_data/step2_model_training_dataset_cell_blind_{i}.pkl',
        'model_file': DATA_DIR / f'working_log_2025/state_transition_model/default_model/cell_blind_{i}.pth',
        'predict_file': DATA_DIR / f'working_log_2025/state_transition_model/default_model/cell_blind_{i}.pkl'
    }

config['random_0_5uM'] = {
    'input_file': DATA_DIR / 'preprocessed_data_2025/state_transition_model_input_data/step2_model_training_dataset_random_split_5uM.pkl',
    'model_file': DATA_DIR / 'working_log_2025/state_transition_model/default_model/random_model_5uM.pth',
    'predict_file': DATA_DIR / 'working_log_2025/state_transition_model/default_model/random_predict_5uM.pkl'
}

config_file_path = 'model.config'
with open(config_file_path, 'w') as configfile:
    config.write(configfile)



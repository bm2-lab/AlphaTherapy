import os
import pickle
import argparse
import numpy as np
import pandas as pd
from configparser import ConfigParser
import argparse

# 1. The parameters of input
argparser = argparse.ArgumentParser()
argparser.add_argument('--drugset_file_name', type=str, help='the data file name of drugset (*.pickle)', default="FDA_drugset.pickle")
argparser.add_argument('--drugset_short_name', type=str, help='a phrase or a short name for easy reference', default="FDA")
argparser.add_argument('--cell_line', nargs='+', type=str, help='cell line(s)', default="MCF7")
argparser.add_argument('--terminal_step', nargs='+', type=int, help='terminal step(s)', default="2")

args = argparser.parse_args()

if not isinstance(args.cell_line, list):
    args.cell_line = [args.cell_line]

if not isinstance(args.terminal_step, list):
    args.terminal_step = [args.terminal_step]

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 2. write_config_for_cell_model

cell_model_config_file_path = project_dir + "/scripts/gym_cell_model/gym_cell_model/config/env_cpd.config"

config = ConfigParser()
config.read(cell_model_config_file_path, encoding='UTF-8')

for cell_line in args.cell_line: 
    for terminal_step in args.terminal_step:
        env_name = 'ENV_%s_%s_STEP%d' % (args.drugset_short_name, cell_line, terminal_step)

        if env_name in config:
            print(f"{env_name} existed! Now the {env_name} will be overwrite!")
            config.remove_section(env_name)

        config.add_section(section=env_name)
        config[env_name]['cell_line'] = cell_line
        config[env_name]['max_step_number'] = str(terminal_step)
        config[env_name]['drugset_file'] = args.drugset_file_name

fo = open(cell_model_config_file_path, 'w', encoding='UTF-8')
config.write(fo)  
fo.close()

# 3. write_config_for_RL_agents

RL_agents_config_file_path = project_dir + "/scripts/AlphaTherapy/config/RL_agent.config"

config = ConfigParser()
config.read(RL_agents_config_file_path, encoding='UTF-8')

for cell_line in args.cell_line: 
    for terminal_step in args.terminal_step:
        env_name = 'ENV_%s_%s_STEP%d' % (args.drugset_short_name, cell_line, terminal_step)

        for seed in range(1, 11):
            config_name = env_name + "_SEED" + str(seed)

            if config_name in config:
                print(f"{config_name} existed! Now the {config_name} will be overwrite!")
                config.remove_section(config_name)
                
            config.add_section(section=config_name)
            config[config_name]['env name'] = env_name
            config[config_name]['rl seed'] = str(seed)
            config[config_name]['return n steps'] = str((terminal_step+1))
            if terminal_step <= 3:
                config[config_name]['max epoch'] = str(10000)
            elif terminal_step > 3:
                config[config_name]['max epoch'] = str(30000)
            elif terminal_step > 6:
                config[config_name]['max epoch'] = str(100000)

fo = open(RL_agents_config_file_path, 'w', encoding='UTF-8')
config.write(fo)
fo.close()

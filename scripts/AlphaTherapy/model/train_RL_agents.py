import os
import sys
import time
import pickle
import argparse
import subprocess
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# 1. The parameters of input
argparser = argparse.ArgumentParser()
argparser.add_argument('--drugset_short_name', type=str, help='a phrase or a short name for easy reference', default="FDA")
argparser.add_argument('--cell_line', nargs='+', type=str, help='cell line(s)', default="MCF7")
argparser.add_argument('--terminal_step', nargs='+', type=int, help='terminal step(s)', default="2")
argparser.add_argument('--max_workers', type=int, help='cpu numbers', default="5")

args = argparser.parse_args()

if not isinstance(args.cell_line, list):
    args.cell_line = [args.cell_line]

if not isinstance(args.terminal_step, list):
    args.terminal_step = [args.terminal_step]

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_script(config_name):
    start_time = time.time()
    script_file_path = project_dir + "/scripts/AlphaTherapy/model/RL_agent.py"
    cmd = f"python {script_file_path} {config_name}"
    result = subprocess.run(cmd, shell=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return config_name, result.returncode, elapsed_time

def main(config_names, max_workers):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_config = {executor.submit(run_script, config_name): config_name for config_name in config_names}

        for future in as_completed(future_to_config):
            config_name = future_to_config[future]
            try:
                config_name, return_code, elapsed_time = future.result()
                print(f"{config_name} finished with return code {return_code} in {elapsed_time:.2f} seconds")
            except Exception as exc:
                print(f"{config_name} generated an exception: {exc}")


config_names = []
for cell_line in args.cell_line: 
    for terminal_step in args.terminal_step:
        env_name = 'ENV_%s_%s_STEP%d' % (args.drugset_short_name, cell_line, terminal_step)
        for seed in range(1, 11):
            config_name = env_name + "_SEED" + str(seed)
            config_names.append(config_name)

start_time = time.time()
main(config_names, args.max_workers)
end_time = time.time()
print(f"All tasks completed in {end_time - start_time:.2f} seconds")



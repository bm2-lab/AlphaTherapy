
import gym
import pickle
import numpy as np
import pandas as pd

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.resolve().parent
sys.path.append(str(ROOT_DIR))
from path import DATA_DIR 

from downstream_utils import expression_simulate


if __name__ == "__main__":

    all_ccls = [sys.argv[1]]

    for cell_line in all_ccls:
        print(cell_line)
        
        file_path = DATA_DIR / f"preprocessed_data_2025/down_stream_analysis/plans/{cell_line}.csv"
        total_res = pd.read_csv(file_path)
        
        drugA_index_arr = np.unique(total_res["first_ind"])

        env_name = "ENV_FDA_%s_STEP9" % cell_line
        env = gym.make('gym_cell_model:ccl-env-cpd-v1', env_name=env_name)

        cnt = 0
        for drugA_index in drugA_index_arr:
            print(cnt, len(drugA_index_arr))
            cnt += 1
            expression_simulate(cell_line, drugA_index, env)
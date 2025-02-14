import gym
import numpy as np

import os
import sys
from pathlib import Path

# Set root and data directories
ROOT_DIR = Path(os.getcwd()).resolve().parent
sys.path.append(str(ROOT_DIR))
from path import DATA_DIR

def simulate_by_actions(action_ls, env):

    action_ls = np.array(action_ls)

    if len(np.unique(action_ls)) == 1:
        first_drug = action_ls[0]
        first_len = np.sum(action_ls == first_drug)
        env.reset()
        for _ in range(first_len):
            env.step(first_drug)
        return env.cv_ls

    first_drug = action_ls[0]
    second_drug = action_ls[-1]
    first_len = np.sum(action_ls == first_drug)
    second_len = np.sum(action_ls == second_drug)

    env.reset()
    for _ in range(first_len):
        env.step(first_drug)

    

    if second_len <= 1:
        env.step(second_drug)
    else:
        env.step(second_drug)
        for _ in range(second_len-1):
            env.step(0)
    return env.cv_ls

env_name = "ENV_FDA_MCF7_STEP9"
env = gym.make('gym_cell_model:ccl-env-cpd-v1', env_name=env_name)

drugA_index = 373   # Erlotinib
drugB_index = 41    # Doxorubicin
drug_regimen = [drugA_index] * 4 + [drugB_index] * 5

drugAB_cv_ls = simulate_by_actions(drug_regimen, env)
drugA_cv_ls = simulate_by_actions([drugA_index]*9, env)
drugB_cv_ls = simulate_by_actions([drugB_index]*9, env)

import pickle
with open(DATA_DIR / f"result_2025/AlphaTherapy/example_cv_data.pkl", "wb") as f:
    pickle.dump([drugAB_cv_ls, drugA_cv_ls, drugB_cv_ls], f)
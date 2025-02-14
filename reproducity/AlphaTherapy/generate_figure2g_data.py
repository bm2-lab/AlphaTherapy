import gym
import numpy as np
import pandas as pd


import os
import sys
from pathlib import Path

# Set root and data directories
ROOT_DIR = Path(os.getcwd()).resolve().parent
sys.path.append(str(ROOT_DIR))
from path import DATA_DIR
from utils import sample_based_pcc

print("ROOT_DIR:", ROOT_DIR)
print("DATA_DIR:", DATA_DIR)


def generate_drug_regimen(n_drugs, max_steps):
    drug1, drug2 = np.random.choice(n_drugs, size=2, replace=False)

    # Determine the step count for drug1
    drug1_steps = np.random.randint(1, max_steps)  # At least 1 step for drug1

    # Construct the regimen
    regimen = [drug1] * drug1_steps + [drug2] * (max_steps - drug1_steps)

    return regimen


def simulate_by_actions(action_ls, env):

    action_ls = np.array(action_ls)

    if len(np.unique(action_ls)) == 1:
        print("Not sequential drug combination")
        return None

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
    return env.cv_ls[0] - env.cv_ls[-1]

def generate_random_data(env_name, data_point_number=1000, output_file="output.csv"):
    """
    Generate random drug regimen data and calculate efficacy and synergy rewards.

    Args:
        env_name (str): Name of the environment.
        data_point_number (int): Number of data points to generate.
        output_file (str): File path to save the resulting DataFrame.
    """
    # Initialize environment
    env = gym.make('gym_cell_model:ccl-env-cpd-v1', env_name=env_name)
    drug_number = env.action_space.n

    # Initialize arrays to store rewards
    Efficacy_rewards = np.zeros(data_point_number)
    Synergy_rewards = np.zeros(data_point_number)

    for i in range(data_point_number):
        # Generate a random drug regimen

        drug_regimen = generate_drug_regimen(drug_number, env.max_step_number)
        first_drug = int(drug_regimen[0])
        second_drug = int(drug_regimen[-1])

        # Calculate rewards
        AA_reward = env.single_episode_rewards[first_drug]  # Reward for using first drug alone
        BB_reward = env.single_episode_rewards[second_drug]  # Reward for using second drug alone
        AB_reward = simulate_by_actions(drug_regimen, env)  # Reward for using the regimen as is

        # Efficacy reward
        Efficacy_rewards[i] = AB_reward

        # Synergy reward
        Synergy_rewards[i] = AB_reward - np.max([AA_reward, BB_reward])

    # Combine results into a DataFrame
    results_df = pd.DataFrame({
        "Efficacy_reward": Efficacy_rewards,
        "Synergy_reward": Synergy_rewards
    })

    # Save DataFrame to CSV
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


for i in range(2,10):
    env_name = f"ENV_FDA_MCF7_STEP{i}"
    output_file = DATA_DIR / f"result_2025/AlphaTherapy/{env_name}_random_sequential_data.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True) 
    generate_random_data(env_name=env_name, data_point_number=1000, output_file=output_file)

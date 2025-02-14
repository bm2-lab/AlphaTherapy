import os
import sys
import gym
import time
import shutil
import torch
import numpy as np
import pandas as pd
from configparser import ConfigParser
np.set_printoptions(precision=3)

from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer
from tianshou.utils.net.common import Net

from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings('ignore')

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# read model configurations

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


# Organize the output results
def output_aggregator(model_config_name):
    config = ConfigParser()
    config.read(project_dir + "/scripts/AlphaTherapy/config/RL_agent.config", encoding="UTF-8")

    env_name = config[model_config_name]['env name']
    test_episode_num = config.getint(model_config_name, 'test episode num')

    epsilon_train = config.getfloat(model_config_name, 'eps train')
    epsilon_test = config.getfloat(model_config_name, 'eps test')
    gamma = config.getfloat(model_config_name, 'gamma')
    return_n_steps = config.getint(model_config_name, 'return n steps')

    max_epoch = config.getint(model_config_name, 'max epoch')
    step_per_epoch = config.getint(model_config_name, 'step per epoch')
    collect_per_step = config.getint(model_config_name, 'collect per step')
    batch_size = config.getint(model_config_name, 'batch size')
    target_update_freq = config.getint(model_config_name, 'target update freq')
    layer_num = config.getint(model_config_name, 'layer num')
    buffer_size = config.getint(model_config_name, 'buffer size')

    lr = config.getfloat(model_config_name, 'lr')
    device = config.get(model_config_name, 'device')
    rl_seed = config.getint(model_config_name, 'rl seed')

    np.random.seed(rl_seed)  # Numpy module.
    torch.manual_seed(rl_seed)     # 
    torch.cuda.manual_seed(rl_seed)    # 

    env = gym.make('gym_cell_model:ccl-env-cpd-v1', env_name=env_name)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    # network model
    net = Net(layer_num, state_shape, action_shape, device).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    policy = DQNPolicy(
        net, optim, gamma, return_n_steps,
        use_target_network=target_update_freq > 0,
        target_update_freq=target_update_freq)

    # log
    child_path = model_config_name
    log_path = project_dir + "/scripts/AlphaTherapy/working_log/" + child_path

    drug_regimens = []
    AB_rewards = np.zeros(10)
    BA_rewards = np.zeros(10)
    AA_rewards = np.zeros(10)
    BB_rewards = np.zeros(10)
    synergy_scores = np.zeros(10)
    first_drugs = np.zeros(10, dtype=np.int16)
    second_drugs = np.zeros(10, dtype=np.int16)
    first_lens = np.zeros(10, dtype=np.int16)
    second_lens = np.zeros(10, dtype=np.int16)
    
    for i in range(0, 10):
        best_policy = torch.load(os.path.join(log_path, 'policy%d.pth'%i))
        result_collector = Collector(best_policy, env)
        best_policy.eval()
        best_policy.set_eps(epsilon_test)
        res = result_collector.collect_one_best(n_episode=test_episode_num, render=True)
        first_drugs[i] = int(env.action_ls[0])
        second_drugs[i] = int(env.action_ls[-1])
        
        first_lens[i] = np.sum(np.array(env.action_ls) == first_drugs[i])
        second_lens[i] = np.sum(np.array(env.action_ls) == second_drugs[i])
        AB_rewards[i] = simulate_by_actions(env.action_ls, env)
        BA_rewards[i] = simulate_by_actions(env.action_ls[::-1], env)
        AA_rewards[i] = env.single_episode_rewards[first_drugs[i]]
        BB_rewards[i] = env.single_episode_rewards[second_drugs[i]]
        synergy_scores[i] = AB_rewards[i] - np.max([AA_rewards[i], BB_rewards[i]])

    plan = pd.DataFrame({"first_ind": first_drugs, "second_ind": second_drugs, "first_drugs":env.init_drug_info[first_drugs,2], "second_drugs":env.init_drug_info[second_drugs,2], "AB_rewards":AB_rewards, "BA_rewards":BA_rewards, "AA_rewards":AA_rewards, "BB_rewards":BB_rewards, "synergy_scores":synergy_scores, "first_len": first_lens, "second_len": second_lens})
    return(plan)


import argparse 

# The parameters of input
argparser = argparse.ArgumentParser()
argparser.add_argument('--drugset_short_name', type=str, help='a phrase or a short name for easy reference', default="FDA")
argparser.add_argument('--cell_line', nargs='+', type=str, help='cell line(s)', default="MCF7")
argparser.add_argument('--terminal_step', nargs='+', type=int, help='terminal step(s)', default="2")

args = argparser.parse_args()

if not isinstance(args.cell_line, list):
    args.cell_line = [args.cell_line]

if not isinstance(args.terminal_step, list):
    args.terminal_step = [args.terminal_step]

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

drug_info = pd.read_excel(project_dir +  "/data/400_drugs_MoA.xlsx", engine="openpyxl")

for cell_line in args.cell_line: 
    for terminal_step in args.terminal_step:
        plans = []
        env_name = 'ENV_%s_%s_STEP%d' % (args.drugset_short_name, cell_line, terminal_step)
        for seed in range(1, 11):
            config_name = env_name + "_SEED" + str(seed)
            print(config_name)
            plan = output_aggregator(config_name)
            plans.append(plan)

        res = pd.concat(plans)
        res = res.sort_values("synergy_scores", ascending=False)
        res = res.drop_duplicates()

        res = pd.merge(res, drug_info, left_on='first_drugs', right_on='Name', how = "left")
        res = pd.merge(res, drug_info, left_on='second_drugs', right_on='Name', how = "left")

        res = res.loc[:, ['first_ind', 'second_ind', 'first_drugs', 'second_drugs', 'AB_rewards', 'BA_rewards', 'AA_rewards', 'BB_rewards', 'synergy_scores', 'MoA_x', 'MoA_y', 'first_len', 'second_len']]
        res.columns = ['first_ind', 'second_ind', 'first_drug', 'second_drug', 'AB_score', 'BA_score', 'AA_score', 'BB_score', 'synergy_score', 'first_moa', 'second_moa', 'first_len', 'second_len']

        res.to_csv(project_dir + "/output/AlphaTherapy/" + "%s.csv" % env_name)

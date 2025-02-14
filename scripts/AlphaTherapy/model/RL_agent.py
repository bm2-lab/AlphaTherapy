import os
import sys
import gym
import time
import shutil
import torch
import numpy as np
import pandas as pd
from configparser import ConfigParser

from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer
from tianshou.utils.net.common import Net

from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings('ignore')

# 1. read model configurations

start_time = time.time()

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
model_config_name = sys.argv[1]
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

# environment initialization

np.random.seed(rl_seed)  # Numpy module.
torch.manual_seed(rl_seed)     
torch.cuda.manual_seed(rl_seed)    

train_envs = gym.make('gym_cell_model:ccl-env-cpd-v1', env_name=env_name)
test_envs = gym.make('gym_cell_model:ccl-env-cpd-v1', env_name=env_name)
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

# collector
train_collector = Collector(policy, train_envs, ReplayBuffer(buffer_size))
test_collector = Collector(policy, test_envs)
train_collector.collect(n_step=batch_size)

# log
child_path = model_config_name
log_path = project_dir + "/scripts/AlphaTherapy/working_log/" + child_path

if os.path.exists(log_path):
    shutil.rmtree(log_path)
os.makedirs(log_path)


def save_fn(policy):
    torch.save(policy, os.path.join(log_path, 'policy.pth'))


def save_fn(policy, rank_ind):
    save_policy_name = os.path.join(log_path, 'policy%d.pth'%rank_ind)
    if os.path.exists(save_policy_name):
        for file_ind in range(9,rank_ind-1,-1):
            if os.path.exists(os.path.join(log_path, 'policy%d.pth' % file_ind)):
                os.rename(os.path.join(log_path, 'policy%d.pth' % file_ind), os.path.join(log_path, 'policy%d.pth'%(file_ind+1)))
    torch.save(policy, save_policy_name)


def stop_fn(x):
    return x >= 1000.0


def train_fn(x):
    policy.set_eps((1-x/max_epoch) * epsilon_train)


def test_fn(x):
    policy.set_eps(epsilon_test)



# RL agent training
writer = SummaryWriter(log_dir= log_path)

plot_train_data, plot_test_data, _ = offpolicy_trainer(
    policy, train_collector, test_collector, max_epoch,
    step_per_epoch, collect_per_step, test_episode_num,
    batch_size, train_fn=train_fn, test_fn=test_fn,
    stop_fn=stop_fn, save_fn=save_fn, writer=writer)

plot_train_data.to_csv(os.path.join(log_path, 'plot_train_data.csv'))
plot_test_data.to_csv(os.path.join(log_path, 'plot_test_data.csv'))

train_collector.close()
test_collector.close()

# show results.

for i in range(0, 10):
    best_policy = torch.load(os.path.join(log_path, 'policy%d.pth'%i))
    result_collector = Collector(best_policy, env)
    best_policy.eval()
    best_policy.set_eps(epsilon_test)
    res = result_collector.collect_one_best(n_episode=test_episode_num, render=True)

cost_time = time.time()-start_time
print("%.2f S" % cost_time)

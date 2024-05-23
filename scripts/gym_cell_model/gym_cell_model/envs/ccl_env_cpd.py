import gym
from gym import spaces

import os
import sys
import torch
import pickle
import numpy as np
import pandas as pd
from configparser import ConfigParser

env_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(env_dir)
from model import STATE_TRANSITION


class CCLEnvCPD(gym.Env):
    def __init__(self, env_name):
        super(CCLEnvCPD, self).__init__()

        self.current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # 1. read configurations
        config = ConfigParser()
        config.read(self.current_dir + "/config/env_cpd.config", encoding="UTF-8")
        self.cell_line = config[env_name].get('cell_line')
        self.drugset_file = config[env_name].get('drugset_file')
        self.drugset_file = self.current_dir + "/data/" + self.drugset_file
        self.max_step_number = config[env_name].getint('max_step_number')
        self.model_file = config[env_name].get('model_file')
        self.model_file = self.current_dir + "/data/" + self.model_file


        # 2. load models and information
        # 2.1 Drug info [index, BRD, drug_name, SMILES]
        with open(self.drugset_file, 'rb') as f:
            self.drug_info = pickle.load(f)
        self.init_drug_info = self.drug_info    # self.init_drug_info is for reset()

        # 2.2 State transition model
        self.StateTransition = torch.load(self.model_file,
                                          map_location="cpu")
        with open(self.current_dir + "/data/dataset_scaler.pkl", "rb") as f:
            self.x_scaler, self.y_scaler = pickle.load(f)

        # 2.3 Cell viability model
        with open(self.current_dir + "/data/cv_model.pkl", "rb") as f:
            self.CellViabModel = pickle.load(f)

        # 2.4 Initial state
        self.initail_state = self.generate_env_data()

        # 3. Reset
        self.reset()
        

        # 4. spaces
        self.action_space = spaces.Discrete(len(self.drug_info))
        self.observation_space = spaces.Box(
            low=np.float32("-inf"),
            high=np.float32("inf"),
            shape=(self.initail_state.shape[1], ))

        self.single_episode_rewards = self.cal_single_episode_rewards()
        self.episode_rewards_thre = np.percentile(list(self.single_episode_rewards.values()), 50)


    def reset(self):

        self.cur_state = self.initail_state
        self.drug_info = self.init_drug_info
        self.action_space = spaces.Discrete(len(self.drug_info))

        # render record variables
        self.cv = 1.0
        self.env_step = 0
        self.done = False
        self.diff_score = 0

        self.action_ls = []
        self.cv_ls = [self.cv]
        self.expression_ls = []
        self.expression_ls.append(self.cur_state)

        return self.cur_state


    def reset_best(self):
        # for show the last trajectory with record variables saved [cooperate with tianshou script]

        self.cur_state = self.initail_state
        return self.cur_state


    def step(self, action):
        obs = self._observe(action)
        reward, self.done = self._reward()
        info = {"diff": self.diff_score, "eff": (self.cv_ls[0]-self.cv_ls[-1])}
        self.cur_state = self.next_state

        return obs, reward, self.done, info


    def single_action_step(self, action):

        drug = self.drug_info[action][3].reshape(1, -1)  # (1, 166)
        self.cur_state_ = self.x_scaler.transform(self.cur_state).reshape(1, -1)  # (1, 978)
        self.delta_state_ = self.StateTransition.predict(
            drug, self.cur_state_)
        self.delta_state = self.y_scaler.inverse_transform(self.delta_state_)
        self.next_state = self.cur_state + self.delta_state
        self.next_state = self.next_state.reshape(1, -1)  # (1, 978)

        # update record variables
        self.env_step += 1
        self.action_ls.append(action)
        self.expression_ls.append(self.next_state)

        delta_cv = self.CellViabModel.predict(self.delta_state.reshape(-1, 978))[0]
        reward = self.cv - self.cv * delta_cv

        self.cv = delta_cv * self.cv
        self.cv_ls.append(self.cv)

        self.done = self.env_step >= self.max_step_number

        info = {}
        self.cur_state = self.next_state

        return self.next_state, reward, self.done, info


    def _observe(self, action):
        # update: self.delta_state, self.next_state
        # update: self.action_ls, (self.drug_info, self.action_space)
        # update: self.env_step, self.expression_ls

        # State ---(State transition model)---> Next State
        drug = self.drug_info[action][3].reshape(1, -1)  # (1, 166)
        self.cur_state_ = self.x_scaler.transform(self.cur_state).reshape(1, -1)  # (1, 978)
        self.delta_state_ = self.StateTransition.predict(
            drug, self.cur_state_)
        self.delta_state = self.y_scaler.inverse_transform(self.delta_state_)
        self.next_state = self.cur_state + self.delta_state
        self.next_state = self.next_state.reshape(1, -1)  # (1, 978)

        if self.action_space.n == 1:
            self.action_ls.append(self.action_ls[-1])
        else:
            self.action_ls.append(action)
        if len(np.unique(self.action_ls)) == 2:
            self.drug_info = self.drug_info[action].reshape(1, -1)
            self.action_space = spaces.Discrete(len(self.drug_info))

        # update record variables
        self.env_step += 1
        self.expression_ls.append(self.next_state)

        return self.next_state


    def _reward(self):
        # update: self.cv, self.cv_ls
        delta_cv = self.CellViabModel.predict(self.delta_state.reshape(-1, 978))[0]
        reward = self.cv - self.cv * delta_cv

        self.cv = delta_cv * self.cv
        self.cv_ls.append(self.cv)
        done = self.env_step >= self.max_step_number

        if done:
            self.diff_score = 0.0
            # if episode includes 2 actions, calculate synergy score
            if len(np.unique(self.action_ls)) != 1:
                episode_r = self.cv_ls[0] - self.cv_ls[-1]
                self.diff_score = episode_r - np.max([self.single_episode_rewards[self.action_ls[0]], self.single_episode_rewards[self.action_ls[-1]]])
                if episode_r > self.episode_rewards_thre:
                    return (reward+(1-episode_r)+self.diff_score), done
                    # return ((reward+(1-episode_r)) + self.reward_weight * self.diff_score), done

                else:
                    return (reward+self.diff_score), done

        return reward, done


    def seed(self, seed):
        np.random.seed(seed)


    def render(self):
        # show trajectory of one episode
        # cv0 [a0] cv1 [a1] cv2 [a3] cv3
        self.pathway_ls = [self.init_drug_info[i, 5] for i in self.action_ls]
        self.target_ls = [self.init_drug_info[i, 4] for i in self.action_ls]
        if self.done:
            print("\n")
            print("==========================================================")
            line = "Episode reward: %.3f\n" % (self.cv_ls[0] - self.cv_ls[-1])
            line += "Cell viability trajectory:\t" + " -> ".join(
                str(round(i, 3)) for i in self.cv_ls) + "\n"
            line += "Actions trajectory      :\t" + " -> ".join(
                str(i) for i in self.action_ls) + "\n"
            line += "Pathway trajectory      :\t" + " -> ".join(
                str(i) for i in self.pathway_ls) + "\n"
            print(line)
            print(self.target_ls)
        return None


    def generate_env_data(self):
        # generate expression data of initial state.
        env_ccl_init_expression_file = self.current_dir + "/data/env_ccl_init_expression.csv"
        env_ccl_init_expression = pd.read_table(env_ccl_init_expression_file, sep="\t", index_col=0, header=None)
        init_expression = env_ccl_init_expression.loc[self.cell_line, :].values
        return init_expression.reshape(1, 978)


    def cal_single_episode_rewards(self):
        # calculate episode rewards for only single drug administration.
        single_episode_rewards = {}
        for a in range(self.action_space.n):
            self.reset()
            for _ in range(self.max_step_number):
                self.single_action_step(a)
            single_episode_rewards[a] = self.cv_ls[0] - self.cv_ls[-1]
        return single_episode_rewards

import gym
import time
from gym.spaces.discrete import Discrete


class MyTestEnv(gym.Env):
    """This is a "going right" task. The task is to go right ``size`` steps.
    """

    def __init__(self, size, sleep=0, dict_state=False, ma_rew=0):
        self.size = size
        self.sleep = sleep
        self.dict_state = dict_state
        self.ma_rew = ma_rew
        self.action_space = Discrete(2)
        self.reset()

    def reset(self, state=0):
        self.done = False
        self.index = state
        return self._get_dict_state()

    def _get_reward(self):
        """Generate a non-scalar reward if ma_rew is True."""
        x = int(self.done)
        if self.ma_rew > 0:
            return [x] * self.ma_rew
        return x

    def _get_dict_state(self):
        """Generate a dict_state if dict_state is True."""
        return {'index': self.index} if self.dict_state else self.index

    def step(self, action):
        if self.done:
            raise ValueError('step after done !!!')
        if self.sleep > 0:
            time.sleep(self.sleep)
        if self.index == self.size:
            self.done = True
            return self._get_dict_state(), self._get_reward(), self.done, {}
        if action == 0:
            self.index = max(self.index - 1, 0)
            return self._get_dict_state(), self._get_reward(), self.done, \
                {'key': 1, 'env': self} if self.dict_state else {}
        elif action == 1:
            self.index += 1
            self.done = self.index == self.size
            return self._get_dict_state(), self._get_reward(), \
                self.done, {'key': 1, 'env': self}

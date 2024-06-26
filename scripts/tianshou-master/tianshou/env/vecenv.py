import gym
import numpy as np
from abc import ABC, abstractmethod
# modify: multiprocessing-> torch.multiprocessing
# torch.multiprocessing is an alternative to Python's multiprocessing module. It supports the same operations but extends it so that all tensors sent via multiprocessing.Queue have their data moved to shared memory, and only a handle is sent to other processes.
from torch.multiprocessing import Process, Pipe
from typing import List, Tuple, Union, Optional, Callable, Any

try:
    import ray
except ImportError:
    pass

from tianshou.env.utils import CloudpickleWrapper


class BaseVectorEnv(ABC, gym.Env):
    """Base class for vectorized environments wrapper. Usage:
    ::

        env_num = 8
        envs = VectorEnv([lambda: gym.make(task) for _ in range(env_num)])
        assert len(envs) == env_num

    It accepts a list of environment generators. In other words, an environment
    generator ``efn`` of a specific task means that ``efn()`` returns the
    environment of the given task, for example, ``gym.make(task)``.

    All of the VectorEnv must inherit :class:`~tianshou.env.BaseVectorEnv`.
    Here are some other usages:
    ::

        envs.seed(2)  # which is equal to the next line
        envs.seed([2, 3, 4, 5, 6, 7, 8, 9])  # set specific seed for each env
        obs = envs.reset()  # reset all environments
        obs = envs.reset([0, 5, 7])  # reset 3 specific environments
        obs, rew, done, info = envs.step([1] * 8)  # step synchronously
        envs.render()  # render all environments
        envs.close()  # close all environments
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]) -> None:
        self._env_fns = env_fns
        self.env_num = len(env_fns)
        self._obs = None
        self._rew = None
        self._done = None
        self._info = None

    def __len__(self) -> int:
        """Return len(self), which is the number of environments."""
        return self.env_num

    def __getattribute__(self, key):
        """Switch between the default attribute getter or one
           looking at wrapped environment level depending on the key."""
        if key not in ('observation_space', 'action_space'):
            return super().__getattribute__(key)
        else:
            return self.__getattr__(key)

    @abstractmethod
    def __getattr__(self, key):
        """Try to retrieve an attribute from each individual wrapped
           environment, if it does not belong to the wrapping vector
           environment class."""
        pass

    @abstractmethod
    # modify: Add an `iter` parameter to the `reset` method in the `env` class (where `reset_cycle` is the parameter for resetting in the environment).
    def reset(self, id: Optional[Union[int, List[int]]] = None, iter = False):
        """Reset the state of all the environments and return initial
        observations if id is ``None``, otherwise reset the specific
        environments with given id, either an int or a list.
        """
        pass

    @abstractmethod
    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run one timestep of all the environments’ dynamics. When the end of
        episode is reached, you are responsible for calling reset(id) to reset
        this environment’s state.

        Accept a batch of action and return a tuple (obs, rew, done, info).

        :param numpy.ndarray action: a batch of action provided by the agent.

        :return: A tuple including four items:

            * ``obs`` a numpy.ndarray, the agent's observation of current \
                environments
            * ``rew`` a numpy.ndarray, the amount of rewards returned after \
                previous actions
            * ``done`` a numpy.ndarray, whether these episodes have ended, in \
                which case further step() calls will return undefined results
            * ``info`` a numpy.ndarray, contains auxiliary diagnostic \
                information (helpful for debugging, and sometimes learning)
        """
        pass

    @abstractmethod
    def seed(self, seed: Optional[Union[int, List[int]]] = None) -> List[int]:
        """Set the seed for all environments.

        Accept ``None``, an int (which will extend ``i`` to
        ``[i, i + 1, i + 2, ...]``) or a list.

        :return: The list of seeds used in this env's random number \
        generators. The first value in the list should be the "main" seed, or \
        the value which a reproducer pass to "seed".
        """
        pass

    @abstractmethod
    def render(self, **kwargs) -> None:
        """Render all of the environments."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close all of the environments.

        Environments will automatically close() themselves when garbage
        collected or when the program exits.
        """
        pass


class VectorEnv(BaseVectorEnv):
    """Dummy vectorized environment wrapper, implemented in for-loop.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for more detailed
        explanation.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]) -> None:
        super().__init__(env_fns)
        self.envs = env_fns

    def __getattr__(self, key):
        return [getattr(env, key) if hasattr(env, key) else None
                for env in self.envs]

    # modify: Add an `iter` parameter to the `reset` method in the `env` class (where `reset_cycle` is the parameter for resetting in the environment).
    def reset(self, id: Optional[Union[int, List[int]]] = None, iter = False) -> np.ndarray: 

        if id is None:
            self._obs = np.stack([e.reset(reset_cycle=iter) for e in self.envs])
        else:
            if np.isscalar(id):
                id = [id]
            for i in id:
                self._obs[i] = self.envs[i].reset(reset_cycle=iter)
        return self._obs

    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert len(action) == self.env_num
        result = [e.step(a) for e, a in zip(self.envs, action)]
        self._obs, self._rew, self._done, self._info = zip(*result)
        self._obs = np.stack(self._obs)
        self._rew = np.stack(self._rew)
        self._done = np.stack(self._done)
        self._info = np.stack(self._info)
        return self._obs, self._rew, self._done, self._info

    def seed(self, seed: Optional[Union[int, List[int]]] = None) -> List[int]:
        if np.isscalar(seed):
            seed = [seed + _ for _ in range(self.env_num)]
        elif seed is None:
            seed = [seed] * self.env_num
        result = []
        for e, s in zip(self.envs, seed):
            if hasattr(e, 'seed'):
                result.append(e.seed(s))
        return result

    def render(self, **kwargs) -> List[Any]:
        result = []
        for e in self.envs:
            if hasattr(e, 'render'):
                result.append(e.render(**kwargs))
        return result

    def close(self) -> List[Any]:
        return [e.close() for e in self.envs]


def worker(parent, p, env):
    parent.close()
    try:
        while True:
            cmd, data = p.recv()
            if cmd == 'step':
                p.send(env.step(data))
            # modify: Add an `iter` parameter to the `reset` method in the `env` class (where `reset_cycle` is the parameter for resetting in the environment).
            elif cmd == 'reset':
                p.send(env.reset(reset_cycle=data))
            # modify: add reset_best
            elif cmd == 'reset_best':
                p.send(env.reset_best())
            # modify: add save_trajectory_data
            elif cmd == 'save_trajectory_data':
                p.send(env.save_trajectory_data(data))
            elif cmd == 'close':
                p.send(env.close())
                p.close()
                break
            elif cmd == 'render':
                p.send(env.render(**data) if hasattr(env, 'render') else None)
            elif cmd == 'seed':
                p.send(env.seed(data) if hasattr(env, 'seed') else None)
            elif cmd == 'getattr':
                p.send(getattr(env, data) if hasattr(env, data) else None)
            # modify: add set_drugs
            elif cmd == 'set_drugs':
                p.send(env.set_drugs(data) if hasattr(env, 'set_drugs') else None)
            # modify: add set_initial_state 
            elif cmd == 'set_initial_state':
                p.send(env.set_initial_state(data) if hasattr(env, 'set_drugs') else None)
            else:
                p.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        p.close()


class SubprocVectorEnv(BaseVectorEnv):
    """Vectorized environment wrapper based on subprocess.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for more detailed
        explanation.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]) -> None:
        super().__init__(env_fns)
        # modify: Pass the environment directly.
        self.envs = env_fns
        self.closed = False
        # 1. Create 5 pipes for communication between the main process and the child processes.
        self.parent_remote, self.child_remote = \
            zip(*[Pipe() for _ in range(self.env_num)])
        # 2. 
        # modify: 
        self.processes = [
            Process(target=worker, args=(
                parent, child, env), daemon=True)
            for (parent, child, env) in zip(
                self.parent_remote, self.child_remote, self.envs)
        ]
        # 3. 
        for p in self.processes:
            p.start()
        # 4. 
        for c in self.child_remote:
            c.close()

    def __getattr__(self, key):
        for p in self.parent_remote:
            p.send(['getattr', key])
        return [p.recv() for p in self.parent_remote]

    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert len(action) == self.env_num
        for p, a in zip(self.parent_remote, action):
            p.send(['step', a])
        result = [p.recv() for p in self.parent_remote]
        self._obs, self._rew, self._done, self._info = zip(*result)
        self._obs = np.stack(self._obs)
        self._rew = np.stack(self._rew)
        self._done = np.stack(self._done)
        self._info = np.stack(self._info)
        return self._obs, self._rew, self._done, self._info

    # modify: 
    def reset(self, id: Optional[Union[int, List[int]]] = None, iter = False) -> np.ndarray:
        if id is None:
            for p in self.parent_remote:
                # modify: 
                p.send(['reset', iter])
            self._obs = np.stack([p.recv() for p in self.parent_remote])
            return self._obs
        else:
            if np.isscalar(id):
                id = [id]
            for i in id:
                # modify: 
                self.parent_remote[i].send(['reset', iter])
            for i in id:
                self._obs[i] = self.parent_remote[i].recv()
            return self._obs

    # modify: reset_best for show the last trajectory with record variables saved
    def reset_best(self, id: Optional[Union[int, List[int]]] = None) -> np.ndarray:
        if id is None:
            for p in self.parent_remote:
                # modify: reset_best for show the last trajectory with record variables saved
                p.send(['reset_best',""])
            self._obs = np.stack([p.recv() for p in self.parent_remote])
            return self._obs
        else:
            if np.isscalar(id):
                id = [id]
            for i in id:
                # modify: reset_best for show the last trajectory with record variables saved
                self.parent_remote[i].send(['reset_best',""])
            for i in id:
                self._obs[i] = self.parent_remote[i].recv()
            return self._obs

    def seed(self, seed: Optional[Union[int, List[int]]] = None) -> List[int]:
        if np.isscalar(seed):
            seed = [seed + _ for _ in range(self.env_num)]
        elif seed is None:
            seed = [seed] * self.env_num
        for p, s in zip(self.parent_remote, seed):
            p.send(['seed', s])
        return [p.recv() for p in self.parent_remote]

    def render(self, **kwargs) -> List[Any]:
        for p in self.parent_remote:
            p.send(['render', kwargs])
        return [p.recv() for p in self.parent_remote]

    def close(self) -> List[Any]:
        if self.closed:
            return []
        for p in self.parent_remote:
            p.send(['close', None])
        result = [p.recv() for p in self.parent_remote]
        self.closed = True
        for p in self.processes:
            p.join()
        return result
    
    # modify: add set_drugs
    def set_drugs(self, id):
        for p in self.parent_remote:
            p.send(['set_drugs', id])
        return [p.recv() for p in self.parent_remote]

    # modify: add set_initial_state
    def set_initial_state(self, initail_state):
        for i in range(self.env_num):
            self.parent_remote[i].send(['set_initial_state', initail_state])
        return [p.recv() for p in self.parent_remote]

    # modify: add set_initial_state
    def save_trajectory_data(self,trajectory_path):
        self.parent_remote[0].send(["save_trajectory_data",trajectory_path])
        self.parent_remote[0].recv()

class RayVectorEnv(BaseVectorEnv):
    """Vectorized environment wrapper based on
    `ray <https://github.com/ray-project/ray>`_. However, according to our
    test, it is about two times slower than
    :class:`~tianshou.env.SubprocVectorEnv`.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for more detailed
        explanation.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]) -> None:
        super().__init__(env_fns)
        try:
            if not ray.is_initialized():
                ray.init()
        except NameError:
            raise ImportError(
                'Please install ray to support RayVectorEnv: pip3 install ray')
        self.envs = [
            ray.remote(gym.Wrapper).options(num_cpus=0).remote(e())
            for e in env_fns]

    def __getattr__(self, key):
        return ray.get([e.getattr.remote(key) for e in self.envs])

    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert len(action) == self.env_num
        result = ray.get([e.step.remote(a) for e, a in zip(self.envs, action)])
        self._obs, self._rew, self._done, self._info = zip(*result)
        self._obs = np.stack(self._obs)
        self._rew = np.stack(self._rew)
        self._done = np.stack(self._done)
        self._info = np.stack(self._info)
        return self._obs, self._rew, self._done, self._info

    def reset(self, id: Optional[Union[int, List[int]]] = None) -> np.ndarray:
        if id is None:
            result_obj = [e.reset.remote() for e in self.envs]
            self._obs = np.stack(ray.get(result_obj))
        else:
            result_obj = []
            if np.isscalar(id):
                id = [id]
            for i in id:
                result_obj.append(self.envs[i].reset.remote())
            for _, i in enumerate(id):
                self._obs[i] = ray.get(result_obj[_])
        return self._obs

    def seed(self, seed: Optional[Union[int, List[int]]] = None) -> List[int]:
        if not hasattr(self.envs[0], 'seed'):
            return []
        if np.isscalar(seed):
            seed = [seed + _ for _ in range(self.env_num)]
        elif seed is None:
            seed = [seed] * self.env_num
        return ray.get([e.seed.remote(s) for e, s in zip(self.envs, seed)])

    def render(self, **kwargs) -> List[Any]:
        if not hasattr(self.envs[0], 'render'):
            return [None for e in self.envs]
        return ray.get([e.render.remote(**kwargs) for e in self.envs])

    def close(self) -> List[Any]:
        return ray.get([e.close.remote() for e in self.envs])

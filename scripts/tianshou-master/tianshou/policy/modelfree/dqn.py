import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from typing import Dict, Union, Optional

from tianshou.policy import BasePolicy
from tianshou.data import Batch, ReplayBuffer, PrioritizedReplayBuffer, \
    to_torch_as, to_numpy


class DQNPolicy(BasePolicy):
    """Implementation of Deep Q Network. arXiv:1312.5602
    Implementation of Double Q-Learning. arXiv:1509.06461

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int estimation_step: greater than 1, the number of steps to look
        ahead.
    :param int target_update_freq: the target network update frequency (``0``
        if you do not use the target network).

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 optim: torch.optim.Optimizer,
                 discount_factor: float = 0.99,
                 estimation_step: int = 1,
                 target_update_freq: Optional[int] = 0,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        self.eps = 0
        assert 0 <= discount_factor <= 1, 'discount_factor should in [0, 1]'
        self._gamma = discount_factor
        assert estimation_step > 0, 'estimation_step should greater than 0'
        self._n_step = estimation_step
        self._target = target_update_freq > 0
        self._freq = target_update_freq
        self._cnt = 0
        if self._target:
            self.model_old = deepcopy(self.model)
            self.model_old.eval()

        self.epoch = 0

    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps

    def train(self, mode=True) -> torch.nn.Module:
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.model.train(mode)
        return self

    def sync_weight(self) -> None:
        """Synchronize the weight for the target network."""
        self.model_old.load_state_dict(self.model.state_dict())

    def _target_q(self, buffer: ReplayBuffer,
                  indice: np.ndarray) -> torch.Tensor:
        batch = buffer[indice]  # batch.obs_next: s_{t+n}
        if self._target:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            a = self(batch, input='obs_next', eps=0).act
            with torch.no_grad():
                target_q = self(
                    batch, model='model_old', input='obs_next').logits
            target_q = target_q[np.arange(len(a)), a]
        else:
            with torch.no_grad():
                target_q = self(batch, input='obs_next').logits.max(dim=1)[0]
        return target_q

    def process_fn(self, batch: Batch, buffer: ReplayBuffer,
                   indice: np.ndarray) -> Batch:
        r"""Compute the n-step return for Q-learning targets:

        .. math::
            G_t = \sum_{i = t}^{t + n - 1} \gamma^{i - t}(1 - d_i)r_i +
            \gamma^n (1 - d_{t + n}) \max_a Q_{old}(s_{t + n}, \arg\max_a
            (Q_{new}(s_{t + n}, a)))

        , where :math:`\gamma` is the discount factor,
        :math:`\gamma \in [0, 1]`, :math:`d_t` is the done flag of step
        :math:`t`. If there is no target network, the :math:`Q_{old}` is equal
        to :math:`Q_{new}`.
        """
        batch = self.compute_nstep_return(
            batch, buffer, indice, self._target_q, self._gamma, self._n_step)
        if isinstance(buffer, PrioritizedReplayBuffer):
            batch.update_weight = buffer.update_weight
            batch.indice = indice
        return batch

    def forward(self, batch: Batch,
                state: Optional[Union[dict, Batch, np.ndarray]] = None,
                model: str = 'model',
                input: str = 'obs',
                eps: Optional[float] = None,
                action_flag: Optional[int] = None, # modify: for experiment
                first_drug_number: Optional[int] = None, # modify: for experiment
                **kwargs) -> Batch:
        """Compute action over the given batch data.

        :param float eps: in [0, 1], for epsilon-greedy exploration method.

        :return: A :class:`~tianshou.data.Batch` which has 3 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        # modify: for experiment
        if action_flag is None:
            model = getattr(self, model)
            obs = getattr(batch, input)
            q, h = model(obs, state=state, info=batch.info)
            act = to_numpy(q.max(dim=1)[1]) # modify q[:,0:act_num_1]; q[:,act_num_1:]
            # add eps to act
            if eps is None:
                eps = self.eps
            if not np.isclose(eps, 0):
                for i in range(len(q)):
                    if np.random.rand() < eps:
                        act[i] = np.random.randint(q.shape[1]) # modify randint(0,act_num_1);randint(act_num_1,q.shape[1])
            return Batch(logits=q, act=act, state=h)
        if action_flag == 1:
            model = getattr(self, model)
            obs = getattr(batch, input)
            q, h = model(obs, state=state, info=batch.info)
            q = q[:, 0:first_drug_number]
            act = to_numpy(q.max(dim=1)[1]) # modify q[:,0:act_num_1]; q[:,act_num_1:]
            # add eps to act
            if eps is None:
                eps = self.eps
            if not np.isclose(eps, 0):
                for i in range(len(q)):
                    if np.random.rand() < eps:
                        act[i] = np.random.randint(q.shape[1]) # modify randint(0,act_num_1);randint(act_num_1,q.shape[1])
            return Batch(logits=q, act=act, state=h)
        if action_flag == 2:
            model = getattr(self, model)
            obs = getattr(batch, input)
            q, h = model(obs, state=state, info=batch.info)
            q = q[:,first_drug_number:]
            act = to_numpy(q.max(dim=1)[1]) + first_drug_number # modify 
            # add eps to act
            if eps is None:
                eps = self.eps
            if not np.isclose(eps, 0):
                for i in range(len(q)):
                    if np.random.rand() < eps:
                        act[i] = np.random.randint(q.shape[1]) + first_drug_number # modify randint(0,act_num_1);randint(act_num_1,q.shape[1])
            return Batch(logits=q, act=act, state=h)
        # model = getattr(self, model)
        # obs = getattr(batch, input)
        # q, h = model(obs, state=state, info=batch.info)
        # act = to_numpy(q.max(dim=1)[1]) # modify q[:,0:act_num_1]; q[:,act_num_1:]
        # # add eps to act
        # if eps is None:
        #     eps = self.eps
        # if not np.isclose(eps, 0):
        #     for i in range(len(q)):
        #         if np.random.rand() < eps:
        #             act[i] = np.random.randint(q.shape[1]) # modify randint(0,act_num_1);randint(act_num_1,q.shape[1])
        # return Batch(logits=q, act=act, state=h)

    def learn(self, batch: Batch, **kwargs) -> Dict[str, float]:
        if self._target and self._cnt % self._freq == 0:
            print("updating target......")
            self.sync_weight()
        self.optim.zero_grad()
        q = self(batch).logits
        q = q[np.arange(len(q)), batch.act]
        r = to_torch_as(batch.returns, q)
        if hasattr(batch, 'update_weight'):
            td = r - q
            batch.update_weight(batch.indice, to_numpy(td))
            impt_weight = to_torch_as(batch.impt_weight, q)
            loss = (td.pow(2) * impt_weight).mean()
        else:
            loss = F.mse_loss(q, r)
        loss.backward()
        self.optim.step()
        self._cnt += 1
        return {'loss': loss.item()}

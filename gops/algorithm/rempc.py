#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Approximate Dynamic Program Algorithm for Finity Horizon (FHADP)
#  Reference: Li SE (2023)
#             Reinforcement Learning for Sequential Decision and Optimal Control. Springer, Singapore.

__all__ = ["REMPC"]

from copy import deepcopy
from typing import Tuple
import torch
import torch.nn as nn
from torch.optim import Adam
import time
import warnings
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.create_pkg.create_env_model import create_env_model
from gops.utils.common_utils import get_apprfunc_dict
from gops.utils.gops_typing import DataDict, InfoDict
from gops.utils.tensorboard_setup import tb_tags
from gops.algorithm.base import AlgorithmBase, ApprBase


class ApproxContainer(ApprBase):
    def __init__(
        self,
        *,
        policy_learning_rate: float,
        value_learning_rate: float,
        **kwargs,
    ):
        """Approximate function container for FHADP."""
        """Contains one policy network."""
        super().__init__(kwargs)
        policy_args = get_apprfunc_dict("policy", **kwargs)
        v_args = get_apprfunc_dict("value", **kwargs)

        self.policy = create_apprfunc(**policy_args)
        self.v = create_apprfunc(**v_args)
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=policy_learning_rate
        )
        self.v_optimizer = Adam(
            self.v.parameters(), lr=value_learning_rate
        )

        self.optimizer_dict = {
            "policy": self.policy_optimizer,
            "v": self.v_optimizer,
        }
        self.init_scheduler(**kwargs)

    def create_action_distributions(self, logits):
        """create action distribution"""
        return self.policy.get_act_dist(logits)


class REMPC(AlgorithmBase):
    """Approximate Dynamic Program Algorithm for Finity Horizon

    Paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4124940

    :param int pre_horizon: envmodel forward step.
    :param float gamma: discount factor.
    """

    def __init__(self, index=0, **kwargs):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.envmodel = create_env_model(**kwargs)
        self.pre_horizon = kwargs["pre_horizon"]
        self.gamma = 1.0
        self.tb_info = dict()

    @property
    def adjustable_parameters(self):
        para_tuple = ("pre_horizon", "gamma")
        return para_tuple

    def _local_update(self, data: DataDict, iteration: int) -> InfoDict:
        self._compute_gradient(data)
        self.networks.policy_optimizer.step()
        self.networks.v_optimizer.step()
        return self.tb_info

    def get_remote_update_info(self, data: DataDict, iteration: int) -> Tuple[InfoDict, DataDict]:
        self._compute_gradient(data)
        policy_grad = [p._grad for p in self.networks.policy.parameters()]
        v_grad = [p._grad for p in self.networks.v.parameters()]
        update_info = dict()
        update_info["policy_grad"] = policy_grad
        update_info["v_grad"] = v_grad
        return self.tb_info, update_info

    def _remote_update(self, update_info: DataDict):
        for p, grad in zip(self.networks.policy.parameters(), update_info["policy_grad"]):
            p.grad = grad
        for p, grad in zip(self.networks.v.parameters(), update_info["v_grad"]):
            p.grad = grad
        self.networks.policy_optimizer.step()
        self.networks.v_optimizer.step()

    def _compute_gradient(self, data: DataDict):
        start_time = time.time()
        self.networks.policy.zero_grad()
        loss_policy, loss_info = self._compute_loss_policy(deepcopy(data))
        loss_policy.backward()
        loss_v, loss_info_v = self._compute_loss_v(deepcopy(data))
        loss_v.backward()
        end_time = time.time()
        self.tb_info.update(loss_info)
        self.tb_info.update(loss_info_v)
        self.tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000  # ms

    def _compute_loss_policy(self, data):
        o, d = data["obs"], data["done"]
        info = data
        v_pi = 0
        state_list = []
        a = self.networks.policy.forward_all_policy(o)
        for step in range(self.pre_horizon):
            o, _, d, info = self.envmodel.forward_dynamics(o, a[:, step, :], d, info)
            state_list.append(info['state'])
        v_pi, v_pi_details = self.envmodel.forward_reward(state_list, a)
        loss_policy = -v_pi.mean()
        loss_info = {
            tb_tags["loss_actor"]: loss_policy.item()
        }
        return loss_policy, loss_info

    def _compute_loss_v(self, data: DataDict) -> Tuple[torch.Tensor, InfoDict]:
        o, d = data["obs"], data["done"]
        info = data
        v_pi = torch.tensor(0.)
        state_list = []
        a = self.networks.policy.forward_all_policy(o)
        for step in range(self.pre_horizon):
            if step == 0:
                value_nn = self.networks.v(o)
            o, _, d, info = self.envmodel.forward_dynamics(o, a[:, step, :], d, info)
            state_list.append(info['state'])
        v_pi, v_pi_details = self.envmodel.forward_reward(state_list, a)
        loss_v = ((value_nn - v_pi.detach()) ** 2).mean()
        loss_info = {
            tb_tags["loss_critic"]: loss_v.item()
        }
        return loss_v, loss_info
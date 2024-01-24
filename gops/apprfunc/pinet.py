#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Attention network
#  Update: 2023-07-03, Tong Liu: create attention
__all__ = [
    "PINet",
    "DetermPolicy",
    "FiniteHorizonPolicy",
    "FiniteHorizonFullPolicy",
    "StochaPolicy",
    "ActionValue",
    "ActionValueDis",
    "ActionValueDistri",
    "ActionValueDistriMultiR",
]
import numpy as np
import torch
import warnings
import itertools
import torch.nn as nn
from typing import Tuple
from functools import reduce
from abc import abstractmethod, ABCMeta
from gops.utils.act_distribution_cls import Action_Distribution
from gops.utils.common_utils import get_activation_func, FreezeParameters
from gops.apprfunc.mlp import mlp



def init_weights(m):
    if isinstance(m, nn.Linear):
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.zero_()

class PINet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.begin = kwargs["pi_begin"]
        self.end = kwargs["pi_end"]
        self.d_encodings = self.end - self.begin
        self.enable_mask = kwargs["enable_mask"]
        self.d_obj = kwargs["obj_dim"]
        assert self.d_encodings % self.d_obj == 0
        self.num_objs = int(self.d_encodings / self.d_obj)
        if self.enable_mask:
            self.pi_in_dim = self.d_obj -1 # the last dimension is mask
        else:
            self.pi_in_dim = self.d_obj
        self.pi_out_dim = kwargs.get("pi_out_dim", self.pi_in_dim*self.num_objs +1)


        self.encoding_others = kwargs["encoding_others"]
        obs_dim = kwargs["obs_dim"]
        self.others_in_dim = obs_dim - self.d_encodings
        if self.encoding_others:
            self.others_out_dim = kwargs["others_out_dim"]
        else: 
            self.others_out_dim = self.others_in_dim
        self.output_dim = self.others_out_dim + self.pi_out_dim
        hidden_sizes = kwargs["pi_hidden_sizes"]
        pi_sizes =  [self.pi_in_dim] + list(hidden_sizes) + [self.pi_out_dim]
        self.pi = mlp(
            pi_sizes,
            get_activation_func(kwargs["pi_hidden_activation"]),
            get_activation_func(kwargs["pi_output_activation"]),
        )
        if self.encoding_others:
            self.others_encoder = mlp(
                [self.others_in_dim] + list(kwargs["others_hidden_sizes"]) + [self.others_out_dim],
                get_activation_func(kwargs["others_hidden_activation"]),
                get_activation_func(kwargs["others_output_activation"]),
            )
        else:
            self.others_encoder = nn.Identity()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        objs = obs[:, self.begin:self.end]
        others = torch.cat([obs[:, :self.begin], obs[:, self.end:]], dim=1)
        objs = objs.reshape(-1, self.num_objs, self.d_obj)
        others = self.others_encoder(others)

        if self.enable_mask:
            mask = objs[:, :, -1:]
            objs = objs[:, :, :-1]*mask

        pi = self.pi(objs)
        pi = pi.sum(dim=1, keepdim=False)
        
        return torch.cat([others, pi], dim=1)
    


class DetermPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy.
    Input: observation.
    Output: action.
    """

    def __init__(self, **kwargs):
        raise NotImplementedError


class FiniteHorizonPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy for finite-horizon.
    Input: observation, time step.
    Output: action.
    """

    def __init__(self, **kwargs):
        raise NotImplementedError


class FiniteHorizonFullPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy for finite-horizon.
    Input: observation, time step.
    Output: action.
    """

    def __init__(self, **kwargs):
        raise NotImplementedError


# Stochastic Policy
class StochaPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of stochastic policy.
    Input: observation.
    Output: parameters of action distribution.
    """

    def __init__(self, **kwargs):
        super().__init__()
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.std_type = kwargs["std_type"]
        self.pi_net = kwargs["pi_net"]
        self.freeze_pi_net = kwargs["freeze_pi_net"] == "actor" 
        input_dim = self.pi_net.output_dim

        # mean and log_std are calculated by different MLP
        if self.std_type == "mlp_separated":
            pi_sizes = [input_dim] + list(hidden_sizes) + [act_dim]
            self.mean = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
            self.log_std = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
        # mean and log_std are calculated by same MLP
        elif self.std_type == "mlp_shared":
            pi_sizes = [input_dim] + list(hidden_sizes) + [act_dim * 2]
            self.policy = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
        # mean is calculated by MLP, and log_std is learnable parameter
        elif self.std_type == "parameter":
            pi_sizes = [input_dim] + list(hidden_sizes) + [act_dim]
            self.mean = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
            self.log_std = nn.Parameter(-0.5*torch.ones(1, act_dim))

        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def shared_paras(self):
        return self.pi_net.parameters()

    def ego_paras(self):
        return itertools.chain(*[modules.parameters() for modules in self.children() if modules != self.pi_net])
        

    def forward(self, obs):
        with FreezeParameters([self.pi_net], self.freeze_pi_net):
            encoding = self.pi_net(obs)
        if self.std_type == "mlp_separated":
            action_mean = self.mean(encoding)
            action_std = torch.clamp(
                self.log_std(encoding), self.min_log_std, self.max_log_std
            ).exp()
        elif self.std_type == "mlp_shared":
            logits = self.policy(encoding)
            action_mean, action_log_std = torch.chunk(
                logits, chunks=2, dim=-1
            )  # output the mean
            action_std = torch.clamp(
                action_log_std, self.min_log_std, self.max_log_std
            ).exp()
        elif self.std_type == "parameter":
            action_mean = self.mean(encoding)
            action_log_std = self.log_std + torch.zeros_like(action_mean)
            action_std = torch.clamp(
                action_log_std, self.min_log_std, self.max_log_std
            ).exp()

        return torch.cat((action_mean, action_std), dim=-1)


class ActionValue(nn.Module, Action_Distribution):
    """
    Approximated function of action-value function.
    Input: observation, action.
    Output: action-value.
    """

    def __init__(self, **kwargs):
        super().__init__()
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.pi_net = kwargs["pi_net"]
        self.freeze_pi_net = kwargs["freeze_pi_net"] == "critic"
        input_dim = self.pi_net.output_dim + act_dim

        self.q = mlp(
            [input_dim] + list(hidden_sizes) + [1],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def shared_paras(self):
        return self.pi_net.parameters()
    
    def ego_paras(self):
        return itertools.chain(*[modules.parameters() for modules in self.children() if modules != self.pi_net])

    def forward(self, obs, act):
        with FreezeParameters([self.pi_net], self.freeze_pi_net):
            encoding = self.pi_net(obs)
        q = self.q(torch.cat([encoding, act], dim=-1))
        return torch.squeeze(q, -1)


class ActionValueDis(nn.Module, Action_Distribution):
    """
    Approximated function of action-value function for discrete action space.
    Input: observation.
    Output: action-value for all action.
    """

    def __init__(self, **kwargs):
        raise NotImplementedError

class ActionValueDistri(nn.Module):
    """
    Approximated function of distributed action-value function.
    Input: observation.
    Output: parameters of action-value distribution.
    """

    def __init__(self, **kwargs):
        super().__init__()
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.pi_net = kwargs["pi_net"]
        self.freeze_pi_net = kwargs["freeze_pi_net"] == "critic"
        input_dim = self.pi_net.output_dim + act_dim
        self.q = mlp(
            [input_dim] + list(hidden_sizes) + [2],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        if "min_log_std"  in kwargs or "max_log_std" in kwargs:
            warnings.warn("min_log_std and max_log_std are deprecated in ActionValueDistri.")

    def shared_paras(self):
        return self.pi_net.parameters()

    def ego_paras(self):
        return itertools.chain(*[modules.parameters() for modules in self.children() if modules != self.pi_net])

    def forward(self, obs, act):
        
        with FreezeParameters([self.pi_net], self.freeze_pi_net):
            encoding = self.pi_net(obs)
        
        logits = self.q(torch.cat([encoding, act], dim=-1))
        value_mean, value_std = torch.chunk(logits, chunks=2, dim=-1)
        value_log_std = torch.nn.functional.softplus(value_std) 

        return torch.cat((value_mean, value_log_std), dim=-1)
    


class ActionValueDistriMultiR(nn.Module):
    """
    Approximated function of distributed action-value function.
    Input: observation.
    Output: parameters of action-value distribution.
    """

    def __init__(self, **kwargs):
        super().__init__()
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.pi_net = kwargs["pi_net"]
        self.freeze_pi_net = kwargs["freeze_pi_net"] == "critic"
        input_dim = self.pi_net.output_dim + act_dim
        self.q = mlp(
            [input_dim] + list(hidden_sizes) + [2],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        if "min_log_std"  in kwargs or "max_log_std" in kwargs:
            warnings.warn("min_log_std and max_log_std are deprecated in ActionValueDistri.")

        #rew_comp_dim = reduce(lambda x, y: x + y, [value["shape"] for value in kwargs["additional_info"].values()])
        rew_comp_dim = kwargs["additional_info"]["reward_comps"]["shape"][0]
        self.q_comp = mlp(
            [input_dim] + list(hidden_sizes) + [rew_comp_dim],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )

    def shared_paras(self):
        return self.pi_net.parameters()

    def ego_paras(self):
        return itertools.chain(*[modules.parameters() for modules in self.children() if modules != self.pi_net])

    def forward(self, obs, act):
        
        with FreezeParameters([self.pi_net], self.freeze_pi_net):
            encoding = self.pi_net(obs)
        
        logits = self.q(torch.cat([encoding, act], dim=-1))
        value_mean, value_std = torch.chunk(logits, chunks=2, dim=-1)
        value_log_std = torch.nn.functional.softplus(value_std) 

        return torch.cat((value_mean, value_log_std), dim=-1)
    
    def cal_comp(self, obs, act):
        with FreezeParameters([self.pi_net], True):
            encoding = self.pi_net(obs)
        return self.q_comp(torch.cat([encoding, act], dim=-1))
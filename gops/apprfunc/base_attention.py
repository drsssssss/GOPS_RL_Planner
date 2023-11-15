#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Attention network
#  Update: 2023-07-03, Tong Liu: create attention
__all__ = [
    "BaseAttention",
]
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from abc import abstractmethod, ABCMeta
from gops.utils.act_distribution_cls import Action_Distribution
from gops.utils.common_utils import get_activation_func
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

class BaseAttention(Action_Distribution, nn.Module, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.begin = kwargs["attn_begin"]
        self.end = kwargs["attn_end"]
        self.d_others = self.end - self.begin + 1
        self.d_obj = kwargs["attn_in_per_dim"]
        self.d_model = kwargs["attn_out_dim"]
        assert self.d_others % self.d_obj == 0
        self.num_objs = int(self.d_others / self.d_obj)
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))

        obs_dim = kwargs["obs_dim"]
        self.act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.pre_horizon = self.pre_horizon if isinstance(self.pre_horizon, int) else 1
        pi_sizes = [obs_dim] + list(hidden_sizes) + [self.act_dim * self.pre_horizon]
        self.pi = mlp(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )

        self.embedding = nn.Sequential(
            nn.Linear(self.d_obj - 1, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, self.d_model),
        )
        self.Uq = nn.Linear(self.d_model, self.d_model, bias=False, dtype=torch.float32)
        self.Ur = nn.Linear(self.d_model, self.d_model, bias=False, dtype=torch.float32)

        init_weights(self.embedding)
        init_weights(self.Uq)
        init_weights(self.Ur)

    @abstractmethod
    def preprocessing(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        # Return attention_obs("torch.Tensor") for attention_forward() and all other necessary auxiliary data("Tuple") in a tuple.
        ...

    def attention_forward(self, attention_obs: torch.Tensor) -> torch.Tensor:
        attention_obs = torch.reshape(attention_obs, [-1, self.num_objs, self.d_obj])
        attention_mask = attention_obs[:, :, -1].squeeze(axis=-1)
        attention_obs = attention_obs[:, :, :-1]
        attention_obs = self.embedding(attention_obs)

        x_real = attention_obs * attention_mask.unsqueeze(axis=-1)  # fake tensors are all zeros
        query = x_real.sum(axis=-2) / (attention_mask.sum(axis=-1) + 1e-5).unsqueeze(
            axis=-1)  # [b, d_model] / [B, 1] --> [B, d_model]

        logits = torch.bmm(self.Uq(query).unsqueeze(1), self.Ur(x_real).transpose(-1, -2)).squeeze(1)  # [B, N]

        logits = logits + ((1 - attention_mask) * -1e9)
        attention_weights = torch.softmax(logits, axis=-1)

        attention_obs = torch.matmul(attention_weights.unsqueeze(axis=1),x_real).squeeze( axis=-2)

        #o = torch.concat((ego_obs, ref_obs, surr_encode), dim=1)
        #return o, surr_encode, attention_weights
        return attention_obs

    def postprocessing(self, attention_obs: torch.Tensor, auxiliary: Tuple) -> torch.Tensor:
        obs = torch.concat([auxiliary, attention_obs], dim = -1)
        actions = self.pi(obs).reshape(obs.shape[0], self.pre_horizon, self.act_dim)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(actions) \
                 + (self.act_high_lim + self.act_low_lim) / 2
        return action

    def forward_all_policy(self, obs):
        attention_obs, auxilary = self.preprocessing(obs)
        attention_obs = self.attention_forward(attention_obs)
        action = self.postprocessing(attention_obs, auxilary)
        return action

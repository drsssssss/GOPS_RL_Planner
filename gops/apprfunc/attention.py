__all__ = [
    "AttentionPolicy",
    "AttentionFullPolicy"
]

import torch
from gops.utils.common_utils import get_activation_func
from gops.apprfunc.mlp import mlp
from typing import Tuple
from gops.apprfunc.base_attention import BaseAttention

class AttentionPolicy(BaseAttention):
    def __init__(self, **kwargs):
        self.action_distribution_cls = kwargs["action_distribution_cls"]
        self.pre_horizon = kwargs["pre_horizon"]
        super().__init__(**kwargs)
        #obs_dim = kwargs["obs_dim"]+1
        obs_dim = kwargs["obs_dim"]-( kwargs["attn_end"]-kwargs["attn_begin"]+1)+kwargs["attn_out_dim"]+1
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        
    def preprocessing(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
     
        attention_obs = obs[:, self.begin:(self.end+1)]
        auxiliary = torch.concat([obs[:, 0:self.begin], obs[:, (self.end+1):]], dim = -1)
        return attention_obs, auxiliary
    
    def postprocessing(self, attention_obs: torch.Tensor, auxiliary: Tuple) -> torch.Tensor:
        obs = torch.concat([auxiliary, attention_obs], dim = -1)
        return obs
    
    def forward(self, obs, virtual_t:int=1):
        attention_obs, auxilary = self.preprocessing(obs)
        attention_obs = self.attention_forward(attention_obs)
        obs_processed = self.postprocessing(attention_obs, auxilary)
        virtual_t = virtual_t * torch.ones(
            size=[obs.shape[0], 1], dtype=torch.float32, device=obs.device
        )
        expand_obs = torch.cat((obs_processed, virtual_t), 1)
        actions = self.pi(expand_obs).reshape(obs.shape[0], self.act_dim)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(actions) \
                 + (self.act_high_lim + self.act_low_lim) / 2
        return action
    
    
class AttentionFullPolicy(BaseAttention):
    def __init__(self, **kwargs):
        self.action_distribution_cls = kwargs["action_distribution_cls"]
        self.pre_horizon = kwargs["pre_horizon"]
        super().__init__(**kwargs)
        #obs_dim = kwargs["obs_dim"]+1
        obs_dim = kwargs["obs_dim"]-( kwargs["attn_end"]-kwargs["attn_begin"]+1)+kwargs["attn_out_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim * self.pre_horizon]
        self.pi = mlp(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        
    def preprocessing(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
     
        attention_obs = obs[:, self.begin:(self.end+1)]
        auxiliary = torch.concat([obs[:, 0:self.begin], obs[:, (self.end+1):]], dim = -1)
        return attention_obs, auxiliary
    
    def postprocessing(self, attention_obs: torch.Tensor, auxiliary: Tuple) -> torch.Tensor:
        obs = torch.concat([auxiliary, attention_obs], dim = -1)
        return obs
    
    def forward_all_policy(self, obs):
        attention_obs, auxilary = self.preprocessing(obs)
        attention_obs = self.attention_forward(attention_obs)
        obs_processed = self.postprocessing(attention_obs, auxilary)
        actions = self.pi(obs_processed).reshape(obs.shape[0], self.pre_horizon, self.act_dim)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(actions) \
                 + (self.act_high_lim + self.act_low_lim) / 2
        return action

    def forward(self, obs):
        return self.forward_all_policy(obs)[0, :]
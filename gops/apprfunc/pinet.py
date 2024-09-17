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
    "StochaCoherentPolicy",
    "StochaFourierPolicy",
    "StochaFourierCoherentPolicy",
    "StochaGuassianPolicy",
    "StochaRNNPolicy",
    "ActionValue",
    "ActionValueDis",
    "ActionValueDistri",
    "ActionValueDistriMultiR",
]
import numpy as np
import torch
import torch.nn.functional as F
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
        self.enable_self_attention = kwargs.get("enable_self_attention", False)
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
        init_weights(self.pi)
        if self.encoding_others:
            warnings.warn("encoding_others is enabled")
            self.others_encoder = mlp(
                [self.others_in_dim] + list(kwargs["others_hidden_sizes"]) + [self.others_out_dim],
                get_activation_func(kwargs["others_hidden_activation"]),
                get_activation_func(kwargs["others_output_activation"]),
            )
            init_weights(self.others_encoder)
        else:
            self.others_encoder = nn.Identity()

        if self.enable_self_attention:
            embedding_dim = self.pi_out_dim
            self.embedding_dim = embedding_dim
            warnings.warn("self_attention is enabled")
            if kwargs.get("attn_dim") is not None:
                self.attn_dim = kwargs["attn_dim"]
            else:
                self.attn_dim = self.pi_out_dim # default attn_dim is pi_out_dim
                warnings.warn("attn_dim is not specified, using pi_out_dim as attn_dim")
                if kwargs.get("head_num") is None:
                    head_num = 1
                    warnings.warn("head_num is not specified, using 1 as head_num")
                else:
                    head_num = kwargs["head_num"]
                head_dim = embedding_dim // head_num
                self.head_dim = head_dim
                self.head_num = head_num
                self.Wq =  nn.Linear(self.others_out_dim, head_dim*head_num)
                self.Wk =  nn.Linear(head_dim, head_dim)
                self.Wv =  nn.Linear(head_dim, head_dim)
            self.attn_weights = None 

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        objs = obs[:, self.begin:self.end]
        others = torch.cat([obs[:, :self.begin], obs[:, self.end:]], dim=1)
        objs = objs.reshape(-1, self.num_objs, self.d_obj)
        others = self.others_encoder(others)

        if self.enable_mask:
            mask = objs[:, :, -1]
            objs = objs[:, :, :-1]
        else:
            mask = torch.ones_like(objs[:, :, 0]) # [B, N]
        
        
        embeddings = self.pi(objs)*mask.unsqueeze(-1) # [B, N, d_model]

        if self.enable_self_attention:
            query = self.Wq(others).reshape(-1,1,self.head_num, self.head_dim) # [B, 1 head_num, head_dim]
            reshaped_embeddings = embeddings.reshape(-1, self.num_objs, self.head_num, self.head_dim) # [B, N, head_num, head_dim]
            key = self.Wk(reshaped_embeddings) # [B, N, head_num, head_dim]
            value = self.Wv(reshaped_embeddings) # [B, N, head_num, head_dim]
            value = value*mask.unsqueeze(-1).unsqueeze(-1) # [B, N, head_num, head_dim]
            # logits = torch.einsum("nqhd,nkhd->nhqk", [query, key]) / np.sqrt(self.embedding_dim) # [B, head_num, 1, N] donot use einsum
            query = query.permute(0, 2, 1, 3)  #  [B, head_num, 1, head_dim]
            key = key.permute(0, 2, 1, 3)  # 形状变为  [B, head_num, seq_len, head_dim]

            # 进行矩阵乘法
            logits = torch.matmul(query, key.transpose(-2, -1))  # 形状变为 [batch_size, num_heads, 1, seq_len]

            # 缩放
            logits = logits / np.sqrt(self.embedding_dim)  # head_dim 即 self.embedding_dim
            logits = logits.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
            attn_weights = torch.softmax(logits, axis=-1) # [B, head_num, 1, N]
            # embeddings = torch.einsum("nhqk,nkhd->nqhd", [attn_weights, value]).reshape(-1, self.embedding_dim) # [B, d_model]
            attn_p = attn_weights
            value_p = value.permute(0, 2, 1, 3)
            embeddings = torch.matmul(attn_p, value_p).reshape(-1, self.embedding_dim) # [B, d_model]
            self.attn_weights = attn_weights.squeeze(2).sum(1)/self.head_num



            
            # query = embeddings.sum(axis=-2) / (mask.sum(axis=-1) + 1e-5).unsqueeze(axis=-1) # [b, d_model] / [B, 1] --> [B, d_model]
            # query = torch.concat([query, others], dim=1) # [B, d_model + d_others]
            # logits = torch.bmm(self.Uq(query).unsqueeze(1), self.Ur(embeddings).transpose(-1, -2)).squeeze(1) / np.sqrt(self.attn_dim) # [B, N]
            # logits = logits + ((1 - mask) * -1e9)  # mask ==1 means the object is the true vehicle
            # self.attn_weights = torch.softmax(logits, axis=-1) # [B, N]
            # # self.attn_weights = (self.attn_weights + 0*mask)
            # # self.attn_weights = self.attn_weights / (self.attn_weights.sum(axis=-1, keepdim=True) + 1e-5)
            # # print(self.attn_weights)
            # embeddings = torch.matmul(self.attn_weights.unsqueeze(axis=1),embeddings).squeeze(axis=-2) # [B, d_model]
        else:
            embeddings = embeddings.sum(dim=1, keepdim=False) # [B, d_model]
        
        return torch.cat([others, embeddings], dim=1)
    


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


# coherent noise Policy of MLP
class StochaCoherentPolicy(nn.Module, Action_Distribution):
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

        self.act_seq_nn = kwargs.get("act_seq_nn", 1)
        assert act_dim % self.act_seq_nn == 0, "act_dim should be divisible by act_seq_nn"
        self.actual_act_dim = act_dim // self.act_seq_nn

        self.policy_latent_dim = 10 # FIXME: hard code

        # mean and log_std are calculated by different MLP
        if self.std_type == "mlp_separated":
            pi_sizes = [input_dim] + list(hidden_sizes) + [self.policy_latent_dim]
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
            pi_sizes = [input_dim] + list(hidden_sizes) + [self.policy_latent_dim * 2]
            self.policy = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
        # mean is calculated by MLP, and log_std is learnable parameter
        elif self.std_type == "parameter":
            pi_sizes = [input_dim] + list(hidden_sizes) + [self.policy_latent_dim]
            self.mean = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
            self.log_std = nn.Parameter(-0.5*torch.ones(1, self.policy_latent_dim))

        # FIXME: hard code
        self.planing_policy = mlp(
            [self.policy_latent_dim] + [64, 64] + [act_dim], 
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )

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

        return torch.cat((action_mean, action_std), dim=-1), self.planing_policy


class FourierFilterWrapper:
    def __init__(self, policy, act_dim, act_seq_nn=1):
        self.policy = policy
        self.act_seq_nn = act_seq_nn
        assert act_dim % self.act_seq_nn == 0, "act_dim should be divisible by act_seq_nn"
        self.actual_act_dim = act_dim // self.act_seq_nn
        
        # shape of the mask matrix: [act_seq_nn//2+1, act_dim]
        self.freq_mask = nn.Parameter(torch.ones(self.act_seq_nn//2+1, self.actual_act_dim))
      
    @staticmethod
    def fourier_filter(signal: torch.Tensor, freq_mask: torch.Tensor) -> torch.Tensor:
        """
        Applies a Fourier filter to the input signal using the provided frequency mask.

        This function performs a Fast Fourier Transform (FFT) on the real-valued input signal,
        applies the frequency mask to the FFT coefficients, and then performs an inverse FFT to
        obtain the filtered signal.

        Parameters:
        signal (torch.Tensor): The input signal tensor of shape (B, N, D), where B is the batch size
                            and N is the length of the signal, and D is the number of features.
        freq_mask (torch.Tensor): The frequency mask tensor of shape (F, D), 
                            where F is the number of frequency components and D is the number of features.
                            F = N//2+1 for real-to-complex FFT.
        Returns:
        torch.Tensor: The filtered signal tensor of shape (B, N, D).
        """
        fft_coeffs = torch.fft.rfft(signal, dim=1)  # [B, N//2+1, D]
        fft_coeffs_filtered = fft_coeffs * freq_mask.unsqueeze(0)  # [B, N//2+1, D]
        filtered_signal = torch.fft.irfft(fft_coeffs_filtered, dim=1)
        return filtered_signal
    
    def forward(self, obs):
        action_mean, action_std = self.policy.forward(obs).chunk(2, dim=-1)
        action_mean = action_mean.view(-1, self.act_seq_nn, self.actual_act_dim)
        self.freq_mask.data = torch.clamp(self.freq_mask.data, 0, 1)
        action_mean_filtered = self.fourier_filter(action_mean, self.freq_mask)
        action_mean_filtered = action_mean_filtered.reshape(-1, self.actual_act_dim * self.act_seq_nn)
        return torch.cat((action_mean_filtered, action_std), dim=-1)


class StochaFourierPolicy(StochaPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wrapper = FourierFilterWrapper(self, kwargs["act_dim"], kwargs.get("act_seq_nn", 1))
    
    def forward(self, obs):
        return self.wrapper.forward(obs)
        
    
class StochaFourierCoherentPolicy(StochaCoherentPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.planning_policy = FourierFilterWrapper(self.planing_policy, kwargs["act_dim"], kwargs["act_seq_nn"])


class StochaGuassianPolicy(StochaPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        act_dim = kwargs["act_dim"]
        self.act_seq_nn = kwargs.get("act_seq_nn", 1)
        assert act_dim % self.act_seq_nn == 0, "act_dim should be divisible by act_seq_nn"
        self.actual_act_dim = act_dim // self.act_seq_nn
        
        sigma = self.act_seq_nn / 6.0
        kernel_size = int(2 * (sigma * 3) + 1)
        gaussian_kernel = self.gaussian_kernel1d(kernel_size, sigma).view(1, 1, -1)
        gaussian_kernel = self.gaussian_kernel.expand(self.actual_act_dim, 1, -1)
        self.gaussian_kernel = nn.Parameter(gaussian_kernel, requires_grad=False)
    
    @staticmethod
    def gaussian_kernel1d(kernel_size: int, sigma: float) -> torch.Tensor:
        x = torch.arange(kernel_size) - (kernel_size - 1) / 2
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        return kernel

    def forward(self, obs):
        action_mean, action_std = super().forward(obs).chunk(2, dim=-1)
        action_mean = action_mean.view(-1, self.act_seq_nn, self.actual_act_dim).permute(0, 2, 1)
        padding = self.gaussian_kernel.size(-1) // 2
        action_mean = F.conv1d(
            F.pad(action_mean, (padding, padding), mode='replicate'),
            self.gaussian_kernel, # [self.actual_act_dim, 1, kernel_size]
            padding=0,
            groups=self.actual_act_dim 
        )
        action_mean = action_mean.permute(0, 2, 1).reshape(-1, self.actual_act_dim * self.act_seq_nn)
        return torch.cat((action_mean, action_std), dim=-1)
        
        
# Stochastic RNN Policy
class StochaRNNPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of stochastic policy.
    Input: observation.
    Output: parameters of action distribution.
    """

    def __init__(self, **kwargs):
        super().__init__()
        # In training, act_dim is the dimension of action sequence, and act_seq_len and act_seq_nn is the length of action sequence
        # In evaluation, act_dim is the dimension of action sequence, act_seq_len is 1, and act_seq_nn is the length of action sequence
        act_dim = kwargs["act_dim"]
        self.act_seq_nn = kwargs.get("act_seq_nn", 1)
        assert act_dim % self.act_seq_nn == 0, "act_dim should be divisible by act_seq_nn"
        self.actual_act_dim = act_dim // self.act_seq_nn
        
        hidden_sizes = kwargs["hidden_sizes"]
        rnn_hidden_size = kwargs["rnn_hidden_size"]
        self.rnn_type = kwargs["rnn_type"]
        self.std_type = kwargs["std_type"]
        self.pi_net = kwargs["pi_net"]
        self.freeze_pi_net = kwargs["freeze_pi_net"] == "actor" 
        input_dim = self.pi_net.output_dim
        # RNN
        if self.rnn_type == "GRU":
            self.hidden_encoder = mlp(
                [input_dim, rnn_hidden_size],
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["hidden_activation"]),
            )
            self.rnn_cell = nn.GRUCell(self.actual_act_dim * 2, rnn_hidden_size)
        elif self.rnn_type == "LSTM":
            self.hidden_nn = mlp(
                [input_dim, rnn_hidden_size * 2],
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["hidden_activation"]),
            )
            self.hidden_encoder = lambda x: torch.chunk(self.hidden_nn(x), chunks=2, dim=-1)
            self.rnn_cell = nn.LSTMCell(self.actual_act_dim * 2, rnn_hidden_size)
            
        else:
            raise NotImplementedError(f"'{self.rnn_type}' RNN type not implemented")
        # Decoder
        # 1. mean and log_std are calculated by different MLP
        if self.std_type == "mlp_separated":
            pi_sizes_first = [input_dim] + list(hidden_sizes) + [self.actual_act_dim]
            self.mean_first = mlp(
                pi_sizes_first,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
            self.log_std_first = mlp(
                pi_sizes_first,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
            pi_sizes_other = [rnn_hidden_size] + list(hidden_sizes) + [self.actual_act_dim]
            self.mean_other = mlp(
                pi_sizes_other,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
            self.log_std_other = mlp(
                pi_sizes_other,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
            self.policy = {
                "first": lambda x: (self.mean_first(x), self.log_std_first(x)),
                "other": lambda x: (self.mean_other(x), self.log_std_other(x)),
            }
        # 2. mean and log_std are calculated by same MLP
        elif self.std_type == "mlp_shared":
            pi_sizes_first = [input_dim] + list(hidden_sizes) + [self.actual_act_dim * 2]
            self.policy_first = mlp(
                pi_sizes_first,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
            pi_sizes_other = [rnn_hidden_size] + list(hidden_sizes) + [self.actual_act_dim * 2]
            self.policy_other = mlp(
                pi_sizes_other,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
            self.policy = {
                "first": lambda x: torch.chunk(self.policy_first(x), chunks=2, dim=-1),
                "other": lambda x: torch.chunk(self.policy_other(x), chunks=2, dim=-1),
            }
        # 3. mean is calculated by MLP, and log_std is learnable parameter
        elif self.std_type == "parameter":
            pi_sizes_first = [input_dim] + list(hidden_sizes) + [self.actual_act_dim]
            self.mean_first = mlp(
                pi_sizes_first,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
            self.log_std_first = nn.Parameter(-0.5*torch.ones(1, self.actual_act_dim))
            pi_sizes_other = [rnn_hidden_size] + list(hidden_sizes) + [self.actual_act_dim]
            self.mean_other = mlp(
                pi_sizes_other,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
            self.log_std_other = nn.Parameter(-0.5*torch.ones(1, self.actual_act_dim))
            self.policy = {
                "first": lambda x: (self.mean_first(x), self.log_std_first + torch.zeros_like(self.mean_first(x))),
                "other": lambda x: (self.mean_other(x), self.log_std_other + torch.zeros_like(self.mean_other(x))),
            }
            
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
        # Initialize the action plan
        batch_size = obs.size(0)
        device = obs.device
        act_plan_mean = torch.zeros(batch_size, self.actual_act_dim * self.act_seq_nn).to(device)
        act_plan_logstd = torch.zeros(batch_size, self.actual_act_dim * self.act_seq_nn).to(device)
        
        with FreezeParameters([self.pi_net], self.freeze_pi_net):
            encoding = self.pi_net(obs) # [B, input_dim]
            
        act_mean, act_logstd = self.policy["first"](encoding) # [B, actual_act_dim], [B, actual_act_dim]
        act_logstd = torch.clamp(act_logstd, self.min_log_std, self.max_log_std)
        act_plan_mean[:, :self.actual_act_dim] = act_mean
        act_plan_logstd[:, :self.actual_act_dim] = act_logstd
        
        hidden_state = self.hidden_encoder(encoding) # [B, rnn_input_size]
        
        for i in range(1, self.act_seq_nn):
            if self.rnn_type == "GRU":
                hidden_state = self.rnn_cell(torch.concat([act_mean, act_logstd], dim=-1), hidden_state)
                act_mean, act_logstd = self.policy["other"](hidden_state)
            elif self.rnn_type == "LSTM":
                hidden_state = self.rnn_cell(torch.concat([act_mean, act_logstd], dim=-1), hidden_state)
                act_mean, act_logstd = self.policy["other"](hidden_state[0])
                
            act_logstd = torch.clamp(act_logstd, self.min_log_std, self.max_log_std)
            act_plan_mean[:, i*self.actual_act_dim:(i+1)*self.actual_act_dim] = act_mean
            act_plan_logstd[:, i*self.actual_act_dim:(i+1)*self.actual_act_dim] = act_logstd
        
        act_plan_std = act_plan_logstd.exp()
        return torch.cat((act_plan_mean, act_plan_std), dim=-1)


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
        self.std_type = kwargs["std_type"]
        input_dim = self.pi_net.output_dim + act_dim
        if self.std_type == "mlp_shared":
            self.q = mlp(
                [input_dim] + list(hidden_sizes) + [2],
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
        elif self.std_type == "mlp_separated":
            self.q = mlp(
                [input_dim] + list(hidden_sizes) + [1],
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
            self.q_std = mlp(
                [input_dim] + list(hidden_sizes) + [1],
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
        else:
            raise NotImplementedError
        if "min_log_std"  in kwargs or "max_log_std" in kwargs:
            warnings.warn("min_log_std and max_log_std are deprecated in ActionValueDistri.")

    def shared_paras(self):
        return self.pi_net.parameters()

    def ego_paras(self):
        return itertools.chain(*[modules.parameters() for modules in self.children() if modules != self.pi_net])

    def forward(self, obs, act):
        
        with FreezeParameters([self.pi_net], self.freeze_pi_net):
            encoding = self.pi_net(obs)
        if self.std_type == "mlp_shared":
            logits = self.q(torch.cat([encoding, act], dim=-1))
            value_mean, value_std = torch.chunk(logits, chunks=2, dim=-1)
            value_std = torch.nn.functional.softplus(value_std) 

        elif self.std_type == "mlp_separated":
            value_mean = self.q(torch.cat([encoding, act], dim=-1))
            value_std = torch.nn.functional.softplus(self.q_std(torch.cat([encoding, act], dim=-1)))

        return torch.cat((value_mean, value_std), dim=-1)
    


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
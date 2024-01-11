#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Transform pkl network to onnx version
#  Update: 2023-01-05, Jiaxin Gao: Create codes

import contextlib
import torch, torch.nn as nn
import onnxruntime as ort
import argparse
import os
import sys
import pandas as pd
from gops.utils.common_utils import get_args_from_json
import numpy as np


gops_path = '/root/gops/gops'
# Add algorithm file to sys path
alg_file = "algorithm"
alg_path = os.path.join(gops_path, alg_file)
sys.path.append(alg_path)


def __load_args(log_policy_dir):
    log_policy_dir = log_policy_dir
    json_path = os.path.join(log_policy_dir, "config.json")
    parser = argparse.ArgumentParser()
    args_dict = vars(parser.parse_args())
    args = get_args_from_json(json_path, args_dict)
    return args

def export_model(model: nn.Module, example_obs: torch.Tensor, path: str):
    with _module_inference(model):
        inference_helper = _InferenceHelper(model)
        torch.onnx.export(inference_helper, example_obs, path, input_names=['input'], output_names=['output'],
                          opset_version=11)


@contextlib.contextmanager
def _module_inference(module: nn.Module):
    training = module.training
    module.train(False)
    yield
    module.train(training)


class _InferenceHelper(nn.Module):
    def __init__(self, model):
        super().__init__()

        from gops.apprfunc.mlp import Action_Distribution

        assert isinstance(model, nn.Module) and isinstance(
            model, Action_Distribution
        ), (
            "The model must inherit from nn.Module and Action_Distribution. "
            f"Got {model.__class__.__mro__}"
        )
        self.model = model

    def forward(self, obs: torch.Tensor):
        obs = obs.unsqueeze(0)
        logits = self.model(obs)
        act_dist = self.model.get_act_dist(logits)
        mode = act_dist.mode()
        return mode.squeeze(0)

class _InferenceHelper_FHADP(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, obs: torch.Tensor):
        return self.model(obs, torch.ones(1))

class _InferenceHelper_FHADP2(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, obs: torch.Tensor):
        assert obs.ndim == 1
        obs = obs.unsqueeze(0)
        act = self.model(obs)
        assert act.ndim == 2  # [Horizon, Action_dim]
        return act[0]
    
class _InferenceHelper_Policy_DSAC(nn.Module):
    def __init__(self, model, act_scale_factor,obs_scale_factor, bias):
        super().__init__()
        self.model = model
        self.act_scale_factor = act_scale_factor
        self.obs_scale_factor = obs_scale_factor
        self.bias = bias

    def forward(self, obs: torch.Tensor):
        logits = self.model.policy(obs)
        action_distribution = self.model.create_action_distributions(logits)
        action = action_distribution.mode().float()
        real_act = action*self.act_scale_factor + self.bias
        real_act = action
        return real_act
    
class _InferenceHelper_Q_DSAC(nn.Module):
    def __init__(self, model, act_dim):
        super().__init__()
        self.model = model
        self.act_dim = act_dim


    def forward(self, obs_act: torch.Tensor):
        obs  = obs_act[:,0:-act_dim]
        act = obs_act[:,-act_dim:]
        logits = self.model(obs,act)
        mean ,_ = torch.chunk(logits,2,-1)
        return mean
    
def deterministic_policy_export_onnx_model(networks, input_dim, policy_dir):

    example = torch.rand(1, input_dim)  # network input dim
    output_onnx_model = policy_dir
    torch.onnx.export(networks.policy, example, output_onnx_model, input_names=['input'],
                      output_names=['output'], opset_version=11)

def fhadp_policy_export_onnx_model(networks, input_dim, policy_dir):

    example = torch.rand(1, input_dim)  # network input dim
    output_onnx_model = policy_dir
    model = _InferenceHelper_FHADP(networks.policy)
    torch.onnx.export(model, example, output_onnx_model, input_names=['input', "input1"],
                      output_names=['output'], opset_version=11)

def fhadp2_policy_export_onnx_model(networks, input_dim, policy_dir):

    example = torch.rand(input_dim)  # network input dim
    output_onnx_model = policy_dir
    torch.onnx.export(_InferenceHelper_FHADP2(networks.policy), example, output_onnx_model, input_names=['input'],
                      output_names=['output'], opset_version=11)
    
def deterministic_value_export_onnx_model(networks, input_dim, policy_dir):

    example = torch.rand(1, input_dim)  # network input dim
    output_onnx_model = policy_dir
    torch.onnx.export(networks.v, example, output_onnx_model, input_names=['input'],
                      output_names=['output'], opset_version=11)
    
def stochastic_export_policy_onnx_model(networks, input_dim, policy_dir):

    example = torch.rand(1, input_dim)  # network input dim
    output_onnx_model = policy_dir
    model = networks.policy
    export_model(model, example, output_onnx_model)

def stochastic_export_value_onnx_model(networks, input_dim, policy_dir):

    example = torch.rand(1, input_dim)  # network input dim
    output_onnx_model = policy_dir
    model = networks.v
    export_model(model, example, output_onnx_model)

def DSAC_policy_export_onnx_model(networks, input_dim, policy_dir, act_scale_factor,obs_scale_factor, bias):

    example = torch.rand(1, input_dim)  # network input dim
    output_onnx_model = policy_dir
    model = _InferenceHelper_Policy_DSAC(networks, act_scale_factor,obs_scale_factor,bias)
    torch.onnx.export(model, example, output_onnx_model, input_names=['input'], output_names=['output'],
                          opset_version=11)

def DSAC_Q1_export_onnx_model(networks, input_dim_obs, input_dim_act,policy_dir):

    example_obs_act = torch.rand(1, input_dim_obs+input_dim_act)  # network input dim
    output_onnx_model = policy_dir
    model = _InferenceHelper_Q_DSAC(networks.q1,input_dim_act)
    torch.onnx.export(model, example_obs_act, output_onnx_model, input_names=['input'], output_names=['output'],
                          opset_version=11)

def DSAC_Q2_export_onnx_model(networks, input_dim_obs, input_dim_act,policy_dir):

    example_obs_act = torch.rand(1, input_dim_obs+input_dim_act)  # network input dim
    output_onnx_model = policy_dir
    model = _InferenceHelper_Q_DSAC(networks.q2,input_dim_act)
    torch.onnx.export(model, example_obs_act, output_onnx_model, input_names=['input'], output_names=['output'],
                          opset_version=11)



if __name__=='__main__':

    # Load trained policy
    log_policy_dir = "//root/gops/results/idsim_multilane_vec/dsact_pi/12345_250000_2000_10_1_1_10_200000_before_plus_with_oppo_no_tar_pi_new_reward5_random_takeover_dec_acc_refv_8_short_ref_no_sur_punish_2_run0"
    args = __load_args(log_policy_dir)
    alg_name = args["algorithm"]
    alg_file_name = alg_name.lower()
    file = __import__(alg_file_name)
    ApproxContainer = getattr(file, "ApproxContainer")
    networks = ApproxContainer(**args)

    # Load trained policy
    log_path = log_policy_dir + "/apprfunc/apprfunc_{}.pkl".format('200000')  # network position
    networks.load_state_dict(torch.load(log_path))
    networks.eval()

    # create onnx model
    ### example of deterministic policy FHADP algorithm
    # input_dim = 202
    # policy_dir = '../../transform_onnx_network/idsim_policy.onnx'
    # value_dir = '../../transform_onnx_network/idsim_value.onnx'
    # fhadp2_policy_export_onnx_model(networks, input_dim, policy_dir)
    # deterministic_value_export_onnx_model(networks, input_dim, value_dir)

    # DSAC
    obs_dim = args.obsv_dim
    act_dim = 2
    policy_dir = '/root/gops/exp/onnx_network/idsim_DSAC_policy.onnx'
    Q1_dir = '/root/gops/exp/onnx_network/idsim_DSAC_Q1.onnx'
    Q2_dir = '/root/gops/exp/onnx_network/idsim_DSAC_Q2.onnx'
    action_upper_bound = args["env_config"]["action_upper_bound"]
    action_lower_bound = args["env_config"]["action_lower_bound"]    
    action_scale_factor = torch.tensor([action_upper_bound[0]-action_lower_bound[0],action_upper_bound[1]-action_lower_bound[1]]).float().abs()
    action_scale_factor = action_scale_factor/2
    action_scale_bias = torch.tensor([(action_upper_bound[0]+action_lower_bound[0])/2,(action_upper_bound[1]+action_lower_bound[1])/2]).float()
    DSAC_policy_export_onnx_model(networks, obs_dim, policy_dir,action_scale_factor,action_scale_bias)
    DSAC_Q1_export_onnx_model(networks, obs_dim, act_dim, Q1_dir)
    DSAC_Q2_export_onnx_model(networks, obs_dim, act_dim, Q2_dir)



    # ### example of stochastic policy sac algorithm
    # input_dim = 50
    # policy_dir = '../../transform_onnx_network/network_sac_ziqing.onnx'
    # deterministic_stochastic_export_onnx_model(networks, input_dim, policy_dir)

    # load onnx model for test
    ### example of deterministic policy FHADP algorithm
    # ort_session_policy = ort.InferenceSession("../../transform_onnx_network/idsim_policy.onnx")
    # example_policy = np.random.randn(202).astype(np.float32)
    # inputs_policy = {ort_session_policy.get_inputs()[0].name: example_policy}
    # outputs_policy = ort_session_policy.run(None, inputs_policy)
    # print(outputs_policy[0])
    # action = networks.policy(torch.tensor(example_policy).unsqueeze(0))
    # print(action[0])

    # ort_session_value = ort.InferenceSession("../../transform_onnx_network/idsim_value.onnx")
    # example_value = np.random.randn(1, 202).astype(np.float32)
    # inputs_value = {ort_session_value.get_inputs()[0].name: example_value}
    # outputs_value = ort_session_value.run(None, inputs_value)
    # print(outputs_value[0])
    # value = networks.v(torch.tensor(example_value))
    # print(value)

    # ### example of stochastic policy sac algorithm
    # ort_session = ort.InferenceSession("../../transform_onnx_network/network_sac_ziqing.onnx")
    # example1 = np.random.randn(1, 50).astype(np.float32)
    # inputs = {ort_session.get_inputs()[0].name: example1}
    # outputs = ort_session.run(None, inputs)
    # print(outputs)
    # action = networks.policy(torch.tensor(example1))
    # act_dist = model.get_act_dist(action).mode()
    # print(act_dist)

    # ### example of DSAC algorithm
    ort_session_policy = ort.InferenceSession(policy_dir)
    example_policy = np.random.randn(1,obs_dim).astype(np.float32)
    obs_file = '/root/gops/exp/onnx_network/idsim2023-12-19_23-17-39.csv'
    obs = pd.read_csv(obs_file,header=None).values
    init_obs = obs[-1,:obs_dim]
    init_obs = init_obs.astype(np.float32)
    init_obs = init_obs.reshape(1,-1)
   # example_policy = init_obs



    inputs_policy = {ort_session_policy.get_inputs()[0].name: example_policy}
    outputs_policy = ort_session_policy.run(None, inputs_policy)
    print(outputs_policy[0])
    logits = networks.policy(torch.tensor(example_policy).unsqueeze(0))
    action,_ = torch.chunk(logits,2,-1) 
    action = torch.tanh(action)
    action = action*action_scale_factor + action_scale_bias
    print(action)

    ort_session_value = ort.InferenceSession(Q1_dir)
    example_obs_act = np.random.randn(1, obs_dim+act_dim).astype(np.float32)
    inputs_value = {ort_session_value.get_inputs()[0].name: example_obs_act,} 
    outputs_value = ort_session_value.run(None, inputs_value)
    print(outputs_value[0])
    value = networks.q1(torch.tensor(example_obs_act)[:,:-act_dim],torch.tensor(example_obs_act)[:,-act_dim:])
    print(value)
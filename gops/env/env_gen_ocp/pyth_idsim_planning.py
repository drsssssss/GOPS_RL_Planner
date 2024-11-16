from dataclasses import dataclass
import copy
from pathlib import Path
from typing import Any, Dict, List, Generic, Optional, Tuple, Union
from typing_extensions import Self

import gym
import time
import numpy as np
import torch
from gops.env.env_gen_ocp.pyth_base import (Context, ContextState, Env, State, stateType)
from gops.env.env_gen_ocp.resources.idsim_tags import reward_tags
from gops.env.env_gen_ocp.pyth_idsim import idSimEnv, get_idsimcontext, reward_type
from idsim.config import Config
from idsim.envs.env import CrossRoad
from idsim_model.model import IdSimModel
from idsim_model.model_context import Parameter, BaseContext
from idsim_model.crossroad.context import CrossRoadContext
from idsim_model.multilane.context import MultiLaneContext
from idsim_model.model_context import State as ModelState
from idsim_model.params import ModelConfig


reward_type = [
    "env_scaled_reward_done",
    "env_scaled_reward_collision",
    "env_reward_collision_risk",
    "env_scaled_reward_boundary",
    "env_scaled_reward_step",
    "env_scaled_reward_dist_lat",
    "env_scaled_reward_vel_long",
    "env_scaled_reward_head_ang",
    "env_scaled_reward_yaw_rate",
    "env_scaled_reward_steering",
    "env_scaled_reward_acc_long",
    "env_scaled_reward_delta_steer",
    "env_scaled_reward_jerk",
]


def cal_ave_exec_time(print_interval=100):
    def decorator(func):
        total_time = 0
        count = 0

        def wrapper(*args, **kwargs):
            nonlocal total_time, count
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            total_time += execution_time
            count += 1
            if count % print_interval == 0:
                print(f"Average execution time after {count} steps: {total_time / count:.9f} seconds")
            return result
        return wrapper
    return decorator


def generate_bezier_curve_with_phi(origin_point:np.array, dest_point:np.array, n_points=100) -> np.array:
    x0, y0, phi0, v_o = origin_point
    x3, y3, phi3, v_d = dest_point
    delta_v = v_d - v_o
    p1_x = x0 + np.cos(phi0) * 0.4 * np.linalg.norm([x3 - x0, y3 - y0])
    p1_y = y0 + np.sin(phi0) * 0.4 * np.linalg.norm([x3 - x0, y3 - y0])
    
    p2_x = x3 - np.cos(phi3) * 0.4 * np.linalg.norm([x3 - x0, y3 - y0])
    p2_y = y3 - np.sin(phi3) * 0.4 * np.linalg.norm([x3 - x0, y3 - y0])
    
    P0 = np.array([x0, y0])
    P1 = np.array([p1_x, p1_y])
    P2 = np.array([p2_x, p2_y])
    P3 = np.array([x3, y3])

    t_values = np.linspace(0, 1, n_points)

    bezier_points = []
    for t in t_values:
        x = (1 - t)**3 * P0[0] + 3 * (1 - t)**2 * t * P1[0] + 3 * (1 - t) * t**2 * P2[0] + t**3 * P3[0]
        y = (1 - t)**3 * P0[1] + 3 * (1 - t)**2 * t * P1[1] + 3 * (1 - t) * t**2 * P2[1] + t**3 * P3[1]

        dx = 3 * (1 - t)**2 * (P1[0] - P0[0]) + 6 * (1 - t) * t * (P2[0] - P1[0]) + 3 * t**2 * (P3[0] - P2[0])
        dy = 3 * (1 - t)**2 * (P1[1] - P0[1]) + 6 * (1 - t) * t * (P2[1] - P1[1]) + 3 * t**2 * (P3[1] - P2[1])

        phi = np.arctan2(dy, dx)
        bezier_points.append(np.array([x, y, phi, v_o + delta_v * t]))

    bezier_points = np.array(bezier_points)
    return bezier_points


def dense_ref_by_bessel(ref_param: np.ndarray, ratio_list: list = [1]):
    """
    Densify reference parameters by add Bessel curves.

    Parameters:
    ref_param (np.ndarray): Input reference parameters with shape [R, 2N+1, 4].
                            Each element represents [ref_x, ref_y, ref_phi, ref_v].

    Returns:
    np.ndarray: Densified reference parameters with shape [R+4, 2N+1, 4].
                Each element represents [ref_x, ref_y, ref_phi, ref_v].
    """
    bezier_list=[]
    num_point = ref_param.shape[-2]
    for sample_ratio in ratio_list:
        target_index = int(sample_ratio * num_point)
        for i in range(ref_param.shape[0]):
            if i == 0:
                ref_bezier = generate_bezier_curve_with_phi(ref_param[i][0], ref_param[i+1][target_index-1],target_index)
                if int(num_point-target_index)!=0:
                    bezier_list.append(np.concatenate((ref_bezier,ref_param[i+1][-int(num_point-target_index):])))
                else:
                    bezier_list.append(ref_bezier)
            elif i==ref_param.shape[0]-1:
                ref_bezier = generate_bezier_curve_with_phi(ref_param[i][0], ref_param[i-1][target_index-1],target_index)
                if int(num_point-target_index)!=0:
                    bezier_list.append(np.concatenate((ref_bezier,ref_param[i-1][-int(num_point-target_index):])))
                else:
                    bezier_list.append(ref_bezier)
            else:
                ref_bezier = generate_bezier_curve_with_phi(ref_param[i][0], ref_param[i-1][target_index-1],target_index)
                if int(num_point-target_index)!=0:
                    bezier_list.append(np.concatenate((ref_bezier,ref_param[i-1][-int(num_point-target_index):])))
                else:
                    bezier_list.append(ref_bezier)
                
                ref_bezier = generate_bezier_curve_with_phi(ref_param[i][0], ref_param[i+1][target_index-1],target_index)
                if int(num_point-target_index)!=0:
                    bezier_list.append(np.concatenate((ref_bezier,ref_param[i+1][-int(num_point-target_index):])))
                else:
                    bezier_list.append(ref_bezier)
                    
    return np.concatenate([np.array(bezier_list),ref_param])


def dense_ref_by_boundary(ref_param: np.ndarray):
    """
    Densify reference parameters by add boundaries.

    Parameters:
    ref_param (np.ndarray): Input reference parameters with shape [R, 2N+1, 4].
                            Each element represents [ref_x, ref_y, ref_phi, ref_v].

    Returns:
    np.ndarray: Densified reference parameters with shape [R+2, 2N+1, 4].
                Each element represents [ref_x, ref_y, ref_phi, ref_v].
    """

    A, B, C = ref_param.shape
    ret = np.zeros((2*A - 1, B, C))

    for j in range(A):
        ret[2*j, :, :] = ref_param[j, :, :]
    for j in range(A - 1):
        ret[2*j + 1, :, :] = (ref_param[j, :, :] + ref_param[j + 1, :, :]) / 2

    return ret


class idSimEnvPlanning(idSimEnv):
    def __init__(self, env_config: Config, model_config: ModelConfig, 
                 scenario: str, rou_config: Dict[str, Any]=None, env_idx: int=None, scenerios_list: List[str]=None):
        super(idSimEnvPlanning, self).__init__(env_config, model_config, scenario, rou_config, env_idx, scenerios_list)

        self.ref_replann = True
        self.ref = None
        self.planning_horizon = 0
        self.cum_reward = None
        self.cum_reward_info = None
        
        self.set_dense_ref_function()

    def set_dense_ref_function(self):
        if self.env_config.dense_ref_mode == "boundary":
            self.dense_ref = dense_ref_by_boundary
        elif self.env_config.dense_ref_mode == "bessel":
            self.dense_ref = dense_ref_by_bessel
        else:
            raise ValueError(f"Unknown dense_ref_mode: {self.env_config.dense_ref_mode}")
        print(f"INFO: dense ref mode: {self.env_config.dense_ref_mode}")

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        self.lc_cooldown_counter = 0
        if self.rou_config is not None:
            if self.engine is None:
                # first create engine
                print("INFO: change rou")
                self.change_rou_file()
            else:
                print(self.engine.context.episode_count, self.config.scenario_reuse)
                if (self.engine.context.episode_count+1) % self.config.scenario_reuse == 0:
                    # need to change new map
                    print("INFO: change rou")
                    self.change_rou_file()
        obs, info = super(idSimEnv, self).reset(**kwargs)
        env_context = self.engine.context
        vehicle = env_context.vehicle
        self.ref_index = None
        if self.scenario == "multilane" and self.use_random_ref_param and not env_context.vehicle.in_junction:
            lane_list = env_context.scenario.network.get_edge_lanes(
                vehicle.edge, vehicle.v_class)
            cur_index = lane_list.index(vehicle.lane)
            self.ref_index = np.random.choice([0,1,-1]) + cur_index # only allow lane change by 1
            self.ref_index = np.clip(self.ref_index, 0, len(lane_list)-1)
        else:
            self.ref_index = np.random.choice(
                np.arange(self.model_config.num_ref_lines)
            ) if self.use_random_ref_param else None
        if self.random_ref_v and not env_context.vehicle.in_junction:  # TODO: check if this is correct
            ref_v = np.random.uniform(*self.ref_v_range)
            self.model_config.ref_v_lane = float(ref_v)
            self.env_config.ref_v = float(ref_v)
            # print(f"INFO: change ref_v to {ref_v}")
        self.new_ref_index = self.ref_index
            
        self._state = self._get_state_from_idsim(ref_index_param=self.ref_index)
        self._fix_state()
        self._info = self._get_info(info)
        return self._get_obs(), self._info
    
    def end_planning(self):
        self.ref_replann = True
        self.planning_horizon = 0
            
    # @cal_ave_exec_time(print_interval=1000)
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, terminated, truncated, info = super(idSimEnv, self).step(action)

        self._state = self._get_state_from_idsim(ref_index_param=self.ref_index)

        #  ----- ref replanning-----
        if self.ref_replann:
            self.ref_replann = False
            self.ref = self.dense_ref(self._state.context_state.reference)
            self.cum_reward = [0] * self.ref.shape[0]
            init_reward_info = {key: 0 for key in reward_type}
            self.cum_reward_info = [init_reward_info.copy() for _ in range(self.ref.shape[0])]

        # ----- recalculate ref and time horizon -----
        replan_state = copy.deepcopy(self._state)
        replan_state.context_state.t = np.array(self.planning_horizon, dtype=np.int32)
        replan_state.context_state.reference = self.ref
        self.planning_horizon += 1

        # ----- cal the next_obs, reward -----
        # 
        # test = self._get_model_free_reward(action)
        # reward_model_free_list, mf_info_list = [], []
        reward_model_free_list, mf_info_list = zip(*[self._get_model_free_reward_by_state(replan_state, action, np.array(idx)) for idx in np.arange(self.ref.shape[0])])
        
        for idx in range(self.ref.shape[0]):
            self.cum_reward[idx] += reward_model_free_list[idx] + reward
            for key in mf_info_list[idx]:
                if key in reward_type:
                    self.cum_reward_info[idx][key] += mf_info_list[idx][key]
                else:
                    self.cum_reward_info[idx][key] = mf_info_list[idx][key]
            for key in info:
                if key in reward_type:
                    self.cum_reward_info[idx][key] += info[key]
                else:
                    self.cum_reward_info[idx][key] = info[key]
        
        done = terminated or truncated
        if truncated:
            info["TimeLimit.truncated"] = True # for gym

        opt_ref_index = np.argmax(self.cum_reward)
        self._info = self._get_info(self.cum_reward_info[opt_ref_index])
        total_reward = self.cum_reward[opt_ref_index]
        # if not terminated:
        #     total_reward = np.maximum(total_reward, 0.05)

        mid_index = self._state.context_state.reference.shape[0] // 2
        self._state = self._get_state_from_idsim(ref_index_param=mid_index) # get state using mid_index to calculate obs

        return self._get_obs(), total_reward, done, self._info

    def _get_model_free_reward_by_state(self, state: State, action: np.ndarray, ref_index_param: np.ndarray) -> float:
        idsim_context = get_idsimcontext(
            State.stack([state]), 
            mode="batch", 
            scenario=self.scenario
        )
        state.context_state.ref_index_param = ref_index_param
        idsim_context = get_idsimcontext(
            State.stack([state]), 
            mode="batch", 
            scenario=self.scenario
        )

        reward, info = self.model_free_reward(
            context=idsim_context,
            last_last_action=state.robot_state[..., -4:-2][None, :], # absolute action
            last_action=state.robot_state[..., -2:][None, :], # absolute action
            action=action[None, :] # incremental action
        )
        
        return reward, info


def env_creator(**kwargs):
    """
    make env `pyth_idsim`
    """
    assert "env_config" in kwargs.keys(), "env_config must be specified"
    env_config = copy.deepcopy(kwargs["env_config"])

    assert "env_scenario" in kwargs.keys(), "env_scenario must be specified"
    env_scenario = kwargs["env_scenario"]

    assert 'scenario_root' in env_config, "scenario_root must be specified in env_config"
    env_config['scenario_root'] = Path(env_config['scenario_root'])
    env_config = Config.from_partial_dict(env_config)

    assert "env_model_config" in kwargs.keys(), "env_model_config must be specified"
    model_config = copy.deepcopy(kwargs["env_model_config"])
    model_config = ModelConfig.from_partial_dict(model_config)

    rou_config = kwargs["rou_config"] if "rou_config" in kwargs.keys() else None

    env_idx = kwargs["env_idx"] if "env_idx" in kwargs.keys() else 0

    scenerios_list = kwargs["scenerios_list"] if "scenerios_list" in kwargs.keys() else None
    env = idSimEnvPlanning(env_config, model_config, env_scenario, rou_config, env_idx, scenerios_list)
    return env
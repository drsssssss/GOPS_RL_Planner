from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Generic, Optional, Tuple, Union
from typing_extensions import Self

import gym
import time
import numpy as np
import torch
from gops.env.env_gen_ocp.pyth_base import (Context, ContextState, Env, State, stateType)
from gops.env.env_gen_ocp.resources.idsim_tags import reward_tags
from gops.env.env_gen_ocp.pyth_idsim import idSimEnv
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


class idSimEnvPlanning(idSimEnv):
    def __new__(cls, env_config: Config, model_config: Dict[str, Any], 
                scenario: str, rou_config: Dict[str, Any]=None, env_idx: int=None, scenerios_list: List[str]=None):
        return super(idSimEnvPlanning, cls).__new__(cls, env_config)
    
    def __init__(self, env_config: Config, model_config: ModelConfig, 
                 scenario: str, rou_config: Dict[str, Any]=None, env_idx: int=None, scenerios_list: List[str]=None):
        super(idSimEnvPlanning, self).__init__(env_config, model_config, scenario, rou_config, env_idx, scenerios_list)
        self.dense_ref_mode = env_config.dense_ref_mode
        print(f"INFO: dense ref mode={self.dense_ref_mode}")
        
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
    
    def unfreeze_lc(self):
        if self.allow_lc:
            self.ref_index = self.new_ref_index
            self.lc_cooldown_counter = 0
            self.allow_lc = False
        
    # @cal_ave_exec_time(print_interval=1000)
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, terminated, truncated, info = super(idSimEnv, self).step(action)

        # ----- cal the next_obs, reward -----
        self.lc_cooldown_counter += 1
        if self.lc_cooldown_counter > self.lc_cooldown:
            # lane change is allowable
            if not self.engine.context.vehicle.in_junction and self.use_random_ref_param and np.random.rand() < self.random_ref_probability :
                self.new_ref_index = np.random.choice(np.arange(self.model_config.num_ref_lines))
                if self.new_ref_index != self.ref_index:
                    # allow lane change
                    self.allow_lc = True
        if self.choose_closest:
            if self.dense_ref:
                # TODO: implement this
                # dense_state = self.
                self._state.context_state.reference = self._state.context_state.reference
            self.choose_closest_lane()
        # reward_model, reward_details = self._get_reward(action)
           
            # check if the ref_index is the same as the closest lane
        reward_model_free, mf_info = self._get_model_free_reward(action)
        info.update(mf_info)

        # info["reward_details"] = dict(
        #     zip(reward_tags, [i.item() for i in reward_details])
        # )
        done = terminated or truncated
        if truncated:
            info["TimeLimit.truncated"] = True # for gym

        self._info = self._get_info(info)
        total_reward = reward + reward_model_free
        # if not terminated:
        #     total_reward = np.maximum(total_reward, 0.05)

        if self.mid_line_obs:
            mid_index = self._state.context_state.reference.shape[0] // 2
            self._state = self._get_state_from_idsim(ref_index_param=mid_index) # get state using mid_index to calculate obs
        return self._get_obs(), total_reward, done, self._info

    def choose_closest_lane(self):
        # set self.ref_index to the closest lane according to self._state.context_state.reference and self._state.robot_state
        self.ref_index = np.argmin(
            np.linalg.norm(
                self._state.robot_state[:2] - self._state.context_state.reference[:, 0, :2], axis=1
            )
        )
        
def env_creator(**kwargs):
    """
    make env `pyth_idsim`
    """
    assert "env_config" in kwargs.keys(), "env_config must be specified"
    env_config = deepcopy(kwargs["env_config"])

    assert "env_scenario" in kwargs.keys(), "env_scenario must be specified"
    env_scenario = kwargs["env_scenario"]

    assert 'scenario_root' in env_config, "scenario_root must be specified in env_config"
    env_config['scenario_root'] = Path(env_config['scenario_root'])
    env_config = Config.from_partial_dict(env_config)

    assert "env_model_config" in kwargs.keys(), "env_model_config must be specified"
    model_config = deepcopy(kwargs["env_model_config"])
    model_config = ModelConfig.from_partial_dict(model_config)

    rou_config = kwargs["rou_config"] if "rou_config" in kwargs.keys() else None

    env_idx = kwargs["env_idx"] if "env_idx" in kwargs.keys() else 0

    scenerios_list = kwargs["scenerios_list"] if "scenerios_list" in kwargs.keys() else None
    env = idSimEnvPlanning(env_config, model_config, env_scenario, rou_config, env_idx, scenerios_list)
    return env
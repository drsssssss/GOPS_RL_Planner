from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Generic, Optional, Tuple, Union
from typing_extensions import Self

import gym
import numpy as np
import torch
from gops.env.env_gen_ocp.pyth_base import (Context, ContextState, Env, State, stateType)
from gops.env.env_gen_ocp.resources.idsim_tags import reward_tags
from idsim.config import Config
from idsim.envs.env import CrossRoad
from idsim_model.model import IdSimModel
from idsim_model.model_context import Parameter, BaseContext
from idsim_model.crossroad.context import CrossRoadContext
from idsim_model.multilane.context import MultiLaneContext
from idsim_model.model_context import State as ModelState
from idsim_model.params import ModelConfig



@dataclass
class idSimContextState(ContextState[stateType], Generic[stateType]):
    light_param: Optional[stateType] = None
    ref_index_param: Optional[stateType] = None
    real_t: Union[int, stateType] = 0


class idSimContext(Context):
    def reset(self) -> idSimContextState[np.ndarray]:
        pass

    def step(self) -> idSimContextState[np.ndarray]:
        pass

    def get_zero_state(self) -> idSimContextState[np.ndarray]:
        pass
    

class idSimEnv(CrossRoad, Env):
    def __new__(cls, env_config: Config, model_config: Dict[str, Any], scenario: str) -> Self:
        return super(idSimEnv, cls).__new__(cls, env_config)
    
    def __init__(self, env_config: Config, model_config: ModelConfig, scenario: str):
        super(idSimEnv, self).__init__(env_config)
        self.model_config = model_config
        self.scenario = scenario

        self._state = None
        # get observation_space
        self.model = IdSimModel(env_config, model_config)
        obs_dim = self.model.obs_dim
        self.use_random_ref_param = env_config.use_multiple_path_for_multilane
        self.random_ref_probability = env_config.random_ref_probability
        if self.random_ref_probability > 0.0:
            print(f'INFO: randomly choosing reference when resetting env')
        if self.random_ref_probability > 0.0:
            print(f'INFO: randomly choosing reference when stepping at P={self.random_ref_probability}')

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.context = idSimContext() # fake idsim context
        self.set_scenario(scenario)
        self.ref_index = None
    
    def reset(self) -> Tuple[np.ndarray, dict]:
        obs, info = super(idSimEnv, self).reset()
        self.ref_index = np.random.choice(
            np.arange(self.model_config.num_ref_lines)
        ) if self.use_random_ref_param else None
        self._state = self._get_state_from_idsim(ref_index_param=self.ref_index)
        return self._get_obs(), self._get_info(info)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, terminated, truncated, info = super(idSimEnv, self).step(action)
        if self.use_random_ref_param and np.random.rand() < self.random_ref_probability :
            self.ref_index = np.random.choice(np.arange(self.model_config.num_ref_lines))

        reward_model, reward_details = self._get_reward(action)
        self._state = self._get_state_from_idsim(ref_index_param=self.ref_index)
        reward_model_free = self._get_model_free_reward(action)

        info["reward_details"] = dict(
            zip(reward_tags, [i.item() for i in reward_details])
        )
        done = terminated or truncated
        return self._get_obs(), reward + reward_model_free, done, self._get_info(info)

    def set_ref_index(self, ref_index: int):
        self.ref_index = ref_index
    
    def _get_info(self, info) -> dict:
        info.update(Env._get_info(self))
        return info
    
    def _get_obs(self) -> np.ndarray:
        idsim_context = get_idsimcontext(
            State.stack([self._state.array2tensor()]), mode="batch", scenario=self.scenario)
        model_obs = self.model.observe(idsim_context)
        return model_obs.numpy().squeeze(0)

    def _get_reward(self, action: np.ndarray) -> Tuple[float, Tuple]:
        cur_state = self._state.array2tensor()
        next_state = self._get_state_from_idsim(ref_index_param=self.ref_index).array2tensor()
        idsim_context = get_idsimcontext(State.stack([next_state]), mode="batch", scenario=self.scenario)
        action = torch.tensor(action)
        reward_details = self.model.reward_nn_state(
            context=idsim_context,
            last_last_action=cur_state.robot_state[..., -4:-2].unsqueeze(0), # absolute action
            last_action=cur_state.robot_state[..., -2:].unsqueeze(0), # absolute action
            action=action.unsqueeze(0) # incremental action
        )
        return reward_details[0].item(), reward_details
    
    def _get_model_free_reward(self, action: np.ndarray) -> float:
        idsim_context = get_idsimcontext(
            State.stack([self._state]), 
            mode="batch", 
            scenario=self.scenario
        )
        reward = self.model_free_reward(
            context=idsim_context,
            last_last_action=self._state.robot_state[..., -4:-2][None, :], # absolute action
            last_action=self._state.robot_state[..., -2:][None, :], # absolute action
            action=action[None, :] # incremental action
        )
        return reward
    
    def _get_terminated(self) -> bool:
        """abandon this function, use terminated from idsim instead"""
        ...
    
    def _get_state_from_idsim(self, ref_index_param=None) -> State:
        if self.scenario == "crossroad":
            idsim_context = CrossRoadContext.from_env(self, self.model_config, ref_index_param)
        elif self.scenario == "multilane":
            idsim_context = MultiLaneContext.from_env(self, self.model_config, ref_index_param)
        else:
            raise NotImplementedError

        return State(
            robot_state=torch.concat([
                idsim_context.x.ego_state, 
                idsim_context.x.last_last_action, 
                idsim_context.x.last_action],
            dim=-1),
            context_state=idSimContextState(
                reference=idsim_context.p.ref_param, 
                constraint=idsim_context.p.sur_param,
                light_param=idsim_context.p.light_param, 
                ref_index_param=idsim_context.p.ref_index_param,
                real_t = torch.tensor(idsim_context.t).int(),
                t = torch.tensor(idsim_context.i).int()
            )
        ).tensor2array()

    def get_zero_state(self) -> State[np.ndarray]:
        if self._state is None:
            self.reset()
        return State(
            robot_state=np.zeros_like(self._state.robot_state, dtype=np.float32),
            context_state=idSimContextState(
                reference=np.zeros_like(self._state.context_state.reference, dtype=np.float32),
                constraint=np.zeros_like(self._state.context_state.constraint, dtype=np.float32),
                t=np.zeros_like(self._state.context_state.t, dtype=np.int64),
                light_param=np.zeros_like(self._state.context_state.light_param, dtype=np.float32),
                ref_index_param=np.zeros_like(self._state.context_state.ref_index_param, dtype=np.int64),
                real_t=np.zeros_like(self._state.context_state.real_t, dtype=np.int64)
            )
        )
    # close
    def close(self) -> None:
        super(idSimEnv, self).close()


def get_idsimcontext(state: State, mode: str, scenario: str) -> BaseContext:
    if scenario == "crossroad":
        Context = CrossRoadContext
    elif scenario == "multilane":
        Context = MultiLaneContext
    else:
        raise NotImplementedError
    if mode == "full_horizon":
        context = Context(
            x = ModelState(
                ego_state = state.robot_state[..., :-4].unsqueeze(0),
                last_last_action = state.robot_state[..., -4:-2].unsqueeze(0),
                last_action = state.robot_state[..., -2:].unsqueeze(0)
            ),
            p = Parameter(
                ref_param = state.context_state.reference.unsqueeze(0),
                sur_param = state.context_state.constraint.unsqueeze(0),
                light_param = state.context_state.light_param.unsqueeze(0),
                ref_index_param = state.context_state.ref_index_param.unsqueeze(0)
            ),
            t = state.context_state.real_t.unsqueeze(0),
            i = state.context_state.t.long()
        )
    elif mode == "batch":
        if isinstance(state.context_state.t, np.ndarray):
            assert np.unique(state.context_state.t).shape[0] == 1, "batch mode only support same t"
        elif isinstance(state.context_state.t, torch.Tensor):
            assert state.context_state.t.unique().shape[0] == 1, "batch mode only support same t"
        else:
            raise NotImplementedError
        context = Context(
            x = ModelState(
                ego_state = state.robot_state[..., :-4],
                last_last_action = state.robot_state[..., -4:-2],
                last_action = state.robot_state[..., -2:]
            ),
            p = Parameter(
                ref_param = state.context_state.reference,
                sur_param = state.context_state.constraint,
                light_param = state.context_state.light_param,
                ref_index_param = state.context_state.ref_index_param
            ),
            t = state.context_state.real_t,
            i = state.context_state.t[0]
        )
    else:
        raise NotImplementedError
    return context


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

    env = idSimEnv(env_config, model_config, env_scenario)
    return env
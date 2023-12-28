from dataclasses import dataclass
from typing import Optional, Any, Union
from gops.env.env_gen_ocp.env_model.pyth_base_model import RobotModel, EnvModel
from gops.env.env_gen_ocp.pyth_idsim import get_idsimcontext
from gops.env.env_gen_ocp.pyth_base import State
from idsim_model.params import ModelConfig
from idsim.config import Config
import numpy as np
import torch

from idsim_model.model_context import State as ModelState
from idsim_model.model import IdSimModel


@dataclass
class FakeModelContext:
    x: Optional[torch.Tensor] = None


class idSimRobotModel(RobotModel):
    def __init__(self,
        idsim_model: IdSimModel,
    ):
        self.robot_state_dim = 6 + 2 * 2
        self.robot_state_lower_bound = torch.tensor([-np.inf] * self.robot_state_dim, dtype=torch.float32)
        self.robot_state_upper_bound = torch.tensor([np.inf] * self.robot_state_dim, dtype=torch.float32)
        self.idsim_model = idsim_model
        self.Ts = idsim_model.Ts
        self.vehicle_spec = idsim_model.vehicle_spec
        self.fake_model_context = FakeModelContext()

    def get_next_state(self, robot_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        self.fake_model_context.x = ModelState(
            ego_state = robot_state[..., :-4],
            last_last_action = robot_state[..., -4:-2],
            last_action = robot_state[..., -2:]
        )
        model_state = self.idsim_model.dynamics(self.fake_model_context, action)
        robot_state = torch.concat([model_state.ego_state, model_state.last_last_action, model_state.last_action], dim=-1)
        return robot_state


class idSimEnvModel(EnvModel):
    dt: Optional[float]
    action_dim: int
    obs_dim: int
    robot_model: RobotModel

    def __init__(
        self,
        *,
        device: Union[torch.device, str, None] = None,
        **kwargs: Any,
    ):
        assert "env_config" in kwargs.keys(), "env_config must be specified"
        env_config = kwargs["env_config"]
        env_config = Config.from_partial_dict(env_config)

        assert "env_scenario" in kwargs.keys(), "env_scenario must be specified"
        self.env_scenario = kwargs["env_scenario"]

        assert "env_model_config" in kwargs.keys(), "env_model_config must be specified"
        model_config = kwargs["env_model_config"]
        model_config = ModelConfig.from_partial_dict(model_config)

        self.dt = env_config.dt
        self.action_dim = 2
        self.obs_dim = 1 # FIXME: this is a hack

        super().__init__(
            action_lower_bound = env_config.action_lower_bound,
            action_upper_bound = env_config.action_upper_bound,
            device = device,
        )

        self.idsim_model = IdSimModel(env_config, model_config)
        self.robot_model = idSimRobotModel(idsim_model = self.idsim_model)

    def get_obs(self, state: State) -> torch.Tensor:
        return self.idsim_model.observe(get_idsimcontext(state, mode = 'batch', scenario=self.env_scenario))
        
    def get_reward(self, state: State, action: torch.Tensor, mode: str = "full_horizon",
                   return_details: bool = False) -> torch.Tensor:
        next_state = self.get_next_state(state, action)
        if mode == "full_horizon":
            rewards = self.idsim_model.reward_full_horizon(
                context_full = get_idsimcontext(next_state, mode = mode, scenario=self.env_scenario),
                last_last_action_full = state.robot_state[..., -4:-2], # absolute action
                last_action_full = state.robot_state[..., -2:], # absolute action
                action_full = action # incremental action
            )
        elif mode == "batch":
            rewards = self.idsim_model.reward_nn_state(
                context = get_idsimcontext(next_state, mode = mode, scenario=self.env_scenario),
                last_last_action = state.robot_state[..., -4:-2], # absolute action
                last_action = state.robot_state[..., -2:], # absolute action
                action = action # incremental action
            )
        else:
            raise NotImplementedError
        
        if return_details:
            return rewards[0], rewards
        else:
            return rewards[0]

    def get_terminated(self, state: State) -> torch.bool:
        # only support batched state
        return torch.zeros(state.robot_state.shape[0], dtype=torch.bool)
    
    def forward(self, obs, action, done, info):
        state = info["state"]
        next_state = self.get_next_state(state, action)
        next_obs = self.get_obs(next_state)
        reward, reward_details = self.get_reward(state, action, mode = "batch", return_details=True)
        terminated = self.get_terminated(state)
        next_info = {}
        next_info["state"] = next_state
        next_info["reward_details"] = reward_details
        return next_obs, reward, terminated, next_info


def env_model_creator(**kwargs):
    """
    make env model `pyth_idsim_model`
    """
    return idSimEnvModel(**kwargs)

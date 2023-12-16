#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Implementation of optimal controller based on MPC
#  Update: 2023-12-03, Zhilong Zheng: create OptController4idSim

import zmq
import numpy as np

from gops.env.env_gen_ocp.resources.idsim_mpc import MPCInput, MPCOutput, ReferencePath, SurroundingVehicle, EgoState, PreviousAction, StatePoint
from gops.env.env_gen_ocp.resources.idsim_config import get_idsim_env_config, get_idsim_model_config, pre_horizon, delta_t
from idsim.config import Config


class OptController4idSim:
    """
        An MPC-based optimal controller specialized for idSim envs.
        It requires a running MPC server from the [MPC-GOPS](https://gitee.com/tsinghua-university-iDLab-GOPS/mpc-gops.git) repo.
        It works by sending necessary information to and receiving the optimal action from the MPC server.
    """

    def __init__(
        self, 
        scenario: str, 
        env_config: Config,
        *,
        protocol: str = "tcp", 
        interface: str = "127.0.0.1", 
        port: int = 8000
    ):
        assert scenario in ["crossroad", "multilane"], "Invalid scenario."
        self.scenario = scenario
        self.socket = zmq.Context().socket(zmq.REQ)
        self.socket.connect(f"{protocol}://{interface}:{port}")
        self.env_config = env_config
        self.action_center = (np.array(env_config.action_upper_bound) + np.array(env_config.action_lower_bound)) / 2
        self.action_half_range = (np.array(env_config.action_upper_bound) - np.array(env_config.action_lower_bound)) / 2

    def __call__(self, x) -> np.ndarray:
        ego_state = EgoState(
            x=x.robot_state[0],
            y=x.robot_state[1],
            phi=x.robot_state[4],
            v_long=x.robot_state[2],
            v_lat=x.robot_state[3],
            omega=x.robot_state[5]
        )

        previous_action = PreviousAction(
            front_wheel_angle=x.robot_state[9],
            acceleration=x.robot_state[8]
        )
        pre_action = x.robot_state[8:10]

        ref_path = ReferencePath()
        ref_index = x.context_state.ref_index_param
        ref_path.path_points.extend(
            [StatePoint(
                x=x.context_state.reference[ref_index, i, 0],
                y=x.context_state.reference[ref_index, i, 1],
                phi=x.context_state.reference[ref_index, i, 2],
                speed=x.context_state.reference[ref_index, i, 3]
            ) for i in range(pre_horizon + 1)]
        )

        num_sur_veh = x.context_state.constraint.shape[1]
        sur_vehicles = []
        for i in range(num_sur_veh):
            sur_vehicle = SurroundingVehicle()
            sur_vehicle.predicted_trajectory.extend(
                [StatePoint(
                    x=x.context_state.constraint[j, i, 0],
                    y=x.context_state.constraint[j, i, 1],
                    phi=x.context_state.constraint[j, i, 2]
                ) for j in range(pre_horizon + 1)]
            )
            sur_vehicle.length = x.context_state.constraint[0, i, 4]
            sur_vehicle.width = x.context_state.constraint[0, i, 5]
            sur_vehicles.append(sur_vehicle)

        mpc_input_msg = MPCInput()
        mpc_input_msg.ego_state.CopyFrom(ego_state)
        mpc_input_msg.previous_action.CopyFrom(previous_action)
        mpc_input_msg.reference_path.CopyFrom(ref_path)
        mpc_input_msg.surrounding_vehicles.extend(sur_vehicles)
        mpc_input_msg.time_step = delta_t
        mpc_input_msg.horizon = pre_horizon
        mpc_input_msg.mode = "junction" if self.scenario == "crossroad" else "segment"
        mpc_input_msg.max_acceleration = self.env_config.real_action_upper_bound[0]
        mpc_input_msg.max_deceleration = self.env_config.real_action_lower_bound[0]
        mpc_input_msg.max_acc_incre_rate = self.env_config.action_upper_bound[0] / delta_t
        mpc_input_msg.max_acc_decre_rate = self.env_config.action_lower_bound[0] / delta_t
        mpc_input_msg.max_steer_angle = self.env_config.real_action_upper_bound[1]
        mpc_input_msg.max_steer_angle_rate = self.env_config.action_upper_bound[1] / delta_t

        mpc_input_msg = mpc_input_msg.SerializeToString()
        self.socket.send(mpc_input_msg)
    
        mpc_output = MPCOutput()
        mpc_output_msg = self.socket.recv()
        mpc_output.ParseFromString(mpc_output_msg)

        u = np.array([mpc_output.action.acceleration, mpc_output.action.front_wheel_angle])
        u_increment = u - pre_action
        normalized_u_increment = (u_increment - self.action_center) / self.action_half_range
        return normalized_u_increment


if __name__ == "__main__":
    from gops.create_pkg.create_env import create_env

    scenario_selector = "1"
    scenario = "crossroad"
    env_config = get_idsim_env_config(scenario)
    env_config.update({"scenario_selector": scenario_selector})
    model_config = get_idsim_model_config(scenario)

    controller = OptController4idSim(scenario, Config.from_partial_dict(env_config))

    env = create_env(
        "pyth_idsim", 
        env_config=env_config, 
        env_model_config=model_config, 
        env_scenario=scenario
    )
    obs, info = env.reset()
    horizon = 400
    for i in range(horizon):
        x = env.state
        u = controller(x)
        next_obs, reward, done, info = env.step(u)
        if done:
            break
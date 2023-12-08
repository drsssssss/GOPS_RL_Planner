from typing import Dict

MAP_ROOT_CROSSROAD = 'YOUR_MAP_ROOT_CROSSROAD'
MAP_ROOT_MULTILANE = 'YOUR_MAP_ROOT_MULTILANE'
pre_horizon = 30
delta_t = 0.1

env_config_param_base = {
    "use_render": False,
    "seed": 1,
    "actuator": "ExternalActuator",
    "scenario_reuse": 10,
    "num_scenarios": 20,
    "detect_range": 60,
    "choose_vehicle_retries": 10,
    "scenario_root": MAP_ROOT_CROSSROAD,
    "scenario_selector": '1',
    "extra_sumo_args": ("--start", "--delay", "200"),
    "warmup_time": 5.0,
    "max_steps": 500,
    "ignore_traffic_lights": False,
    "incremental_action": True,
    "action_lower_bound": (-4.0 * delta_t, -0.25 * delta_t),
    "action_upper_bound": (2.5 * delta_t, 0.25 * delta_t),
    "real_action_lower_bound": (-3.0, -0.571),
    "real_action_upper_bound": (0.8, 0.571),
    "obs_num_surrounding_vehicles": {
        "passenger": 5,
        "bicycle": 0,
        "pedestrian": 0,
    },
    "ref_v": 8.0,
    "ref_length": 48.0,
    "obs_num_ref_points": 2 * pre_horizon + 1,
    "obs_ref_interval": 0.8,
    "vehicle_spec": (1880.0, 1536.7, 1.13, 1.52, -128915.5, -85943.6, 20.0, 0.0),
    "singleton_mode": "reuse",
    "use_multiple_path_for_multilane": True,
}

model_config_base = {
    "N": pre_horizon,
    "full_horizon_sur_obs": False,
    "ahead_lane_length_min": 6.0,
    "ahead_lane_length_max": 30.0,
    "v_discount_in_junction_straight": 0.75,
    "v_discount_in_junction_left_turn": 0.5,
    "v_discount_in_junction_right_turn": 0.375,
    "num_ref_lines": 3,
    "dec_before_junction": 0.8,
    "ego_length": 5.0,
    "ego_width": 1.8,
    "safe_dist_incremental": 1.2,

    "num_ref_points": pre_horizon + 1,
    "ego_feat_dim": 7,  # vx, vy, r, last_last_acc, last_last_steer, last_acc, last_steer
    "per_sur_state_dim": 6,  # x, y, phi, speed, length, width
    "per_sur_state_withinfo_dim": 7,  # x, y, phi, speed, length, width, mask
    "per_sur_feat_dim": 5,  # x, y, cos(phi), sin(phi), speed
    "per_ref_feat_dim": 5,  # x, y, cos(phi), sin(phi), speed
    "real_action_upper": (0.8, 0.571),
    "real_action_lower": (-3.0, -0.571),
    "steer_rate_2_min": -0.2,
    "steer_rate_2_max": 0.2,

    "vx_min": 0.0,
    "vx_max": 20.0,
    "vy_min": -4.0,
    "vy_max": 4.0,

    "max_dist_from_ref": 1.8,

    "Q": (
        0.4,
        0.4,
        500.0,
        1.0,
        2.0,
        300.0,
    ),
    "R": (
        1.0,
        20.0,
    ),

    # C_steer_rate_2_min, C_steer_rate_2_max #TODO: check rationality
    "C_steer_rate_2": (100, 100),
    # C_vx_min, C_vx_max, C_vy_min, C_vy_max #TODO: check rationality
    "C_v": (100., 100., 100., 100.),

    "gamma": 1.0,  # should equal to discount factor of algorithm
    "lambda_c": 0.99,  # discount of lat penalty
    "lambda_p": 0.99,  # discount of lon penalty
    "C_lat": 3.0,
    "C_obs": 300.0,
    "C_back": (
        0.1,  # surr is behind ego
        1.0  # surr is in front of ego
    ),
    "C_road": 300.0,
    "ref_v_lane": 8.0,
    "filter_num": 5
}

env_config_param_crossroad = env_config_param_base
model_config_crossroad = model_config_base

env_config_param_multilane = {
    **env_config_param_base,
    "scenario_root": MAP_ROOT_MULTILANE,
    "action_lower_bound": (-4.0 * delta_t, -0.065 * delta_t),
    "action_upper_bound": (2.5 * delta_t, 0.065 * delta_t),
    "real_action_lower_bound": (-3.0, -0.065),
    "real_action_upper_bound": (0.8, 0.065),
}

model_config_multilane = {
    **model_config_base,
    "real_action_lower": (-3.0, -0.065),
    "real_action_upper": (0.8, 0.065),
    "Q": (
        0.2,
        0.2,
        500.0,
        0.5,
        2.0,
        2000.0,
    ),
    "R": (
        1.0,
        500.0,
    )
}


def get_idsim_env_config(scenario="crossroad") -> Dict:
    if scenario == "crossroad":
        return env_config_param_crossroad
    elif scenario == "multilane":
        return env_config_param_multilane
    else:
        raise NotImplementedError


def get_idsim_model_config(scenario="crossroad") -> Dict:
    if scenario == "crossroad":
        return model_config_crossroad
    elif scenario == "multilane":
        return model_config_multilane
    else:
        raise NotImplementedError

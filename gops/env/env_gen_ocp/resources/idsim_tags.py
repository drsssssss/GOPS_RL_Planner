reward_tags = (
    "reward",
    "reward_tracking_lon",
    "reward_tracking_lat",
    "reward_tracking_phi",
    "reward_tracking_vx",
    "reward_tracking_vy",
    "reward_tracking_yaw_rate",
    "reward_action_acc",
    "reward_action_steer",
    "reward_cost_acc_rate_1",
    "reward_cost_steer_rate_1",
    "reward_cost_steer_rate_2_min",
    "reward_cost_steer_rate_2_max",
    "reward_cost_vx_min",
    "reward_cost_vx_max",
    "reward_cost_vy_min",
    "reward_cost_vy_max",
    "reward_penalty_lat_error",
    "reward_penalty_sur_dist",
    "reward_penalty_road",
    "collision_flag",
)

done_tags = (
    "done/arrival",
    "done/red_violation",
    "done/yellow_violation",
    "done/out_of_driving_area",
    "done/collision",
    "done/max_steps")

idsim_tb_tags = (
    "Evaluation/Arrival rate-RL iter",
    "Evaluation/Red violation rate-RL iter",
    "Evaluation/Yellow violation rate-RL iter",
    "Evaluation/Out of driving area rate-RL iter",
    "Evaluation/Collision rate-RL iter",
    "Evaluation/Max steps rate-RL iter",
    "Evaluation/total_reward",
    "Evaluation/reward_tracking_lon",
    "Evaluation/reward_tracking_lat",
    "Evaluation/reward_tracking_phi",
    "Evaluation/reward_tracking_vx",
    "Evaluation/reward_tracking_vy",
    "Evaluation/reward_tracking_yaw_rate",
    "Evaluation/reward_action_acc",
    "Evaluation/reward_action_steer",
    "Evaluation/reward_cost_acc_rate_1",
    "Evaluation/reward_cost_steer_rate_1",
    "Evaluation/reward_cost_steer_rate_2_min",
    "Evaluation/reward_cost_steer_rate_2_max",
    "Evaluation/reward_cost_vx_min",
    "Evaluation/reward_cost_vx_max",
    "Evaluation/reward_cost_vy_min",
    "Evaluation/reward_cost_vy_max",
    "Evaluation/reward_penalty_lat_error",
    "Evaluation/reward_penalty_sur_dist",
    "Evaluation/reward_penalty_road",
    "Evaluation/collision_flag"
)

idsim_tb_keys = (*done_tags, *reward_tags)
idsim_tb_tags_dict = dict(zip(idsim_tb_keys, idsim_tb_tags))
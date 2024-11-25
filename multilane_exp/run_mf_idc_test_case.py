import os
import sys
import datetime
import pathlib
import json
import pickle
import torch
import re
from typing import NamedTuple

from gops.trainer.idsim_idc_mf_multilane_evaluator import (
    IdsimIDCEvaluator,
    get_args_from_json,
)
from gops.trainer.idsim_render.animation_mf_multilane import AnimationLane
from gops.trainer.idsim_render.animation_crossroad import AnimationCross
from multilane_exp.run_idc_eval import find_optimal_network

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

## How to use this script:
# 1. Choose scenario_root, ini_network_root, nn_index
# 2. Modify IDC_MODE. if true, means using IDC to select path, otherwise, only tracking one path
# 3. Modify path_selector.
#       'loss' means using sum of reward to select path,
#       'value' means using value network to select path.
#        only works for 'value'
# 4. Run this script


class TestCase(NamedTuple):
    scenario_root: str
    map_id: str
    seed: int
    warmup_time: float
    ego_id: str


def load_test_case(test_case_path="example.json", filter = None):
    # load test case
    with open(test_case_path, "r", encoding="utf-8") as file:
        test_case_list = json.load(file)
    # filter test case
    if filter is not None:
        for key, value in filter.items():
            if isinstance(value, str):
                test_case_list = [test_case for test_case in test_case_list if re.match(value, test_case.get(key, ""))]
            elif isinstance(value, int):
                test_case_list = [test_case for test_case in test_case_list if test_case.get(key, 0) > value]
    for test in test_case_list:
        print(f"test case: {test['scene']} map: {test['map_id']} warmup time: {test['warmup_time']}") 

    return test_case_list


def change_type(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = change_type(value)
        return obj
    elif isinstance(obj, list):
        return [change_type(value) for value in obj]
    elif isinstance(obj, pathlib.Path):
        return str(obj)
    else:
        return obj
    
    
def get_args():
    render_only = False
    scenario = "multilane"
    
    ini_network_root = (
        r"results/pyth_idsim_planning/DSACTPI_241123-173659"
    )
    test_case_file = "/root/GOPS/multilane_exp/test_case_training_ml.json"
    # test_case_multilane
    test_filter = {"seed": 971823494}
                   # iteration > 500000
    test_filter = None
    nn_index = '0'
    act_seq_len = None
    RHC_mode = (act_seq_len != 1) or act_seq_len is None
    dpi = 60
    frame_skip = 1
    if act_seq_len == None:
        seq_mode = "openloop"
    elif act_seq_len == 1:
        seq_mode = "closedloop"
    else:
        seq_mode = str(act_seq_len) + "seq"
    test_name = f"eval_training_{nn_index}_ml_lanechange_slow_" + seq_mode
    scenario_filter_surrounding_selector = None 
    direction_selector = None
    IDC_mode = True
    fast_mode = False   
    multi_ref = True
    plot_reward = True
    random_ref_probability = 0.01

    path_selector = "value"
    selector_bias = 0.0
    max_steps = 2000

    # takeover bias config 
    takeover_bias = False
    bias_x = (0.0, 0.5)
    bias_y = (0.0, 0.5)
    bias_phi = (0.0, 0.05)
    bias_vx = (0.0, 1.5)
    bias_ax = (0.0, 0.25)
    bias_steer = (0.0, 0.02)
    minimum_clearance_when_takeover = 5.0

    # random ref_v config
    random_ref_v = False
    ref_v_range = (0, 2)
    ref_v = 9

    # random acc config
    use_random_acc = False
    random_acc_prob = (0.3, 0.3, 1)
    random_acc_cooldown = (50, 50, 50)
    random_acc_range = (0.2, 0.8)
    random_dec_range = (-2.0, -1)

    # env model config
    Q_mat = (0.0, 0.0, 0.0, 0.5, 0.0, 0.0)
    R_mat = (0.0, 0.0)
    C_acc_rate_1 = 0.0
    C_steer_rate_1 = 0.0
    C_steer_rate_2 = (0.0, 0.0)

    if nn_index is None:
        nn_index = find_optimal_network(ini_network_root)
    if test_name is None:
        test_name = datetime.datetime.now().strftime("%y%m%d-%H%M%S") + "IDCTest"
    save_folder = os.path.join(ini_network_root, test_name)
    if os.path.exists(save_folder) and not render_only:
        val = input("Warning: save_folder already exists, press esc to exit, press enter to delete and continue")
        if val == chr(27):
            exit()
        else:
            # clear save_folder
            os.system("rm -rf {}".format(save_folder))
    else:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=False)

    # get training args
    config_path = os.path.join(ini_network_root, "config.json")
    args = get_args_from_json(config_path, {})

    # save args to save_folder
    args["render_only"] = render_only
    args["save_folder"] = save_folder
    args["test_case_file"] = test_case_file
    args['env_scenario'] = scenario
    args["test_filter"] = test_filter
    args["plot_reward"] = plot_reward
    args["IDC_MODE"] = IDC_mode
    args["fast_mode"] = fast_mode
    args["PATH_SELECTION_EVIDENCE"] = path_selector
    args["PATH_SELECTION_DIFF_THRESHOLD"] = selector_bias
    args["frame_skip"] = frame_skip
    args["dpi"] = dpi
    args["ini_network_root"] = ini_network_root
    args["ini_network_dir"] = ini_network_root + f"/apprfunc/apprfunc_{nn_index}.pkl"

    args["env_config"]["max_steps"] = max_steps
    args["env_config"]["use_multiple_path_for_multilane"] = multi_ref
    args["env_config"]["random_ref_probability"] = random_ref_probability
    args["env_config"]["scenario_root"] = ''
    args["env_config"]["direction_selector"] = direction_selector
    # args["env_config"]["real_action_upper_bound"] = (0.8, 0.571)
    # args["env_config"]["real_action_lower_bound"] = (-1.5, -0.571)
    args["env_config"]["scenario_filter_surrounding_selector"] = scenario_filter_surrounding_selector
    # args["env_config"]["vehicle_spec"] = (1880.0, 1536.7, 1.22, 1.70, -128915.5, -85943.6, 20.0, 0.0)

    args["env_config"]["takeover_bias"] = takeover_bias
    args["env_config"]["takeover_bias_x"] = bias_x
    args["env_config"]["takeover_bias_y"] = bias_y
    args["env_config"]["takeover_bias_phi"] = bias_phi
    args["env_config"]["takeover_bias_vx"] = bias_vx
    args["env_config"]["takeover_bias_ax"] = bias_ax
    args["env_config"]["takeover_bias_steer"] = bias_steer
    args["env_config"]["minimum_clearance_when_takeover"] = minimum_clearance_when_takeover


    args["env_config"]["random_ref_v"] = random_ref_v
    args["env_config"]["ref_v_range"] = ref_v_range
    args["env_config"]["ref_v"] = ref_v
    args["env_model_config"]["ref_v_lane"] = ref_v

    args["env_config"]["use_random_acc"] = use_random_acc
    args["env_config"]["random_acc_prob"] = random_acc_prob
    args["env_config"]["random_acc_cooldown"] = random_acc_cooldown
    args["env_config"]["random_acc_range"] = random_acc_range
    args["env_config"]["random_dec_range"] = random_dec_range

    args["env_model_config"]["Q"] = Q_mat
    args["env_model_config"]["R"] = R_mat
    args["env_model_config"]["C_acc_rate_1"] = C_acc_rate_1
    args["env_model_config"]["C_steer_rate_1"] = C_steer_rate_1
    args["env_model_config"]["C_steer_rate_2"] = C_steer_rate_2

    args["eval_save"] = True
    args["record_loss"] = True

    args["env_config"]["use_render"] = False
    args["env_config"]["num_scenarios"] = 28
    args["env_config"]["scenario_reuse"] = 1
    args["env_config"]["use_logging"] = True
    args["env_config"]["singleton_mode"] = "invalidate"
    args["env_config"]["logging_name_template"] = "{context.scenario_id:03d}/{context.episode_count:04d}.pkl"
    args["env_config"]["fcd_name_template"] = "{context.scenario_id:03d}/fcd.xml"

    # if act_seq_len is none, use the original value
    if act_seq_len is not None:
        if act_seq_len > args["act_seq_len"]:
            print("Warning: act_seq_len is greater than the original value. Using the original maximum value.")
            act_seq_len = args["act_seq_len"]
        args["act_seq_len"] = act_seq_len
    args["RHC_mode"] = RHC_mode

    with open(os.path.join(save_folder, "test_config.json"), "w") as f:
        json.dump(args, f, indent=4, default= change_type)
    os.chmod(os.path.join(save_folder, "test_config.json"), 0o444) # read only
    return args


if __name__ == "__main__":
    theme_style = "light"  # only works for AnimationCross
    args = get_args()
    frame_skip = args["frame_skip"]
    dpi = args["dpi"]
    test_case_list = load_test_case(args["test_case_file"], args["test_filter"])
    render_only = args["render_only"]

    if not render_only:
        IDCevaluator = IdsimIDCEvaluator(**args, print_done=True)
        for idx, test_case in enumerate(test_case_list):
            idsim_tb_eval_dict = IDCevaluator.run_testcase(idx, test_case, use_mpc=False)
            print(idsim_tb_eval_dict)
        IDCevaluator.env.close()

    save_folder = args["save_folder"]
    log_path_root = pathlib.Path(save_folder)
    print("use the latest evaluation result")
    # read config.json from log_path_root
    config_path = log_path_root / "test_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
            
    for idx, test_case in enumerate(test_case_list):
        map_id = int(test_case["map_id"])
        test_scene = test_case.get("scene", "scene"+str(idx))
        surfix = "%03d" % map_id
        log_path = log_path_root / "test_{}".format(idx) / surfix

        episode_list = [
            path
            for path in os.listdir(log_path)
            if (path.startswith("episode") and path.endswith(".pkl"))
        ]
        if len(episode_list) != 1:
            raise ValueError("episode_list should only have one episode")
        episode = episode_list[0]
        print("episode: ", episode)

        fcd_file_path = log_path / "fcd.xml"
        print("fcd_file_path: ", fcd_file_path)

        if args["env_scenario"] == "crossroad":
            animation = AnimationCross(theme_style, fcd_file_path, config)
        else:
            animation = AnimationLane(theme_style, fcd_file_path, config)

        episode_path = log_path / episode
        with open(episode_path, "rb") as f:
            episode_data = pickle.load(f)
        animation.generate_animation(episode_data, log_path_root, idx, test_scene= test_scene, mode="debug",dpi=dpi, frame_skip=frame_skip, plot_reward=args["plot_reward"])

import os
import sys
import datetime
import pathlib
import json
import pickle
import torch
from typing import NamedTuple

from gops.trainer.idsim_idc_mf_multilane_evaluator import IdsimIDCEvaluator, get_args_from_json
from gops.trainer.idsim_render.animation_mf_multilane import AnimationLane
from run_idc_eval import find_optimal_network

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
print(sys.path)

## How to use this script:
# 1. Choose scenario_root, ini_network_root, nn_index
# 2. Modify IDC_MODE. if true, means using IDC to select path, otherwise, only tracking one path
# 3. Modify PATH_SELECTION_EVIDENCE.
#       'loss' means using sum of reward to select path,
#       'value' means using value network to select path.
# 4. Run this script

class TestCase(NamedTuple):
    scenario_root: str
    map_id: str
    seed: int
    warmup_time: float
    ego_id: str

def load_test_case(test_case_path='example.json'):
    with open(test_case_path, 'r', encoding='utf-8') as file:
        test_case_list = json.load(file)
    return test_case_list

def get_args():
    scenario_root = pathlib.Path('/root/idsim-scenarios/idsim-multilane-dense/')
    ini_network_root = r"/root/gops/results/idsim_multilane_vec/dsact_pi/12345_before_plus_reward8_random_takeover_acc_scene_refv_15_short_ref_no_oppo_trunc_run0"
    nn_index = None
    eval = True

    if nn_index is None:
        nn_index = find_optimal_network(ini_network_root)
    config_path = os.path.join(ini_network_root, "config.json")
    args = get_args_from_json(config_path, {})
    args["ini_network_root"] = ini_network_root
    args["ini_network_dir"] = ini_network_root+f"/apprfunc/apprfunc_{nn_index}.pkl"
    save_folder = os.path.join(
            ini_network_root,
            datetime.datetime.now().strftime("%y%m%d-%H%M%S") + "IDCevaluation",
        )
    os.makedirs(save_folder, exist_ok=True)
    # save args to save_folder
    with open(os.path.join(save_folder, "config.json"), "w") as f:
        json.dump(args, f, indent=4)
    args['save_folder'] = save_folder
    args['eval_save'] = True
    args['IDC_MODE'] = False
    args['record_loss'] = True
    args["PATH_SELECTION_EVIDENCE"] = 'loss' # 'loss' or 'value'
    args['PATH_SELECTION_DIFF_THRESHOLD'] = 0.0 # preference for current lane
    args['env_config']['max_steps'] = 1000
    args['env_config']['use_render'] = False
    args['env_config']['use_multiple_path_for_multilane'] = True
    args['env_config']['random_ref_probability'] = 0.0
    args['env_config']['num_scenarios'] = num_scenarios = 10
    args['env_config']['scenario_reuse'] = scenario_reuse = 5
    args['env_config']['scenario_root'] = scenario_root
    args['env_config']['use_logging'] = True
    args['env_config']['singleton_mode'] = 'invalidate' 
    args['env_config']['logging_name_template'] = "{context.scenario_id:03d}/{context.episode_count:04d}.pkl"
    args['env_config']['fcd_name_template'] = "{context.scenario_id:03d}/fcd.xml"
    args['env_config']['takeover_bias'] = False
    args['env_config']['use_random_acc'] = False

    args['env_config']['random_acc_prob'] = (0., 1) # probability to accelerate and decelerate, respectively
    args['env_config']['random_acc_cooldown'] = (30, 500, 500) # cooldown for acceleration, deceleration and ref_v, respectively  0  50 100
    args['env_config']['random_acc_range'] = (0., 1)# (m/s^2), used for acceleration
    args['env_config']['random_dec_range'] = (-3.0, -2.5) # (m/s^2), used for deceleration


    args["env_model_config"]["Q"] =    (0.0, 0.0, 0.0, 0.5, 0.0, 0.0)
    args["env_model_config"]["R"] =    (0.0, 0.0)
    args["env_model_config"]["C_acc_rate_1"] =  0.
    args["env_model_config"]["C_steer_rate_1"] = 0.
    args["env_model_config"]["C_steer_rate_2"] =  (0.,0.)
    return args

if __name__ == "__main__":
    dpi = 60 
    frame_skip = 4
    theme_style = 'light' # only works for AnimationCross
    test_case_list = load_test_case('/root/gops/exp/test_case.json')
    args = get_args()
    IDCevaluator = IdsimIDCEvaluator(**args, print_done=True)
    for idx, test_case in enumerate(test_case_list):
        idsim_tb_eval_dict = IDCevaluator.run_testcase(idx, test_case, use_mpc=False)
        print(idsim_tb_eval_dict)
    IDCevaluator.env.close()

    save_folder = args['save_folder']
    log_path_root = pathlib.Path(save_folder)
    print('use the latest evaluation result')
    # read config.json from log_path_root
    config_path = log_path_root/'config.json'
for idx, test_case in enumerate(test_case_list):

    with open(config_path, 'r') as f:
        config = json.load(f)
        map_id = int(test_case['map_id'])
        surfix = '%03d' % map_id
        log_path = log_path_root/'test_{}'.format(idx)/surfix
        save_path = log_path

        episode_list = [log_path for log_path in os.listdir(log_path) if (log_path.startswith('episode') and log_path.endswith('.pkl'))]
        episode_list.sort()

        print('episode_list: ', episode_list)

        fcd_file_path = log_path/'fcd.xml'
        print('fcd_file_path: ', fcd_file_path)

        animation = AnimationLane(theme_style, fcd_file_path, config)


        for i, episode in enumerate(episode_list):
            episode_path = log_path / episode
            with open(episode_path, 'rb') as f:
                episode_data = pickle.load(f)
                episode_data.save_folder = log_path_root
            animation.generate_animation(episode_data, save_path, i, mode='debug', dpi=dpi, frame_skip=frame_skip)

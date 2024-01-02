import os
import datetime
import pathlib
import json
import torch
from typing import NamedTuple

from gops.trainer.idsim_idc_evaluator import IdsimIDCEvaluator, get_args_from_json

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

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
    import json
    with open(test_case_path, 'r', encoding='utf-8') as file:
        test_case_list = json.load(file)
    return test_case_list

def get_args():
    scenario_root = pathlib.Path('idsim-base-multilane')
    ini_network_root = r"gops/results/pyth_idsim/FHADP2_231212-091300-fhadp2-v12"
    nn_index = 20000

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
    args['IDC_MODE'] = True
    args['record_loss'] = True
    args["PATH_SELECTION_EVIDENCE"] = 'loss' # 'loss' or 'value'
    args['PATH_SELECTION_DIFF_THRESHOLD'] = 0.0 # preference for current lane
    args['env_config']['max_steps'] = 1000
    args['env_config']['use_render'] = True
    args['env_config']['use_multiple_path_for_multilane'] = True
    args['env_config']['random_ref_probability'] = 0.0
    args['env_config']['num_scenarios'] = num_scenarios = 4
    args['env_config']['scenario_reuse'] = scenario_reuse = 5
    args['env_config']['scenario_root'] = scenario_root
    args['env_config']['use_logging'] = True
    args['env_config']['singleton_mode'] = 'invalidate' 
    args['env_config']['logging_name_template'] = "{context.scenario_id:03d}/{context.episode_count:04d}.pkl"
    args['env_config']['fcd_name_template'] = "{context.scenario_id:03d}/fcd.xml"
    return args

if __name__ == "__main__":
    test_case_list = load_test_case('example.json')
    args = get_args()
    IDCevaluator = IdsimIDCEvaluator(**args, print_done=True)
    for idx, test_case in enumerate(test_case_list):
        idsim_tb_eval_dict = IDCevaluator.run_testcase(idx, test_case, use_mpc=True)
        print(idsim_tb_eval_dict)
    IDCevaluator.env.close()

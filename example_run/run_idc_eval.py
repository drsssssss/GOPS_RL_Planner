import os
import datetime
import pathlib
from gops.trainer.idsim_idc_evaluator import IdsimIDCEvaluator, get_args_from_json
from idsim.utils.fs import TEMP_ROOT

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)

if __name__ == "__main__":
    ini_network_root = R""
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
    args['save_folder'] = save_folder
    args['eval_save'] = True
    args['IDC_MODE'] = False
    args['record_loss'] = True
    args["PATH_SELECTION_EVIDENCE"] = 'loss' # 'loss' or 'value'
    args['env_config']['max_steps'] = 2000
    args['env_config']['num_scenarios'] = num_scenarios = 4
    args['env_config']['scenario_reuse'] = scenario_reuse = 5
    scenario_root = pathlib.Path('/home/toyota/code/gops_for_idsim/idsim-multilane-v10')
    args['env_config']['scenario_root'] = scenario_root
    args['env_config']['use_logging'] = True
    args['env_config']['singleton_mode'] = 'invalidate' 
    args['env_config']['logging_name_template'] = "{context.scenario_id:03d}/{context.episode_count:04d}.pkl"
    args['env_config']['fcd_name_template'] = "{context.scenario_id:03d}/fcd.xml"
    IDCevaluator = IdsimIDCEvaluator(**args, print_done=True)
    IDCevaluator.run_n_episodes(num_scenarios * scenario_reuse, iteration=0)

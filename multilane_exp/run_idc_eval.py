import os
import datetime
import pathlib
import json
import time
from gops.trainer.idsim_idc_mf_multilane_evaluator import IdsimIDCEvaluator, get_args_from_json
from gops.trainer.idsim_render.animation_crossroad import AnimationCross
from gops.trainer.idsim_render.animation_mf_multilane import AnimationLane
import pickle

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(2)

## How to use this script:
# 1. Choose scenario_root, ini_network_root, nn_index
# 2. Modify IDC_MODE. if true, means using IDC to select path, otherwise, only tracking one path
# 3. Modify PATH_SELECTION_EVIDENCE.
#       'loss' means using sum of reward to select path,
#       'value' means using value network to select path.
# 4. Run this script

def find_optimal_network(ini_network_root):
    # find the netowrk end with _opt.pkl if not found, return the lastest network
    appr_dir = os.path.join(ini_network_root, 'apprfunc')
    network_list = []
    for file in os.listdir(appr_dir):
        if file.endswith('_opt.pkl'):
            network_list.append(file)
    if len(network_list) > 0:
        network_list.sort()
        index , surfix = network_list[-1].split('_')[1:3]
        surfix = surfix.split('.')[0]
        index = index + '_' + surfix
    if len(network_list) == 0:
        network_list = [file for file in os.listdir(appr_dir) if file.endswith('.pkl')]
        network_list.sort()
        index = network_list[-1].split('_')[1]
        index = index.split('.')[0]
    return index

if __name__ == "__main__":
    scenario_root = pathlib.Path('/root/idsim-scenarios/idsim-base-multilane/')
    ini_network_root = r"/root/gops/results/idsim_multilane_vec/dsact_pi/12345_before_plus_reward8_sur8punish_reset_lane_refv_15_short_ref_no_oppo_trunc_old_map_new_eval_setting_run0"
    nn_index = '800000'
    dpi = 60
    frame_skip = 2
    scene_id_list = ['00{}'.format(i) for i in range(0, 5)] #+ ['0{}'.format(i) for i in range(10, 12)]
    # scene_id_list = ['000']
    num_scenarios = len(scene_id_list)
    scenario_reuse = 5
    theme_style = 'light' # only works for AnimationCross
    Animation = AnimationLane
    log_path_root = None
    eval = True


    
    if not eval:
        if log_path_root is None:
            print('no evluation, plot the latest evaluation result')
            # get the floder that latest modified and ends with IDCevaluation in ini_network_root
            IDCevaluation_list = [file for file in os.listdir(ini_network_root) if file.endswith('IDCevaluation')]
            IDCevaluation_list.sort(key=lambda fn: os.path.getmtime(ini_network_root+'/'+fn))
            save_folder = os.path.join(ini_network_root, IDCevaluation_list[-1])
    else: 
        if nn_index is None:
            nn_index = find_optimal_network(ini_network_root)
            print('use the latest network: ', nn_index)
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
        args['env_config']['max_steps'] = 2000
        args['env_config']['seed'] = 1
        args['env_config']['use_multiple_path_for_multilane'] = False
        args['env_config']['random_ref_probability'] = 0.0
        args['env_config']['num_scenarios'] = num_scenarios
        args['env_config']['scenario_reuse'] = scenario_reuse
        args['env_config']['scenario_root'] = scenario_root
        args['env_config']['use_logging'] = True
        args['env_config']['singleton_mode'] = 'invalidate' 
        args['env_config']['logging_name_template'] = "{context.scenario_id:03d}/{context.episode_count:04d}.pkl"
        args['env_config']['fcd_name_template'] = "{context.scenario_id:03d}/fcd.xml"
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
        IDCevaluator = IdsimIDCEvaluator(**args, print_done=True)
        IDCevaluator.run_n_episodes(num_scenarios * scenario_reuse, iteration=0)

        time.sleep(3)



    if log_path_root is None:
        log_path_root = pathlib.Path(save_folder)
        print('use the latest evaluation result')
    # read config.json from log_path_root
    config_path = log_path_root/'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    for scene_id in scene_id_list:
        log_path = log_path_root/scene_id
        save_path = log_path

        episode_list = [log_path for log_path in os.listdir(log_path) if (log_path.startswith('episode') and log_path.endswith('.pkl'))]
        episode_list.sort()

        print('episode_list: ', episode_list)

        fcd_file_path = log_path/'fcd.xml'
        print('fcd_file_path: ', fcd_file_path)

        animation = Animation(theme_style, fcd_file_path, config)


        for i, episode in enumerate(episode_list):
            episode_path = log_path / episode
            with open(episode_path, 'rb') as f:
                episode_data = pickle.load(f)
                episode_data.save_folder = log_path_root
            animation.generate_animation(episode_data, save_path, i, mode='debug', dpi=dpi, frame_skip=frame_skip)





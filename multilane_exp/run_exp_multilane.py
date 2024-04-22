import argparse
import os
import copy
import json
import logging
from exp_runner import BaseExpRunner



base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Experiment parameters
script_path = os.path.join(base_path, 'example_train')
save_folder = os.path.join(base_path, 'results/idsim')

exp_name = 'idsim_multilane_exp_0414_12' 
exp_discription = 'new alg(form 3) old sampling config fix small bug in env, choose in junction veh prob 1->0.5, not filter ego when evluation, nominal steer, increase punish on steer and delta steer, decrease acc bound steer bound break dist 6->4 increas heading and steering punish when turning, low speed steer punish: 10->5, fix clip bug in dsact, low ref_v coff while turning 0.375->0.25, add activate collision conditio, punish overshoot of phi and tracking error when turning, dynamic living reward, add take over bias in cross scene, low nomimal steer punish, low speed mode condition ref_v: 1->0.1, new-map more small car, new vec param, dacc bound -2.5->-2, safe margin front 1->1.5'


script_folder = "dsac"
algs = ['dsact']
apprfuncs = ['pi']
envs = ['idsim_multilane_vec']
repeats_num = 1
surfix_filter = 'offserial.py'
run_config = {
    # 'env_id': ['gym_carracingraw'],
    'seed':[12345],
    'buffer_max_size':[400000],
    # 'eval_interval':[2000],
    # 'sample_batch_size':[10],
    # 'sample_interval':[1],
    # 'reward_scale':[1],
    # 'vector_env_num':[10],
    # 'max_iteration':[200000],
}

project_root = None
save_meta_data = True
save_zip = True
max_subprocess = 1
max_waiting_time = 48 * 3600  # seconds
log_level = 'DEBUG'


paser = argparse.ArgumentParser()
paser.add_argument('--max_subprocess', type=int, default=max_subprocess)
paser.add_argument('--max_waiting_time', type=int, default=max_waiting_time)
paser.add_argument('--script_folder', type=str, default=script_folder)
paser.add_argument('--algs', type=list, default=algs)
paser.add_argument('--apprfuncs', type=list, default=apprfuncs)
paser.add_argument('--envs', type=list, default=envs)
paser.add_argument('--repeats_num', type=int, default=repeats_num)
paser.add_argument('--save_folder', type=str, default=save_folder)
paser.add_argument('--script_path', type=str, default=script_path)
paser.add_argument('--run_config', type=dict, default=run_config)
paser.add_argument('--exp_name', type=str, default=exp_name)
paser.add_argument('--exp_discription', type=str, default=exp_discription)
paser.add_argument('--save_meta_data', type=bool, default=save_meta_data)
paser.add_argument('--save_zip', type=bool, default=save_zip)
paser.add_argument('--project_root', type=str, default=project_root)
paser.add_argument('--surfix_filter', type=str, default=surfix_filter)
paser.add_argument('--log_level', type=str, default=log_level)
args = paser.parse_args()



if __name__ == '__main__':
    args = vars(args)
    exp_runner = BaseExpRunner(**args)
    with open(os.path.join(exp_runner.save_folder, 'exp_config.json'), 'w') as f:
        json.dump(args, f, indent=4)
    exp_runner.run()
    
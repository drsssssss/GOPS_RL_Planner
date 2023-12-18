import argparse
import datetime
from typing import Dict, List, Tuple

import torch
import numpy as np
import os

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib as mpl

from gops.trainer.evaluator import Evaluator
from gops.env.env_gen_ocp.resources.idsim_tags import idsim_tb_tags_dict
from gops.utils.common_utils import get_args_from_json
from gops.utils.gops_path import gops_path

class EvalResult:
    def __init__(self):
        # o, a, r
        self.obs_list: List[np.ndarray] = []
        self.action_list: List[np.ndarray] = []
        self.reward_list: List[np.ndarray] = []
        self.info_list: List[Dict] = []
        self.step_list: List[int] = []


class IdsimTestEvaluator(Evaluator):
    def __init__(self, 
        log_policy_dir: str,
        trained_policy_iteration: str,
        num_eval_episode: int,
        is_render: bool
    ):
        log_policy_dir = os.path.join(gops_path, log_policy_dir)
        self.trained_policy_iteration = trained_policy_iteration

        # load args
        args = self.__load_args(log_policy_dir)
        args.update({
            "num_eval_episode": num_eval_episode,
            "is_render": is_render
        })
        # args['env_config']['use_multiple_path_for_multilane'] = False

        args["env_config"]["reference_selector"] = 1
        args["env_config"]["use_render"] = is_render
        args['env_config']["scenario_root"] = r"D:\Develop\\map\idsim-multilane-v10"
        super().__init__(index=0, **args)

        # load network
        log_path = log_policy_dir + f"/apprfunc/apprfunc_{trained_policy_iteration}.pkl"
        self.load_state_dict(torch.load(log_path))

        # save path
        path = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
        path = os.path.abspath(path)
        self.save_path = os.path.join(
            log_policy_dir,
            "simulation-" + datetime.datetime.now().strftime("%y%m%d-%H%M%S")+ "-iter-" + trained_policy_iteration,
        )
        os.makedirs(self.save_path, exist_ok=True)

        # reward coefficient
        self.config = args['env_model_config']
        self.coefficient = {
            "reward": 1.0,

            "reward_tracking_lon":self.config['Q'][0],
            "reward_tracking_lat":self.config['Q'][1],
            "reward_tracking_phi":self.config['Q'][2],
            "reward_tracking_vx":self.config['Q'][3],
            "reward_tracking_vy":self.config['Q'][4],
            "reward_tracking_yaw_rate":self.config['Q'][5],

            "reward_action_acc":self.config['R'][0],
            "reward_action_steer":self.config['R'][1],

            "reward_cost_acc_rate_1":self.config.get('C_acc_rate_1', 0.0),
            "reward_cost_steer_rate_1":self.config.get('C_steer_rate_1', 0.0),
            "reward_cost_steer_rate_2_min":self.config['C_steer_rate_2'][0],
            "reward_cost_steer_rate_2_max":self.config['C_steer_rate_2'][1],

            "reward_cost_vx_min":self.config['C_v'][0],
            "reward_cost_vx_max":self.config['C_v'][1],
            "reward_cost_vy_min":self.config['C_v'][2],
            "reward_cost_vy_max":self.config['C_v'][3],

            "reward_penalty_lat_error":self.config['C_lat'],
            "reward_penalty_sur_dist":self.config['C_obs'],
            "reward_penalty_road":self.config['C_road'],
        }

    @staticmethod
    def __load_args(log_policy_dir: str):
        json_path = os.path.join(log_policy_dir, "config.json")
        parser = argparse.ArgumentParser()
        args_dict = vars(parser.parse_args())
        args = get_args_from_json(json_path, args_dict)
        return args
    
    def run_an_episode(self, iteration, render=True):
        if self.print_iteration != iteration:
            self.print_iteration = iteration
            self.print_time = 0
        else:
            self.print_time += 1
        eval_result = EvalResult()
        idsim_tb_eval_dict = {key: 0. for key in idsim_tb_tags_dict.keys()}
        obs, info = self.env.reset()
        done = 0
        info["TimeLimit.truncated"] = False
        step = 0

        while not (done or info["TimeLimit.truncated"]):
            eval_result.obs_list.append(obs)

            batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
            logits = self.networks.policy(batch_obs)
            action_distribution = self.networks.create_action_distributions(logits)
            action = action_distribution.mode()
            action = action.detach().numpy()[0]

            next_obs, reward, done, next_info = self.env.step(action)

            eval_result.action_list.append(action)
            eval_result.step_list.append(step)
            eval_result.reward_list.append(reward)
            eval_result.info_list.append(next_info)

            obs = next_obs
            info = next_info
            step = step + 1
            if "TimeLimit.truncated" not in info.keys():
                info["TimeLimit.truncated"] = False
            for eval_key in idsim_tb_eval_dict.keys():
                if eval_key in info.keys():
                    idsim_tb_eval_dict[eval_key] += info[eval_key]
                if eval_key in info["reward_details"].keys():
                    idsim_tb_eval_dict[eval_key] += info["reward_details"][eval_key]
            # Draw environment animation
            if render:
                self.env.render()

        eval_dict = {
            'vx_list': [x[0] for x in eval_result.obs_list],
            'vy_list': [x[1] for x in eval_result.obs_list],
            'r_list': [x[2] for x in eval_result.obs_list],
            'acc_list': [x[5] for x in eval_result.obs_list],
            'steer_list': [x[6] * 180 / np.pi for x in eval_result.obs_list],
            "y_ref_list": [x[7 + 31] for x in eval_result.obs_list],
            "phi_ref_list": [np.arccos(x[7 + 31 + 31]) * 180 / np.pi for x in eval_result.obs_list],
            'step_list': eval_result.step_list,
        }

        reward_dict = {
            k: [info["reward_details"][k] for info in eval_result.info_list] 
            for k in idsim_tb_eval_dict.keys() 
            if ('reward' in k) and (k in info['reward_details'].keys())
        }

        for k in reward_dict.keys():
            if k in info['reward_details'].keys():
                reward_dict[k] = [info['reward_details'][k] for info in eval_result.info_list]

        episode_return = sum(eval_result.reward_list)
        idsim_tb_eval_dict["total_avg_return"] = episode_return

        self.plot_evaluation(iteration, eval_dict, reward_dict, idsim_tb_eval_dict)

        return eval_dict, reward_dict, idsim_tb_eval_dict

    def run_n_episodes(self, n, iteration) -> List[Tuple[Dict]]:
        data_list = []
        for _ in range(n):
            data_list.append(self.run_an_episode(iteration, self.render))
            print(f'episode {_} done')
        return data_list
        
    def run(self):
        data_list = self.run_n_episodes(n = self.num_eval_episode, iteration=0)

        # plot data
        for episode_index, data in enumerate(data_list):
            self.plot_evaluation(episode_index, *data)

        # print idsim_tb_eval_dict
        for key in data_list[0][2].keys():
            avg_return = np.mean([data[2][key] for data in data_list])
            print(f'{key}: {avg_return}')


    def plot_evaluation(self, episode_index, eval_dict, reward_dict, idsim_tb_eval_dict):
        print(f'Plotting...{episode_index}')
        steps = len(eval_dict['step_list']) // 100 + 1
        # steps = 1

        fig, axes = plt.subplots(2, 2, figsize=(16, 8))

        # ego vx, vy, yaw rate
        ax1:Axes = axes[0][0]
        ax1.plot(eval_dict['step_list'][::steps], eval_dict['vx_list'][::steps], '.-', label='vx', color='b')
        ax1.set_ylabel('$v_x$', color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        ax2.plot(eval_dict['step_list'][::steps], eval_dict['vy_list'][::steps], '.-', label='vy', color='r')
        ax2.set_ylabel('$v_y$', color='r')
        ax2.tick_params('y', colors='r')
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))  # Move the last y-axis spine over to the right by 60 points
        ax3.plot(eval_dict['step_list'][::steps], eval_dict['r_list'][::steps], '.-', label='yaw rate', color='g')
        ax3.set_ylabel('yaw rate', color='g')
        ax3.tick_params('y', colors='g')
        ax1.set_title('Vehicle Speeds and Yaw Rate')
        ax1.set_xlabel('Time')

        # reference delta y and delta phi
        ax1:Axes = axes[0][1]
        ax1.plot(eval_dict['step_list'][::steps], eval_dict['y_ref_list'][::steps], '.-', label='lateral error')
        ax1.set_ylabel('$y-y_{ref}$', color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        ax2.plot(eval_dict['step_list'][::steps], eval_dict['phi_ref_list'][::steps], '.-', label='relative orientation', color='r')
        ax2.set_ylabel('$\phi-\phi_{ref}$(degree)', color='r')
        ax2.tick_params('y', colors='r')
        ax2.set_title('Errors with Reference Trajectory')
        ax1.set_xlabel('Time')

        # all the non-zero reward
        ax1:Axes = axes[1][0]
        for k, v in reward_dict.items():
            if np.abs(np.mean(v)) > 1e-6:
                ax1.plot(eval_dict['step_list'][::steps], v[::steps], '.-', label=k)
        ax1.set_ylabel('Reward')
        ax1.set_title('Rewards')
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.set_xlabel('Time')

        # real acc, real steer angle
        ax1:Axes = axes[1][1]
        ax1.plot(eval_dict['step_list'][::steps], eval_dict['acc_list'][::steps], '.-', label='acceleration', color='b')
        ax1.set_ylabel('acceleration', color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        ax2.plot(eval_dict['step_list'][::steps], eval_dict['steer_list'][::steps], '.-', label='steering angle', color='r')
        ax2.set_ylabel('steering angle (degree)', color='r')
        ax2.tick_params('y', colors='r')
        ax1.set_xlabel('Time')
        ax1.set_title('Actual Acceleration and Steering Angle')

        # 显示图表
        plt.tight_layout()
        fig.savefig(os.path.join(self.save_path, f'ep_{episode_index}.png'), bbox_inches='tight')
        plt.close(fig)
        plt.show()
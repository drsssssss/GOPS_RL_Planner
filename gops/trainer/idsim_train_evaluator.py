#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Evaluator for IDSim when training
#  Update Date: 2023-11-22, Guojian Zhan: create this file
from typing import Dict, List

import numpy as np
import torch
from gops.trainer.evaluator import Evaluator
from gops.env.env_gen_ocp.resources.idsim_tags import idsim_tb_tags_dict


class EvalResult:
    def __init__(self):
        # o, a, r
        self.obs_list: List[np.ndarray] = []
        self.action_list: List[np.ndarray] = []
        self.reward_list: List[np.ndarray] = []

class IdsimTrainEvaluator(Evaluator):
    def __init__(self, index=0, **kwargs):
        super().__init__(index, **kwargs)

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
        while not (done or info["TimeLimit.truncated"]):
            batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
            logits = self.networks.policy(batch_obs)
            action_distribution = self.networks.create_action_distributions(logits)
            action = action_distribution.mode()
            action = action.detach().numpy()[0]
            next_obs, reward, done, next_info = self.env.step(action)
            eval_result.obs_list.append(obs)
            eval_result.action_list.append(action)
            obs = next_obs
            info = next_info
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
            eval_result.reward_list.append(reward)
        episode_return = sum(eval_result.reward_list)
        idsim_tb_eval_dict["total_avg_return"] = episode_return
        return idsim_tb_eval_dict

    def run_n_episodes(self, n, iteration):
        eval_list = [self.run_an_episode(iteration, self.render) for _ in range(n)]
        avg_idsim_tb_eval_dict = {
            k: np.mean([d[k] for d in eval_list]) for k in eval_list[0].keys()
            }
        return avg_idsim_tb_eval_dict
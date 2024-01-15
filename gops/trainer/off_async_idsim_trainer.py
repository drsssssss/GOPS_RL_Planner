#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Asynchronous Parallel Trainer for off-policy RL algorithms and idsim environment
#  Update Date: 2024-01-15, Yao Lyu: create trainer

__all__ = ["OffAsyncIdsimTrainer"]

from cmath import inf
import importlib
import os
import random
import time
import warnings

import ray
import torch

from gops.utils.tensorboard_setup import add_scalars, tb_tags
from gops.trainer.off_async_trainer import OffAsyncTrainer
from gops.trainer.idsim_train_evaluator import idsim_tb_tags_dict

warnings.filterwarnings("ignore")


class OffAsyncIdsimTrainer(OffAsyncTrainer):
    def step(self):
        # sampling
        if self.iteration % self.sample_interval == 0:
            if self.sample_tasks.completed_num > 0:
                weights = ray.put(self.networks.state_dict())
                for sampler, objID in self.sample_tasks.completed():
                    batch_data, sampler_tb_dict = ray.get(objID)
                    random.choice(self.buffers).add_batch.remote(batch_data)
                    sampler.load_state_dict.remote(weights)
                    self.sample_tasks.add(sampler, sampler.sample.remote())
                    self.sampler_tb_dict.add_average(sampler_tb_dict)

        # learning
        for alg, objID in self.learn_tasks.completed():
            if self.per_flag:
                extra_info, update_info = ray.get(objID)
                alg_tb_dict, idx, new_priority = extra_info
                self.buffers[0].update_batch.remote(idx, new_priority)
            else:
                alg_tb_dict, update_info = ray.get(objID)

            # replay
            data = ray.get(
                random.choice(self.buffers).sample_batch.remote(self.replay_batch_size)
            )
            if self.use_gpu:
                for k, v in data.items():
                    data[k] = v.cuda()

            weights = ray.put(self.networks.state_dict())
            alg.load_state_dict.remote(weights)
            self.learn_tasks.add(
                alg, alg.get_remote_update_info.remote(data, self.iteration)
            )
            if self.use_gpu:
                for k, v in update_info.items():
                    if isinstance(v, list):
                        for i in range(len(v)):
                            update_info[k][i] = v[i].cpu()
            self.networks.remote_update(update_info)

            self.iteration += 1

            # log
            if self.iteration % self.log_save_interval == 0:
                print("Iter = ", self.iteration)
                add_scalars(alg_tb_dict, self.writer, step=self.iteration)
                add_scalars(self.sampler_tb_dict.pop(), self.writer, step=self.iteration)

            # save networks
            if self.iteration % self.apprfunc_save_interval == 0:
                self.save_apprfunc()

        # evaluate
        if self.iteration - self.last_eval_iteration >= self.eval_interval:
            if self.evluate_tasks.count == 0:
                # There is no evaluation task, add one.
                self._add_eval_task()
            elif self.evluate_tasks.completed_num == 1:
                # Evaluation tasks is completed, log data and add another one.
                objID = next(self.evluate_tasks.completed())[1]
                avg_tb_eval_dict = ray.get(objID)
                total_avg_return = avg_tb_eval_dict['total_avg_return']
                self._add_eval_task()

                if (
                    total_avg_return >= self.best_tar
                    and self.iteration >= self.max_iteration / 5
                ):
                    self.best_tar = total_avg_return
                    print("Best return = {}!".format(str(self.best_tar)))

                    for filename in os.listdir(self.save_folder + "/apprfunc/"):
                        if filename.endswith("_opt.pkl"):
                            os.remove(self.save_folder + "/apprfunc/" + filename)

                    torch.save(
                        self.networks.state_dict(),
                        self.save_folder
                        + "/apprfunc/apprfunc_{}_opt.pkl".format(self.iteration),
                    )

                self.writer.add_scalar(
                    tb_tags["Buffer RAM of RL iteration"],
                    sum(
                        ray.get(
                            [buffer.__get_RAM__.remote() for buffer in self.buffers]
                        )
                    ),
                    self.iteration,
                )
                self.writer.add_scalar(
                    tb_tags["TAR of RL iteration"], total_avg_return, self.iteration
                )
                self.writer.add_scalar(
                    tb_tags["TAR of replay samples"],
                    total_avg_return,
                    self.iteration * self.replay_batch_size,
                )
                self.writer.add_scalar(
                    tb_tags["TAR of total time"],
                    total_avg_return,
                    int(time.time() - self.start_time),
                )
                self.writer.add_scalar(
                    tb_tags["TAR of collected samples"],
                    total_avg_return,
                    sum(
                        ray.get(
                            [
                                sampler.get_total_sample_number.remote()
                                for sampler in self.samplers
                            ]
                        )
                    ),
                )
                for key, value in avg_tb_eval_dict.items():
                    if key != "total_avg_return":
                        self.writer.add_scalar(idsim_tb_tags_dict[key], value, self.iteration)
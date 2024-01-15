#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: example for fhadp2 + idsim + attention + off_serial
#  Update Date: 2024-01-15, Yao Lyu: create example


import argparse
import os
import numpy as np
import json

from copy import deepcopy

from gops.create_pkg.create_alg import create_alg
from gops.create_pkg.create_buffer import create_buffer
from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_evaluator import create_evaluator
from gops.create_pkg.create_sampler import create_sampler
from gops.create_pkg.create_trainer import create_trainer
from gops.utils.init_args import init_args
from gops.utils.plot_evaluation import plot_all
from gops.utils.tensorboard_setup import start_tensorboard, save_tb_to_csv
from gops.env.env_gen_ocp.resources.idsim_config import get_idsim_env_config, get_idsim_model_config, pre_horizon


if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    ################################################
    # Key Parameters for users
    parser.add_argument("--env_id", type=str, default="pyth_idsim", help="id of environment")
    parser.add_argument("--env_scenario", type=str, default="crossroad", help="crossroad / multilane")
    env_scenario = parser.parse_known_args()[0].env_scenario
    parser.add_argument("--env_config", type=dict, default=get_idsim_env_config(env_scenario))
    parser.add_argument("--env_model_config", type=dict, default=get_idsim_model_config(env_scenario))

    parser.add_argument("--algorithm", type=str, default="FHADP2", help="RL algorithm")
    parser.add_argument("--enable_cuda", default=False, help="Enable CUDA")
    parser.add_argument("--seed", default=2099945076, help="seed")
    parser.add_argument("--pre_horizon", type=int, default=pre_horizon)

    # 1. Parameters for environment
    # parser.add_argument("--vector_env_num", type=int, default=2, help="Number of vector envs")
    # parser.add_argument("--vector_env_type", type=str, default='async', help="Options: sync/async")
    # parser.add_argument("--gym2gymnasium", type=bool, default=True, help="Convert Gym-style env to Gymnasium-style")
    parser.add_argument("--action_type", type=str, default="continu", help="Options: continu/discret")
    parser.add_argument("--is_render", type=bool, default=False, help="Draw environment animation")
    parser.add_argument("--is_adversary", type=bool, default=False, help="Adversary training")
    parser.add_argument("--is_constrained", type=bool, default=False, help="Adversary training")
    ################################################
    # 2.1 Parameters of value approximate function
    parser.add_argument(
        "--value_func_name",
        type=str,
        default="StateValue",
        help="Options: StateValue/ActionValue/ActionValueDis/ActionValueDistri",
    )
    parser.add_argument("--value_func_type", type=str, default="MLP", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS")
    value_func_type = parser.parse_known_args()[0].value_func_type
    parser.add_argument("--value_hidden_sizes", type=list, default=[256, 256])
    parser.add_argument(
        "--value_hidden_activation", type=str, default="relu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument("--value_output_activation", type=str, default="linear", help="Options: linear/tanh")

    # 2.2 Parameters of policy approximate function
    parser.add_argument(
        "--policy_func_name",
        type=str,
        default="AttentionFullPolicy",
        help="Options: None/DetermPolicy/FiniteHorizonPolicy/StochaPolicy/AttentionPolicy",
    )

    parser.add_argument("--attn_in_per_dim",type=int,default=8)
    parser.add_argument("--attn_out_dim",type=int,default=64)

    parser.add_argument("--attn_begin",type=int,default=112)
    parser.add_argument("--attn_end",type=int,default=183)

    parser.add_argument(
        "--policy_func_type", type=str, default="Attention", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS/Attention"
    )
    parser.add_argument(
        "--policy_act_distribution",
        type=str,
        default="default",
        help="Options: default/TanhGaussDistribution/GaussDistribution",
    )
    policy_func_type = parser.parse_known_args()[0].policy_func_type
    parser.add_argument("--policy_hidden_sizes", type=list, default=[256, 256, 256])
    parser.add_argument(
        "--policy_hidden_activation", type=str, default="gelu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument("--policy_output_activation", type=str, default="linear", help="Options: linear/tanh")

    ################################################
    # 3. Parameters for RL algorithm
    parser.add_argument("--value_learning_rate", type=float, default=1e-4)
    parser.add_argument("--policy_learning_rate", type=float, default=1e-4)
    parser.add_argument("--policy_scheduler", type=json.loads, default={
        "name": "LinearLR",
        "params": {
            "start_factor": 1.0,
            "end_factor": 0.0,
            "total_iters": 200000,
            }
    })
    ################################################
    # 4. Parameters for trainer
    parser.add_argument(
        "--trainer",
        type=str,
        default="off_serial_idsim_trainer",
        help="Options: on_serial_trainer, on_sync_trainer, off_serial_trainer, off_async_trainer, off_sync_trainer",
    )
    # Maximum iteration number
    parser.add_argument("--max_iteration", type=int, default=200000)
    trainer_type = parser.parse_known_args()[0].trainer
    parser.add_argument("--ini_network_dir", type=str, default=None)

    # 4.1. Parameters for off_serial_trainer
    parser.add_argument(
        "--buffer_name", type=str, default="replay_buffer", help="Options:replay_buffer/prioritized_replay_buffer"
    )
    # Size of collected samples before training
    parser.add_argument("--buffer_warm_size", type=int, default=int(1e3))
    # Max size of reply buffer
    parser.add_argument("--buffer_max_size", type=int, default=int(1e5))
    # Batch size of replay samples from buffer
    parser.add_argument("--replay_batch_size", type=int, default=256)
    # Period of sync central policy of each sampler
    parser.add_argument("--sampler_sync_interval", type=int, default=1)

    ################################################
    # 5. Parameters for sampler
    parser.add_argument("--sampler_name", type=str, default="off_sampler", help="Options: on_sampler/off_sampler")
    # Batch size of sampler for buffer store
    parser.add_argument("--sample_batch_size", type=int, default=64)
    # Add noise to actions for better exploration
    parser.add_argument(
        "--noise_params",
        type=dict,
        default={"mean": np.array([0], dtype=np.float32), "std": np.array([0.0], dtype=np.float32),},
    )
    
    ################################################
    # 6. Parameters for evaluator
    parser.add_argument("--evaluator_name", type=str, default="idsim_train_evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--eval_save", type=str, default=False, help="save evaluation data")

    ################################################
    # 7. Data savings
    parser.add_argument("--save_folder", type=str, default=None)
    # Save value/policy every N updates
    parser.add_argument("--apprfunc_save_interval", type=int, default=10000)
    # Save key info every N updates
    parser.add_argument("--log_save_interval", type=int, default=1000)

    ################################################
    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**args)
    # env = create_env(**{**args, "vector_env_num": None})
    args = init_args(env, **args)

    # start_tensorboard(args["save_folder"])
    # Step 1: create algorithm and approximate function
    alg = create_alg(**args)  # create appr_model in algo **vars(args)
    alg.set_parameters({"gamma": 1.0})
    # Step 2: create sampler in trainer
    sampler = create_sampler(**args)
    # Step 3: create buffer in trainer
    buffer = create_buffer(**args)
    # Step 4: create evaluator in trainer
    eval_args = deepcopy(args)
    eval_args['env_config']['use_multiple_path_for_multilane'] = False
    eval_args['env_config']['random_ref_probability'] = 0.0
    eval_args['env_config']["takeover_bias"] = False
    evaluator = create_evaluator(**eval_args)
    # Step 5: create trainer
    trainer = create_trainer(alg, sampler, buffer, evaluator, **args)

    ################################################
    # Start training ... ...
    trainer.train()
    print("Training is finished!")

    ################################################
    # Plot and save training figures
    plot_all(args["save_folder"])
    save_tb_to_csv(args["save_folder"])
    print("Plot & Save are finished!")

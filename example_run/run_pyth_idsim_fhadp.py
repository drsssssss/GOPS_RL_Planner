#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: template for running policy by PolicyRunner

from gops.sys_simulator.call_terminal_cost import load_apprfunc
from gops.sys_simulator.sys_run import PolicyRunner

#####################################
# Run Systems for Comparison
runner = PolicyRunner(
    log_policy_dir_list=["../results/pyth_idsim/FHADP2_231221-124959-new-obs-use-19000",],
    trained_policy_iteration_list=["19000",],
    save_render=False,
    legend_list=["FHADP"],
    use_opt=True,
    opt_args={
        "opt_controller_type": "MPC", # MPC or OPT
        "num_pred_step": 30,
        "gamma": 1,
        "mode": "shooting",
        "minimize_options": {"max_iter": 200, "tol": 1e-3,
                             "acceptable_tol": 1e0,
                             "acceptable_iter": 10,},
        "use_MPC_for_general_env": True,
        "use_terminal_cost": False,
    },
    dt=None,  # time interval between steps
)

runner.run()
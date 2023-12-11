import os

from gops.sys_simulator.idsim_sys_run import IdsimTestEvaluator

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"


evaluator = IdsimTestEvaluator(
    log_policy_dir = "../results/pyth_idsim/DSAC_231207-201908",
    trained_policy_iteration = "290000",
    num_eval_episode = 10,
    is_render=False
)

evaluator.run()

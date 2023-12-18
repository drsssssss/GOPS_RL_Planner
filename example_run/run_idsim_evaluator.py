import os

from gops.sys_simulator.idsim_sys_run import IdsimTestEvaluator

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"


evaluator = IdsimTestEvaluator(
    log_policy_dir = r"D:\Develop\gops-develop\results\pyth_idsim\FHADP2_231212-091300-v12",
    trained_policy_iteration = "20000",
    num_eval_episode = 10,
    is_render=True
)

evaluator.run()

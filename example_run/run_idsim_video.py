import json
import numpy as np
import pathlib
import os
from gops.trainer.idsim_render.animation_crossroad import AnimationCross
from gops.trainer.idsim_render.animation_multilane import AnimationLane
import pickle
import os

# ------------- How to use --------------
# 1. Choose Animation=AnimationLane/AnimationCross
# 2. Edit log_path_root
# 3. Edit scene_id_list
# ---------------------------------------

# add ffmpeg path to system path
current_path = os.environ.get('Path')
new_path = r'D:\Software\TinySoftware\ffmpeg-2023-12-14-git-5256b2fbe6-full_build\bin'
os.environ['Path'] = current_path + os.pathsep + new_path

Animation = AnimationLane
# Animation = AnimationCross

# log_path_root = pathlib.Path("test/230711-172542IDCevaluation")
# log_path_root = pathlib.Path(r'D:\Develop\gops-develop\results\pyth_idsim\FHADP2_231212-091300-v12\231217-165404IDCevaluation')
log_path_root = pathlib.Path(r'D:\Develop\gops-develop\results\pyth_idsim\FHADP2_231212-163916-v15\231218-184942IDCevaluation')
# log_path_root = pathlib.Path(r'D:\Develop\gops-develop\results\pyth_idsim\FHADP2_231218-091038\231218-153832IDCevaluation')

# scene_id_list = ['000', '001', '002', '003']
scene_id_list = ['000']
theme_style = 'light'
# read config.json from log_path_root
config_path = log_path_root/'config.json'
with open(config_path, 'r') as f:
    config = json.load(f)


for scene_id in scene_id_list:
    log_path = log_path_root/scene_id
    save_path = log_path

    episode_list = [log_path for log_path in os.listdir(
        log_path) if (log_path.startswith('episode'))]
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

        animation.generate_animation(episode_data, save_path, i, mode='debug')

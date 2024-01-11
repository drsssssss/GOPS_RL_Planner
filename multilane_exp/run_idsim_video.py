import json
import pathlib
import os
from gops.trainer.idsim_render.animation_crossroad import AnimationCross
from gops.trainer.idsim_render.animation_mf_multilane import AnimationLane
import pickle
import os

# ------------- How to use --------------
# 1. Ensure that ffmpeg is installed in your computer and is added to system path
# 1. Choose Animation=AnimationLane/AnimationCross
# 2. Edit log_path_root
# 3. Edit scene_id_list
# ---------------------------------------

Animation = AnimationLane
# Animation = AnimationCross

log_path_root = pathlib.Path(r'/root/gops/results/idsim_multilane_vec/dsact_pi/12345_250000_2000_10_1_1_10_sur_punish_new_map_new_reward_scale_padding_random_takeover_selcect_lane_pi_short_epi_run0/231227-011912IDCevaluation')

scene_id_list = ['000','001','002','003', '004']
theme_style = 'light' # only works for AnimationCross

if __name__ == '__main__':
    # read config.json from log_path_root
    config_path = log_path_root/'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    for scene_id in scene_id_list:
        log_path = log_path_root/scene_id
        save_path = log_path

        episode_list = [log_path for log_path in os.listdir(log_path) if (log_path.startswith('episode'))]
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

            animation.generate_animation(episode_data, save_path, i, mode='debug', frame_skip= 2, dpi= 60)

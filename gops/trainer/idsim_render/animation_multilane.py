import os
import pickle
from pathlib import Path

from gops.trainer.idsim_idc_evaluator import EvalResult
from gops.trainer.idsim_render.color import EGO_COLOR, SUR_COLOR, EGO_COLOR_WITH_ALPHA, SUR_COLOR_WITH_ALPHA
from gops.trainer.idsim_render.multilane.info_bar import plot_scale, plot_speed_dashboard, plot_action_bar, \
    plot_value_bar
from gops.trainer.idsim_render.multilane.lane import plot_lane_lines
from gops.trainer.idsim_render.plot_time_flow import plot_action, plot_reward, plot_y_ref_phi_ref, plot_vx_vy_r
from matplotlib import gridspec
from matplotlib.figure import Figure

from gops.trainer.idsim_render.animation_base import AnimationBase, create_veh, remove_veh, update_veh

import matplotlib.pyplot as plt
import numpy as np
from idscene.scenario import ScenarioData
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle as Rt

VEH_LENGTH = 5.0
VEH_WIDTH = 1.8

# REF_POINT_NUM = 31
# REF_LINEWIDTH = 2.0
# REF_ALPHA = 0.3


class AnimationLane(AnimationBase):
    def __init__(self, theme_style, fcd_file, config) -> None:
        super().__init__(theme_style, fcd_file, config)
        self.ax1_1, self.ax1_2, self.ax1_3 = None, None, None
        self.ax2_1, self.ax2_2 = None, None
        self.ax3_1 = None
        self.ax4_1, self.ax4_2 = None, None
        self.ref_artist_list = []
        self.scale_artist_list = []
        self.background_artist_list = []
        self.speed_dashboard_artist_list = []
        self.action_bar_artist_list = []
        self.value_bar_artist_list = []
        self.task_name = "multilane"

    def clear_all_list(self):
        super().clear_all_list()
        self.ref_artist_list = []
        self.scale_artist_list = []
        self.background_artist_list = []
        self.speed_dashboard_artist_list = []
        self.action_bar_artist_list = []
        self.value_bar_artist_list = []


    def generate_animation(
        self,
        episode_data: EvalResult,
        save_path: Path,
        episode_index: int,
        fps=10,
        mode='debug',
    ):
        metadata = dict(title='Demo', artist='Guojian Zhan', comment='idsim')
        writer = FFMpegWriter(fps=fps, metadata=metadata)
        save_video_path = save_path / 'video'
        os.makedirs(save_video_path, exist_ok=True)

        # ------------------ initialization -------------------
        self.clear_all_list()

        fig: Figure = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(4, 2)

        print(f'Plotting traffic...')
        ax = self.plot_traffic(episode_data, fig, gs)

        print(f'Plotting figures...')
        eval_dict = self.plot_figures(episode_data, fig, gs)

        # create ego veh
        ego_veh = create_veh(ax, (0, 0), 0, VEH_LENGTH,
                             VEH_WIDTH, facecolor=EGO_COLOR_WITH_ALPHA, edgecolor=EGO_COLOR)

        writer.setup(fig, os.path.join(save_video_path,
                     f'{self.task_name}_{episode_index}.mp4'))

        value = episode_data.paths_value_list
        value = np.array(value) if value else None
        min_value = np.min(value) if value is not None else 0.0
        max_value = np.max(value) if value is not None else 0.0

        self.ref_artist_list = []
        self.scale_artist_list = []
        self.background_artist_list = []
        self.speed_dashboard_artist_list = []
        self.action_bar_artist_list = []
        self.value_bar_artist_list = []

        # ---------------------- update-----------------------
        for step in range(len(episode_data.time_stamp_list)):
            if mode == 'debug' and step % 10 == 0:
                print(f'step={step}/{len(episode_data.time_stamp_list)}')
            cur_time = episode_data.time_stamp_list[step]
            cur_time = round(cur_time * 10) / 10

            # # ---------------- set limit of figure ------------------
            self.adjust_lim(episode_data, eval_dict, step)

            # # ---------------- update ego veh------------------
            ego_x, ego_y, _, _, ego_phi, _ = episode_data.ego_state_list[step]
            update_veh(ego_veh, (ego_x, ego_y), ego_phi, VEH_LENGTH, VEH_WIDTH)

            # # ---------------- update ego ref------------------
            self.update_ego_ref(ax, episode_data, step)

            # # ---------------- update sur participants------------------
            self.update_sur_participants(ax, cur_time, episode_data, step)

            # set center
            screen_width, screen_height = 100, 125
            center_x, center_y = ego_x, ego_y - \
                (screen_height - screen_width) / 2
            ax.set_xlim(center_x - screen_width / 2,
                        center_x + screen_width / 2)
            ax.set_ylim(center_y - screen_height / 2,
                        center_y + screen_height / 2)

            # plot scale
            for scale in self.scale_artist_list:
                scale.remove()
            self.scale_artist_list = plot_scale(
                ax=ax,
                xy=(center_x - screen_width / 2 + 3,
                    ego_y - screen_width / 2 + 2),
                zorder=3,
            )

            # plot background of speed dashboard and action bar
            margin = 5
            for background in self.background_artist_list:
                background.remove()
            self.background_artist_list = [ax.add_patch(Rt(
                (center_x - screen_width / 2 - margin,
                 center_y - screen_height / 2 - margin),
                screen_width + 2 *
                margin, (screen_height - screen_width) + margin,
                facecolor="white", edgecolor="black", zorder=2
            ))]

            # plot speed dashboard
            speed = episode_data.ego_state_list[step][2] * 3.6
            for artist in self.speed_dashboard_artist_list:
                artist.remove()
            self.speed_dashboard_artist_list = plot_speed_dashboard(
                ax=ax,
                xy=(center_x - 33, center_y - screen_height / 2 + 10),
                speed=speed,
                zorder=3,
            )

            # plot action bar
            action = episode_data.action_real_list[step]
            action[1] = -action[1]  # steer, turn right is positive
            for artist in self.action_bar_artist_list:
                artist.remove()
            self.action_bar_artist_list = plot_action_bar(
                ax=ax,
                xy=(center_x - 12, center_y - screen_height / 2 + 5),
                action=action,
                zorder=3,
                action_upper_bound=np.array(self.config['env_config']['real_action_upper_bound']),
                action_lower_bound=np.array(self.config['env_config']['real_action_lower_bound']),
            )

            # plot value bar
            value = episode_data.paths_value_list[step]
            ref_allowable = episode_data.ref_allowable[step]
            for artist in self.value_bar_artist_list:
                artist.remove()
            self.value_bar_artist_list = plot_value_bar(
                ax=ax,
                xy=(center_x + 28, center_y - screen_height / 2 + 6),
                value=value,
                allowable=ref_allowable,
                min_value=min_value,
                max_value=max_value,
                zorder=3,
            )
            fig.savefig('test.png')
            # exit(0)

            writer.grab_frame()
        writer.finish()
        plt.close(fig)
        print('video export success!')

    def update_ego_ref(self, ax, episode_data, step):
        for ref in self.ref_artist_list:
            ref.remove()
        self.ref_artist_list = []
        selected_path_index = episode_data.selected_path_index_list[step]
        for i in range(len(episode_data.reference_list[step])):
            ref = episode_data.reference_list[step][i][:self.REF_POINT_NUM]
            ref_x, ref_y = ref[:, 0], ref[:, 1]
            self.ref_artist_list.append(ax.add_line(Line2D(
                ref_x, ref_y, color=EGO_COLOR if i == selected_path_index else EGO_COLOR_WITH_ALPHA,
                linewidth=self.REF_LINEWIDTH, zorder=101
            )))

    def adjust_lim(self, episode_data, eval_dict, step):
        index_min = max(0, step - 100)
        index_max = min(len(episode_data.time_stamp_list) - 1, step + 100)
        x_lim_min = episode_data.time_stamp_list[index_min]
        x_lim_max = episode_data.time_stamp_list[index_max]
        self.ax1_1.set_xlim(x_lim_min, x_lim_max)
        self.ax1_1.set_ylim(min(eval_dict['vx_list'][index_min:index_max]) * 0.95,
                            max(eval_dict['vx_list'][index_min:index_max]) * 1.05)
        self.ax1_2.set_ylim(min(eval_dict['vy_list'][index_min:index_max]) * 0.95,
                            max(eval_dict['vy_list'][index_min:index_max]) * 1.05)
        self.ax1_3.set_ylim(min(eval_dict['r_list'][index_min:index_max]) * 0.95,
                            max(eval_dict['r_list'][index_min:index_max]) * 1.05)
        self.ax2_1.set_xlim(x_lim_min, x_lim_max)
        self.ax2_1.set_ylim(min(eval_dict['y_ref_list'][index_min:index_max]) * 0.95,
                            max(eval_dict['y_ref_list'][index_min:index_max]) * 1.05)
        self.ax2_2.set_ylim(min(eval_dict['phi_ref_list'][index_min:index_max]) * 0.95,
                            max(eval_dict['phi_ref_list'][index_min:index_max]) * 1.05)
        self.ax3_1.set_xlim(x_lim_min, x_lim_max)
        self.ax3_1.set_ylim(min(episode_data.reward_info['reward'][index_min:index_max]) * 0.95, 0)
        self.ax4_1.set_xlim(x_lim_min, x_lim_max)
        self.ax4_1.set_ylim(min(eval_dict['acc_list'][index_min:index_max]) * 0.95,
                            max(eval_dict['acc_list'][index_min:index_max]) * 1.05)
        self.ax4_2.set_ylim(min(eval_dict['steer_list'][index_min:index_max]) * 0.95,
                            max(eval_dict['steer_list'][index_min:index_max]) * 1.05)

    def plot_figures(self, episode_data, fig, gs):
        eval_dict = {
            'vx_list': [x[0, 0].item() for x in episode_data.obs_list],
            'vy_list': [x[0, 1].item() for x in episode_data.obs_list],
            'r_list': [x[0, 2].item() for x in episode_data.obs_list],
            'acc_list': [x[0, 5].item() for x in episode_data.obs_list],
            'steer_list': [x[0, 6].item() * 180 / np.pi for x in episode_data.obs_list],
            "y_ref_list": [x[0, 7 + 31].item() for x in episode_data.obs_list],
            "phi_ref_list": [np.arccos(x[0, 7 + 31 + 31].item()) * 180 / np.pi for x in episode_data.obs_list],
            'step_list': episode_data.time_stamp_list,
        }
        del episode_data.reward_info['collision_flag']
        # ego vx, vy, yaw rate
        self.ax1_1, self.ax1_2, self.ax1_3 = plot_vx_vy_r(eval_dict, fig, gs)
        # reference delta y and delta phi
        self.ax2_1, self.ax2_2 = plot_y_ref_phi_ref(eval_dict, fig, gs)
        # all the non-zero reward
        self.ax3_1 = plot_reward(episode_data, eval_dict, fig, gs)
        # real acc, real steer angle
        self.ax4_1, self.ax4_2 = plot_action(eval_dict, fig, gs)
        return eval_dict

    def plot_traffic(self, episode_data, fig, gs):
        ax = fig.add_subplot(gs[:, 0])  # traffic scene plot
        fig.tight_layout(rect=[0, 0.05, 0.85, 0.95])
        fig.subplots_adjust(wspace=0.1, hspace=0.4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        # plot lane lines
        map_path = episode_data.map_path
        map_data_path = f'{map_path}/scene.pkl'
        with open(map_data_path, 'rb') as f:
            scenario_data: ScenarioData = pickle.load(f)
            network = scenario_data.network
        plot_lane_lines(ax, network, zorder=0)
        return ax




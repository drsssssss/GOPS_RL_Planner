import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

from gops.trainer.idsim_idc_evaluator import EvalResult
from matplotlib import gridspec
from matplotlib.axes import Axes

from animation_base import AnimationBase, create_veh, remove_veh, update_veh, veh2vis, rad_to_deg

from process_fcd import FCDLog
import matplotlib.pyplot as plt
import numpy as np
from idscene.network import SumoNetwork
from idscene.scenario import ScenarioData
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Rectangle as Rt
from shapely import offset_curve


VEH_LENGTH = 5.0
VEH_WIDTH = 1.8
EGO_COLOR = 'darkred'
SUR_COLOR = 'darkblue'

REF_POINT_NUM = 31
REF_LINEWIDTH = 2.0
REF_ALPHA = 0.3

ACTION_BAR_COLOR = 'lightcoral'
VALUE_BAR_COLOR = 'dodgerblue'


class AnimationLane(AnimationBase):
    def __init__(self, theme_style, fcd_file, config) -> None:
        super().__init__(theme_style, fcd_file, config)
        self.task_name = "multilane"

    def plot_lane_lines(self, ax: plt.Axes, network: SumoNetwork, zorder: float):
        lane_width = 3.75
        lane_center_lines = list(network._center_lines.values())
        lane_center_lines.insert(0, offset_curve(
            lane_center_lines[0], -lane_width))
        for i, line in enumerate(lane_center_lines):
            if i <= len(lane_center_lines) // 2:
                offset = lane_width / 2
            else:
                offset = -lane_width / 2
            lane_center_lines[i] = offset_curve(line, offset)
        for i, v in enumerate(lane_center_lines):
            x, y = v.xy
            if i in [0, len(lane_center_lines) // 2, len(lane_center_lines) // 2 + 1]:
                ax.add_line(Line2D(x, y, linewidth=1,
                            color="black", zorder=zorder))
            else:
                ax.add_line(Line2D(x, y, linewidth=1, color="black",
                                   linestyle=(0, (12, 18)), zorder=zorder))

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
        fig, axes = plt.subplots(figsize=(20, 12.5))
        gs = gridspec.GridSpec(4, 2)
        ax = fig.add_subplot(gs[:, 0])  # traffic scene plot

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

        # plot lane lines
        map_path = episode_data.map_path
        map_data_path = f'{map_path}/scene.pkl'
        with open(map_data_path, 'rb') as f:
            scenario_data: ScenarioData = pickle.load(f)
            network = scenario_data.network
        self.plot_lane_lines(ax, network, zorder=0)

        print(f'Plotting figures...{episode_index}')
        eval_dict = {
            'vx_list': [x[0] for x in episode_data.obs_list],
            'vy_list': [x[1] for x in episode_data.obs_list],
            'r_list': [x[2] for x in episode_data.obs_list],
            'acc_list': [x[5] for x in episode_data.obs_list],
            'steer_list': [x[6] * 180 / np.pi for x in episode_data.obs_list],
            "y_ref_list": [x[7 + 31] for x in episode_data.obs_list],
            "phi_ref_list": [np.arccos(x[7 + 31 + 31]) * 180 / np.pi for x in episode_data.obs_list],
            'step_list': episode_data.time_stamp_list,
        }
        reward_dict = {
            k: [info["reward_details"][k] for info in episode_data.reward_info]
            for k in idsim_tb_eval_dict.keys()
            if ('reward' in k) and (k in info['reward_details'].keys())
        }

        for k in reward_dict.keys():
            if k in info['reward_details'].keys():
                reward_dict[k] = [info['reward_details'][k] for info in episode_data.reward_info]

        # ego vx, vy, yaw rate
        ax1: Axes = fig.add_subplot(gs[0, 1])
        ax1.plot(eval_dict['step_list'], eval_dict['vx_list'], '.-', label='vx', color='b')
        ax1.set_ylabel('$v_x$', color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        ax2.plot(eval_dict['step_list'], eval_dict['vy_list'], '.-', label='vy', color='r')
        ax2.set_ylabel('$v_y$', color='r')
        ax2.tick_params('y', colors='r')
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(
            ('outward', 60))  # Move the last y-axis spine over to the right by 60 points
        ax3.plot(eval_dict['step_list'], eval_dict['r_list'], '.-', label='yaw rate', color='g')
        ax3.set_ylabel('yaw rate', color='g')
        ax3.tick_params('y', colors='g')
        ax1.set_title('Vehicle Speeds and Yaw Rate')
        ax1.set_xlabel('Time')

        # reference delta y and delta phi
        ax1: Axes = fig.add_subplot(gs[1, 1])
        ax1.plot(eval_dict['step_list'], eval_dict['y_ref_list'], '.-', label='lateral error')
        ax1.set_ylabel('$y-y_{ref}$', color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        ax2.plot(eval_dict['step_list'], eval_dict['phi_ref_list'], '.-',
                 label='relative orientation', color='r')
        ax2.set_ylabel('$\phi-\phi_{ref}$(degree)', color='r')
        ax2.tick_params('y', colors='r')
        ax2.set_title('Errors with Reference Trajectory')
        ax1.set_xlabel('Time')

        # all the non-zero reward
        ax1: Axes = fig.add_subplot(gs[2, 1])
        for k, v in reward_dict.items():
            if np.abs(np.mean(v)) > 1e-6:
                ax1.plot(eval_dict['step_list'], v, '.-', label=k)
        ax1.set_ylabel('Reward')
        ax1.set_title('Rewards')
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.set_xlabel('Time')

        # real acc, real steer angle
        ax1: Axes = fig.add_subplot(gs[3, 1])
        ax1.plot(eval_dict['step_list'], eval_dict['acc_list'], '.-', label='acceleration',
                 color='b')
        ax1.set_ylabel('acceleration', color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        ax2.plot(eval_dict['step_list'], eval_dict['steer_list'], '.-', label='steering angle',
                 color='r')
        ax2.set_ylabel('steering angle (degree)', color='r')
        ax2.tick_params('y', colors='r')
        ax1.set_xlabel('Time')
        ax1.set_title('Actual Acceleration and Steering Angle')



        # create ego veh
        ego_veh = create_veh(ax, (0, 0), 0, VEH_LENGTH,
                             VEH_WIDTH, facecolor='white', edgecolor=EGO_COLOR)

        writer.setup(fig, os.path.join(save_video_path,
                     f'{self.task_name}_{episode_index}.mp4'))

        value = episode_data.paths_value_list
        value = np.array(value) if value else None
        min_value = np.min(value) if value is not None else 0.0
        max_value = np.max(value) if value is not None else 0.0

        ref_artist_list = []
        surr_list = []
        surr_ref_artist_list = []
        scale_artist_list = []
        background_artist_list = []
        speed_dashboard_artist_list = []
        action_bar_artist_list = []
        value_bar_artist_list = []

        # ---------------------- update-----------------------
        for step in range(len(episode_data.time_stamp_list)):
            if mode == 'debug' and step % 10 == 0:
                print(f'step={step}/{len(episode_data.time_stamp_list)}')
            cur_time = episode_data.time_stamp_list[step]
            cur_time = round(cur_time * 10) / 10

            # # ---------------- update ego veh------------------
            ego_x, ego_y, _, _, ego_phi, _ = episode_data.ego_state_list[step]
            update_veh(ego_veh, (ego_x, ego_y), ego_phi, VEH_LENGTH, VEH_WIDTH)

            # # ---------------- update ego ref------------------
            for ref in ref_artist_list:
                ref.remove()
            ref_artist_list = []
            optimal_path_index = episode_data.selected_path_index_list[step]
            ref = episode_data.reference_list[step][optimal_path_index][:REF_POINT_NUM]
            ref_x, ref_y = ref[:, 0], ref[:, 1]
            ref_artist_list.append(ax.add_line(Line2D(
                ref_x, ref_y, color=EGO_COLOR, linewidth=REF_LINEWIDTH, alpha=REF_ALPHA, zorder=0
            )))
            #TODO: add other refs

            # # ---------------- update sur participants------------------
            # p [2N+1, num_veh, feature_dim]
            # feature [x, y, phi, speed, length, width, mask]
            for p in surr_list:
                remove_veh(p)
            surr_list = []
            surr_states = episode_data.surr_state_list[step][0]
            for p in surr_states:
                x, y, phi, speed, length, width, mask = p
                if mask == 1:
                    surr_list.append(create_veh(ax, (x,y), phi, length, width, facecolor='white', edgecolor=SUR_COLOR))
            # surrounding_vehicles = episode_data.surrounding_vehicles[step]
            # for p in surrounding_vehicles:
            #     x, y, phi, speed, length, width = p.x, p.y, p.phi, p.speed, p.length, p.width
            #     # if mask == 1:
            #     surr_list.append(create_veh(
            #         ax, (x, y), phi, length, width, facecolor='white', edgecolor=SUR_COLOR))

            # plot surr reference
            for sr in surr_ref_artist_list:
                sr.remove()
            surr_ref_artist_list = []
            surr_param = episode_data.surr_state_list[step][:REF_POINT_NUM]
            surr_num = surr_states.shape[0]
            for i in range(surr_num):
                mask = surr_states[i][-1]
                if mask == 1:
                    surr_ref_artist_list.append(ax.add_line(Line2D(
                        surr_param[:, i, 0], surr_param[:, i, 1],
                        color=SUR_COLOR, linewidth=REF_LINEWIDTH, alpha=REF_ALPHA, zorder=0
                    )))

            # set center
            screen_width, screen_height = 100, 125
            center_x, center_y = ego_x, ego_y - \
                (screen_height - screen_width) / 2
            ax.set_xlim(center_x - screen_width / 2,
                        center_x + screen_width / 2)
            ax.set_ylim(center_y - screen_height / 2,
                        center_y + screen_height / 2)

            # plot scale
            for scale in scale_artist_list:
                scale.remove()
            scale_artist_list = plot_scale(
                ax=ax,
                xy=(center_x - screen_width / 2 + 3,
                    ego_y - screen_width / 2 + 2),
                zorder=3,
            )

            # plot background of speed dashboard and action bar
            margin = 5
            for background in background_artist_list:
                background.remove()
            background_artist_list = [ax.add_patch(Rt(
                (center_x - screen_width / 2 - margin,
                 center_y - screen_height / 2 - margin),
                screen_width + 2 *
                margin, (screen_height - screen_width) + margin,
                facecolor="white", edgecolor="black", zorder=2
            ))]

            # plot speed dashboard
            speed = episode_data.ego_state_list[step][2] * 3.6
            for artist in speed_dashboard_artist_list:
                artist.remove()
            speed_dashboard_artist_list = plot_speed_dashboard(
                ax=ax,
                xy=(center_x - 33, center_y - screen_height / 2 + 10),
                speed=speed,
                zorder=3,
            )

            # plot action bar
            action = episode_data.action_real_list[step]
            action[1] = -action[1]  # steer, turn right is positive
            for artist in action_bar_artist_list:
                artist.remove()
            action_bar_artist_list = plot_action_bar(
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
            for artist in value_bar_artist_list:
                artist.remove()
            value_bar_artist_list = plot_value_bar(
                ax=ax,
                xy=(center_x + 28, center_y - screen_height / 2 + 6),
                value=value,
                allowable=ref_allowable,
                min_value=min_value,
                max_value=max_value,
                zorder=3,
            )

            # fig.savefig('/home/zhanguojian/idsim-render-zxt/test.png')
            # exit(0)

            writer.grab_frame()
        writer.finish()
        plt.close(fig)
        print('video export success!')


def plot_scale(ax: plt.Axes, xy: Tuple[float, float], zorder: float):
    artist_list = []

    scale_length = 10
    mark_length = 1

    artist_list.append(ax.add_line(Line2D(
        [xy[0], xy[0] + scale_length], [xy[1], xy[1]],
        linewidth=2, color="black", solid_capstyle="butt", zorder=zorder
    )))
    artist_list.append(ax.add_line(Line2D(
        [xy[0], xy[0]], [xy[1], xy[1] + mark_length],
        linewidth=0.5, color="black", zorder=zorder
    )))
    artist_list.append(ax.add_line(Line2D(
        [xy[0] + scale_length, xy[0] + scale_length], [xy[1], xy[1] + mark_length],
        linewidth=0.5, color="black", zorder=zorder
    )))
    artist_list.append(ax.text(
        xy[0] + scale_length / 2, xy[1] + 1, str(scale_length) + 'm',
        horizontalalignment="center", verticalalignment="bottom", zorder=zorder
    ))

    return artist_list


def plot_speed_dashboard(
    ax: plt.Axes,
    xy: Tuple[float, float],
    speed: float,
    zorder: float,
):
    artist_list = []

    dashboard_radius = 12
    mark_speed_max = 100
    mark_angle_max = np.pi / 6
    mark_num = 11
    mark_length = 1
    sub_mark_num = 3 * mark_num - 2
    sub_mark_length = mark_length / 2
    hinge_radius = 0.7
    hinge_gap = 0.4
    pointer_length = 7.5
    pointer_width = 1

    mark_speed = np.linspace(0, mark_speed_max, mark_num)
    mark_angle = np.pi - np.linspace(
        -mark_angle_max, np.pi + mark_angle_max, mark_num, endpoint=True)
    sub_mark_angle = np.pi - np.linspace(
        -mark_angle_max, np.pi + mark_angle_max, sub_mark_num, endpoint=True)

    for i in range(sub_mark_num):
        # add sub_mark
        artist_list.append(ax.add_line(Line2D(
            [xy[0] + dashboard_radius * np.cos(sub_mark_angle[i]),
                xy[0] + (dashboard_radius - sub_mark_length) * np.cos(sub_mark_angle[i])],
            [xy[1] + dashboard_radius * np.sin(sub_mark_angle[i]),
                xy[1] + (dashboard_radius - sub_mark_length) * np.sin(sub_mark_angle[i])],
            linewidth=1, color="black", zorder=zorder
        )))

    for i in range(mark_num):
        # add mark
        artist_list.append(ax.add_line(Line2D(
            [xy[0] + dashboard_radius * np.cos(mark_angle[i]),
                xy[0] + (dashboard_radius - mark_length) * np.cos(mark_angle[i])],
            [xy[1] + dashboard_radius * np.sin(mark_angle[i]),
                xy[1] + (dashboard_radius - mark_length) * np.sin(mark_angle[i])],
            linewidth=1, color="black", zorder=zorder
        )))
        # add mark speed
        artist_list.append(ax.text(
            xy[0] + (dashboard_radius - mark_length - 2) *
            np.cos(mark_angle[i]),
            xy[1] + (dashboard_radius - mark_length - 2) *
            np.sin(mark_angle[i]),
            str(int(mark_speed[i])),
            horizontalalignment="center", verticalalignment="center", zorder=zorder
        ))
        # add pointer
        pointer_angle = np.pi + mark_angle_max - speed / \
            mark_speed_max * (np.pi + 2 * mark_angle_max)
        artist_list.append(ax.add_patch(Polygon(
            ((xy[0] + pointer_width / 2 * np.cos(pointer_angle - np.pi / 2),
                xy[1] + pointer_width / 2 * np.sin(pointer_angle - np.pi / 2)),
                (xy[0] + pointer_width / 2 * np.cos(pointer_angle + np.pi / 2),
                 xy[1] + pointer_width / 2 * np.sin(pointer_angle + np.pi / 2)),
                (xy[0] + pointer_length * np.cos(pointer_angle),
                 xy[1] + pointer_length * np.sin(pointer_angle))),
            facecolor="black", linewidth=0, joinstyle="miter", zorder=zorder
        )))
        artist_list.append(ax.add_patch(Circle(
            xy, radius=hinge_radius + hinge_gap, linewidth=hinge_gap,
            facecolor="white", edgecolor="white", zorder=zorder
        )))
        artist_list.append(ax.add_patch(Circle(
            xy, radius=hinge_radius, linewidth=1, facecolor="white",
            edgecolor="black", zorder=zorder
        )))

    return artist_list


def plot_action_bar(
    ax: plt.Axes,
    xy: Tuple[float, float],
    action: np.ndarray,
    zorder: float,
    action_upper_bound: np.ndarray,
    action_lower_bound: np.ndarray,
):
    artist_list = []

    bar_edge_width = 30
    bar_edge_height = 3
    bar_face_width_max = bar_edge_width / 2
    bar_face_height = bar_edge_height

    action_num = 2
    action_name = ("Acceleration", "Steering")
    action_description = (("Brk", "Acc"), ("Left", "Right"))

    def compute_action_bar_ratio(
            action: float, min_action: float, max_action: float, mid_action: float = 0.0):
        if action < mid_action:
            ratio = (action - mid_action) / (mid_action - min_action)
        else:
            ratio = (action - mid_action) / (max_action - mid_action)
        return ratio

    for i in range(action_num):
        bar_edge_x = xy[0]
        bar_edge_y = xy[1] + (bar_edge_height + 5) * i
        bar_face_x = bar_edge_x + bar_edge_width / 2
        bar_face_y = bar_edge_y
        bar_face_width = compute_action_bar_ratio(
            action[-i - 1], action_lower_bound[-i - 1], action_upper_bound[-i - 1]
        ) * bar_face_width_max

        # add bar face
        artist_list.append(ax.add_patch(Rt(
            (bar_face_x, bar_face_y), bar_face_width, bar_face_height,
            facecolor=ACTION_BAR_COLOR, linewidth=0, zorder=zorder
        )))
        # add bar edge
        artist_list.append(ax.add_patch(Rt(
            (bar_edge_x, bar_edge_y), bar_edge_width, bar_edge_height,
            facecolor="none", edgecolor="black", linewidth=1, zorder=zorder
        )))
        # add action name
        artist_list.append(ax.text(
            bar_face_x, bar_edge_y + bar_edge_height + 1,
            action_name[-i - 1], horizontalalignment="center", zorder=zorder
        ))
        # add descriptions
        artist_list.append(ax.text(
            bar_edge_x - 1, bar_edge_y + 1,
            action_description[-i - 1][0], horizontalalignment="right", zorder=zorder
        ))
        artist_list.append(ax.text(
            bar_edge_x + bar_edge_width + 1, bar_edge_y + 1,
            action_description[-i - 1][1], horizontalalignment="left", zorder=zorder
        ))
    return artist_list


def plot_value_bar(
    ax: plt.Axes,
    xy: Tuple[float, float],
    value: List[float],
    allowable: List[bool],
    min_value: float,
    max_value: float,
    zorder: float,
):
    artist_list = []

    bar_width = 5
    min_bar_height = 0.4
    max_bar_height = 12
    bar_spacing = 1

    artist_list.append(ax.add_line(Line2D(
        (xy[0] - 0.5, xy[0] + (bar_width + bar_spacing) * len(value) -
         bar_spacing + 0.5),
        (xy[1] - 0.5, xy[1] - 0.5),
        linewidth=1, color="black", zorder=zorder
    )))

    opt_index = None
    opt_value = -np.inf
    for i, v in enumerate(value):
        if v > opt_value and allowable[i]:
            opt_index = i
            opt_value = v

    for i, v in enumerate(value):
        x = xy[0] + (len(value) - 1 - i) * (bar_width + bar_spacing)
        if allowable[i]:
            h = min_bar_height + (v - min_value) / (max_value - min_value) * \
                (max_bar_height - min_bar_height)
        else:
            h = min_bar_height
        if i == opt_index:
            alpha = 1.0
        else:
            alpha = 0.3
        artist_list.append(ax.add_patch(Rt(
            (x, xy[1]), bar_width, h, facecolor=VALUE_BAR_COLOR,
            linewidth=0, alpha=alpha, zorder=zorder
        )))
        if i == 0:
            text = "Right"
        elif i == len(value) - 1:
            text = "Left"
        else:
            text = None
        if text is not None:
            artist_list.append(ax.text(
                x + bar_width / 2, xy[1] - 2.5, text,
                horizontalalignment="center", zorder=zorder
            ))

    return artist_list

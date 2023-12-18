from typing import Tuple, List

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Circle, Rectangle as Rt

ACTION_BAR_COLOR = 'lightcoral'
VALUE_BAR_COLOR = 'dodgerblue'

def plot_scale(ax: Axes, xy: Tuple[float, float], zorder: float):
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
    ax: Axes,
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
    ax: Axes,
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
    ax: Axes,
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
        # show value
        artist_list.append(ax.text(
            x + bar_width / 2, xy[1] + max_bar_height + 2.5, f'{v:.2f}',
            weight='bold' if i == opt_index else 'normal', fontsize=8,
            horizontalalignment="center", zorder=zorder
        ))

    return artist_list


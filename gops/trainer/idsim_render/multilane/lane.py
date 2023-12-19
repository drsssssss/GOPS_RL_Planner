from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from shapely import offset_curve

from idscene.network import SumoNetwork


def plot_lane_lines(ax: Axes, network: SumoNetwork, zorder: float):
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

import os
import pickle
from idscene.network.network import SumoNetwork
from idscene.scenario import ScenarioData
import matplotlib.pyplot as plt
from gops.trainer.idsim_render.utils import get_polygon_points, get_yellow_line
from shapely.geometry import LineString, CAP_STYLE, JOIN_STYLE
from gops.trainer.idsim_render.render_params import theme_color_dict, veh_lane_list, cyl_lane_list, lane_linewidth, \
    zebra_length, zebra_width, zebra_interval, lane_width


def static_view(network: SumoNetwork, theme_style: str):
    theme_color = theme_color_dict[theme_style]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300, gridspec_kw={
                           'left': 0, 'right': 1, 'bottom': 0, 'top': 1})
    ax.set_aspect('equal')
    fig.set_facecolor(theme_color['background'])
    ax.set_facecolor(theme_color['background'])
    ax.set_axis_off()
    # fig.tight_layout()

    # plot junction
    junction_points = get_polygon_points(
        network.get_junction_polygon("1").convex_hull)
    ax.fill(*zip(*junction_points), color=theme_color['road'], linewidth=0.0)
    ax.plot(*zip(*junction_points), color=theme_color['line_edge'], linewidth=lane_linewidth, zorder=100)

    # plot edge and lanes
    for edge in network.edges:
        edge_id = edge.getID()
        edge_function = edge.getFunction()
        if edge_function == "":
            # Normal case: normal edge
            # plot edge
            ax.plot(*zip(*get_polygon_points(network.get_edge_polygon(edge_id))),
                    color=theme_color['line_edge'], linewidth=1.0, zorder=10)

            # plot lane
            lane_id_list = network.get_edge_lanes(edge_id)
            lane_points_list = [get_polygon_points(
                network.get_lane_polygon(lane_id)) for lane_id in lane_id_list]
            lane_points_dict = {lane_id: lane_points for lane_id,
                                lane_points in zip(lane_id_list, lane_points_list)}
            for lane_id in lane_id_list:
                if lane_id[-1] in veh_lane_list:
                    # plot vehicle lane
                    ax.fill(
                        *zip(*lane_points_dict[lane_id]), color=theme_color['road'], linewidth=0.0)
                    ax.plot(
                        *zip(*lane_points_dict[lane_id]), color=theme_color['line_roadlane'], linewidth=lane_linewidth, ls=(0, (5, 5)))
                elif lane_id[-1] in cyl_lane_list:
                    # plot cyclist lane
                    ax.fill(
                        *zip(*lane_points_dict[lane_id]), color=theme_color['line_edge'], linewidth=0.0, alpha=1.0)
                    ax.plot(
                        *zip(*lane_points_dict[lane_id]), color=theme_color['line_edge'], linewidth=lane_linewidth)
                else:
                    pass
                # plot middle orange line separating opposite edges
                if lane_id == lane_id_list[-1] and lane_id[0] == '-':
                    opposite_lane_id = lane_id[1:]
                    middle_line = get_yellow_line(
                        network, lane_id, bias=lane_width/2)
                    ax.plot(middle_line[:, 0], middle_line[:, 1],
                            color=theme_color['line_center'], linewidth=2, zorder=100)
        elif edge_function == "crossing":
            # Special case: crossing
            assert len(edge.getLanes()) == 1
            lane = edge.getLanes()[0]
            lane_id = lane.getID()
            lane_shape = lane.getShape()
            center_line = LineString(lane_shape)

            # plot zebra line
            num = int((center_line.length) / (zebra_interval + zebra_length))
            start_offset = (center_line.length - num *
                            (zebra_interval + zebra_length) + zebra_interval) / 2
            for i in range(num):
                offset = start_offset + i * (zebra_interval + zebra_length)
                tmp_start_point = center_line.interpolate(offset)
                tmp_end_point = center_line.interpolate(offset + zebra_length)
                tmp_line = LineString([tmp_start_point, tmp_end_point])
                polygon = tmp_line.buffer(
                    zebra_width / 2, resolution=1, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
                lane_points_list = get_polygon_points(polygon)
                plt.fill(*zip(*lane_points_list),
                         color=theme_color['line_zebra'], linewidth=0.0, zorder=100)
        elif edge_function == "internal":
            # Special case: crossing
            pass
        elif edge_function == "walkingarea":
            # Special case: walking area
            pass
    return fig, ax


if __name__ == "__main__":
    theme_color = theme_color_dict['light']
    save_path = 'black'
    os.makedirs(save_path, exist_ok=True)
    seed = 0
    feature_id_list = ['0', '1', '2', '3', '4']
    for feature_id in feature_id_list:
        map_path = f'/tmp/idsim-scenarios-feature{feature_id}/cross_{seed}/data.pkl'
        with open(map_path, 'rb') as f:
            scenario_data: ScenarioData = pickle.load(f)
            network = scenario_data.network
            flow = scenario_data.flow
        fig, ax = static_view(network, 'light')
        fig.tight_layout()
        fig.savefig(f'{save_path}/map{feature_id}.png',
                    bbox_inches='tight', pad_inches=0, dpi=1000)
        fig.savefig(f'{save_path}/map{feature_id}.pdf',
                    bbox_inches='tight', pad_inches=0, dpi=1000)

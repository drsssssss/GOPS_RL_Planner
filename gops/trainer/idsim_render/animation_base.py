import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

from process_fcd import FCDLog
import matplotlib.pyplot as plt
import numpy as np
from idscene.network import SumoNetwork
from idscene.scenario import ScenarioData
from idsim_model.params import model_config
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Rectangle as Rt
from shapely import offset_curve


def veh2vis(xy: np.ndarray, phi: float, veh_length: float, veh_width: float):
    R = np.array([[np.cos(phi), np.sin(phi)],
                  [-np.sin(phi), np.cos(phi)]])
    xy_vis = xy - np.array([veh_length / 2, veh_width / 2]) @ R
    return tuple(xy_vis)


def rad_to_deg(rad: float) -> float:
    return rad * 180 / np.pi


def create_veh(
    ax: plt.Axes,
    xy: Tuple[float, float],
    phi: float,
    length: float,
    width: float,
    facecolor: str = None,
    edgecolor: str = None,
    zorder=1,
) -> Rt:
    vehPatch = ax.add_patch(Rt(
        [0, 0], length, width, angle=0., facecolor=facecolor, edgecolor=edgecolor, zorder=zorder, linewidth=1
    ))
    update_veh(vehPatch, xy, phi, length, width)
    return vehPatch


def update_veh(vehPatch: Rt, xy: Tuple[float, float], phi: float, length: float, width: float):
    xy_vis = veh2vis(xy, phi, length, width)
    vehPatch.set_xy(xy_vis)
    vehPatch.set_angle(rad_to_deg(phi))


def remove_veh(vehPatch: Rt):
    vehPatch.remove()


class AnimationBase:
    def __init__(self, theme_style='light', fcd_file=None, config=None) -> None:
        self.theme_style = theme_style
        self.fcd_log = FCDLog.from_file(fcd_file)
        self.config = config

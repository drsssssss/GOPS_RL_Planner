import shapely
from shapely.geometry import Polygon, LineString
from idscene.network.network import SumoNetwork
import numpy as np

def get_polygon_points(polygon: Polygon):
    return list(polygon.exterior.coords)

def get_line_points(line: LineString):
    line_points = list(line.coords)
    px, py = zip(*line_points)
    return px, py

def get_yellow_line(network:SumoNetwork, line, bias):
    line_coord = list(network.get_lane_center_line(line).coords)
    line_coord = np.array(line_coord)
    slope = (line_coord[1] - line_coord[0]) / np.linalg.norm(line_coord[1] - line_coord[0])
    normal_slope = np.array([-slope[1], slope[0]])
    line_coord = line_coord + bias * normal_slope
    return line_coord


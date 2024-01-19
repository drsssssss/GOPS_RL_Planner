## veh, cyl and ped are standard names

## static background
theme_color_dict = {
    'light': {
        'road': '#dbdbdb',
        'line': '#555555',
        'line_edge': '#425425',
        'line_zebra': '#b6b6b6',
        'line_roadlane':'#565656',
        'line_center':'#85640a',
        'background': '#8fb74f',
    },
    'dark': {
        'road': '#a5a5a5',
        'line': 'white',
    }
}
ped_lane_list = ['0']
cyl_lane_list = ['1']
veh_lane_list = ['2', '3', '4']

lane_linewidth = 0.8
zebra_length = 0.65 *1.3
zebra_width = 3.0
zebra_interval = 0.95 * 1.3

## animation
# enlarge the size of the bike and pedestrian to make it more visible
cyl_scale = 1.5
ped_scale = 2.0
lane_width = 3.75
multilane_surr_size_dict = {
    "v1": (5.0, 1.8),
    'v2': (5.0, 1.8),
    'vs': (6.0, 2.2),
    'vm': (10,2.5),
    'v3': (2.0 * cyl_scale, 0.65 * cyl_scale),
    'person': (0.21 * ped_scale, 0.48 * ped_scale)
}
crossroad_surr_size_dict = {
    "v1": (5.0, 1.8),
    'v2': (6.0, 2.2),
    'v3': (10.0, 2.5),
    'b1': (2.0 * cyl_scale, 0.65 * cyl_scale),
    'person': (0.21 * ped_scale, 0.48 * ped_scale)
}
traffic_light_length = lane_linewidth
traffic_light_width = lane_width
sur_face_color = 'blue'
ego_face_color = 'magenta'
ref_color_list = ['#4600a3', '#b41f00', '#00a306', '#b47f00']
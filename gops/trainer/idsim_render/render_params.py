## veh, cyl and ped are standard names

## static background
theme_color_dict = {
    'light': {
        'road': '#4f4f5a',
        'line': '#c5c5c5',
        'line_edge': '#b8b8b8',
        'line_zebra': '#FFFFFF',
        'line_roadlane':'#C1C1C1',
        'line_center':'#f9ce41',
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
lane_width = 3.75
veh_length, veh_width = 5.0, 1.8
cyl_length, cyl_width = 2.0, 0.65
ped_length, ped_width = 0.21, 0.48
# enlarge the size of the bike and pedestrian to make it more visible
cyl_scale = 1.5
ped_scale = 2.0
cyl_length, cyl_width = cyl_length * cyl_scale, cyl_width * cyl_scale
ped_length, ped_width = ped_length * ped_scale, ped_width * ped_scale
traffic_light_length = lane_linewidth
traffic_light_width = lane_width
sur_face_color = 'blue'
ego_face_color = 'magenta'
ref_color_list = ['#4BD5FF', '#FF4B4B', '#FFD54B', '#4BFF4B']
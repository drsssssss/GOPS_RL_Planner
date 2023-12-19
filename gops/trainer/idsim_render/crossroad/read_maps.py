from static_background import static_view
from idscene.scenario import ScenarioData
import matplotlib.pyplot as plt
import pickle
import os

resources = '/tmp'
test_list = ['idsim-scenarios-A1',
    'idsim-scenarios-D1-test',
    'idsim-scenarios-D1',
    'idsim-scenarios-A3',
    'idsim-scenarios-A2',
    'idsim-scenarios-C3',
    'idsim-scenarios-C2',
    'idsim-scenarios-D3-test',
    'idsim-scenarios-C1',
    'idsim-scenarios-D2',
    'idsim-scenarios-B2',
    'idsim-scenarios-D2-test',
    'idsim-scenarios-B3',
    'idsim-scenarios-D3',
    'idsim-scenarios-B1']

theme_style = 'light'
save_path = 'maps'
os.makedirs(save_path, exist_ok=True)
for test in test_list:
    maps = [f for f in os.listdir(os.path.join(resources, test))]
    for map in maps:
        map_path = os.path.join(resources, test, map, 'data.pkl')
        with open(map_path, 'rb') as f:
            scenario_data: ScenarioData = pickle.load(f)
        network = scenario_data.network
        flow = scenario_data.flow
        fig, ax = static_view(network, theme_style)
        fig.tight_layout()
        fig.savefig(f'{save_path}/{test}_{map}.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
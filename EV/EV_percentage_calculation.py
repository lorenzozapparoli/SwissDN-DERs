import pandas as pd
import numpy as np
import os 
import json
import geopandas as gpd
import tqdm

### data preparation
script_path = os.path.dirname(os.path.abspath(__file__))
LV_data_path = 'LV/'
dict_path = 'data_processing/'

with open(os.path.join(dict_path, 'dict_folder.json')) as f:
    dict_folder = json.load(f)
municipality_names = pd.read_csv(os.path.join(dict_path, 'dict_grid_municipality.csv'))
# if / is in any row in municipality column, replace it with _
municipality_names['municipality'] = municipality_names['municipality'].apply(lambda x: x.replace('/', '_') if '/' in x else x)

### functions
def concat_all_grids(grid_ids, grids_name):
    node_total = pd.DataFrame()
    print("Concatenating all the nodes and edges of the grids...")
    for n in tqdm.tqdm(range(len(grid_ids))):
        i = grid_ids[n]
        node_id = i+"_nodes"
        node = gpd.read_file(LV_data_path+dict_folder[grids_name]+'/'+node_id)
        if 'el_dmd' in node.columns:
            dmd = node['el_dmd']
            grid_name = i
            node_name = node['osmid']
            node_total = pd.concat([node_total, pd.DataFrame({'grid_name':grid_name, 'node_name':node_name, 'dmd':dmd})], ignore_index=True)
        else:
            print("No 'el_dmd' column in the grid "+i)
    node_total = node_total[node_total['dmd'].apply(lambda x: isinstance(x, (int, float)))]
    return node_total


### main
len_dict = len(dict_folder)
keys = list(dict_folder.keys())
for key in keys[752:]: #el_dmd' column in the grid 2-14_1_3
    # 261 474
    print("Processing grid "+key+" ("+str(list(dict_folder.keys()).index(key)+1)+"/"+str(len_dict)+")")
    path = LV_data_path+dict_folder[key]+'/'
    grid_ids = list(set([str(f.split('.')[0][:-6]) for f in os.listdir(path) if f.startswith(key+'-')]))
    node_total = concat_all_grids(grid_ids, key)
    # filter out node_total with node_total['dmd'] that is str or None
    node_total['percentage'] = node_total['dmd']/node_total['dmd'].sum()
    if not os.path.exists(os.path.join(script_path, 'EV_percentage')):
        os.makedirs(os.path.join(script_path, 'EV_percentage'))
    municipality_name = municipality_names[municipality_names['grid']==int(key)]['municipality']
    # save the file, file name: EV_percentage/key_municipality_name.csv
    node_total.to_csv(os.path.join(script_path, 'EV_percentage', key+'_'+municipality_name.iloc[0]+'.csv'), index=False)
    

    




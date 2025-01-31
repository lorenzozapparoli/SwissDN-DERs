import pandas as pd 
import numpy as np
import os
import json
import geopandas as gpd
from shapely.geometry import Point
import tqdm

# load the geometry information of each municipality
script_path = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_path, 'PV_split')
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
dict_path = 'data_processing/'
muni_geometry = gpd.read_file(dict_path+'municipality_boundary.geojson')
muni_geometry['NAME'] = muni_geometry['NAME'].str.replace('/','_')

# load the lv_grid - municipality mapping
with open (dict_path+'dict_municipality_grid_LV.json') as f:
    dict_municipality_grid = json.load(f)
dict_municipality_grid={k.replace('/','_'): v for k, v in dict_municipality_grid.items()}
keys = list(dict_municipality_grid.keys())

# load PV_data
PV_data = pd.read_csv(os.path.join(script_path, 'PV_data', 'rooftop_PV_CH_annual_by_building.csv'))
PV_data['geometry'] = [Point(xy) for xy in zip(PV_data.XCOORD, PV_data.YCOORD)]
PV_data = gpd.GeoDataFrame(PV_data, crs='EPSG:21781')
PV_data = PV_data.to_crs('EPSG:2056')
PV_data['split'] = False

for key in keys:
    # find PV_data in the municipality
    muni = muni_geometry[muni_geometry['NAME']==key]
    if len(dict_municipality_grid[key])==0:
        continue
    number = dict_municipality_grid[key][0].split('-')[0]
    PV_data_muni = PV_data[PV_data['split']==False]
    PV_data_muni = PV_data_muni[PV_data_muni.within(muni.geometry.values[0])]
    index_muni = PV_data_muni.index
    PV_data.loc[index_muni, 'split'] = True
    PV_data_muni.to_csv(os.path.join(save_path, number+'_PV.csv'), index=False)
    print(len(PV_data_muni), 'PV data in', key,'(',keys.index(key)+1,'/',len(keys),')')
    
    

    
    

    
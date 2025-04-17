"""
Author: Lorenzo Zapparoli
Institution: ETH Zurich
Date: 15/03/2025

Introduction:
This script, `PV_split.py`, is designed to split the buildings from the PV dataset into municipalities. The purpose of this splitting is to ease the processing of building data by dividing it into smaller, more manageable subsets based on geographical boundaries. The script uses convex hulls around grid nodes to identify buildings within each municipality and assigns them accordingly.

The script leverages MPI for parallel processing, allowing multiple processes to handle different municipalities simultaneously. The results are saved as separate CSV files for each municipality in the `Building_split` directory.

Usage:
1. Ensure the required input files (pv data, grid node data, and municipality dictionary) are available in the specified directories.
2. Run the script using MPI to process the data in parallel.
3. The output files will be saved in the `PV_input/Building_split` directory.

Dependencies:
- pandas
- geopandas
- shapely
- mpi4py
- scipy.spatial.ConvexHull
"""

import pandas as pd
import os
import json
import geopandas as gpd
from shapely.geometry import Point

# load the geometry information of each municipality
script_path = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_path, 'PV_input', 'PV_split')
if not os.path.exists(save_path):
    os.makedirs(save_path)

base_pth = os.path.dirname(script_path)
dict_path = os.path.join(script_path, 'Grids', 'Additional_files')
muni_geometry = gpd.read_file(dict_path+'municipality_boundary.geojson')
muni_geometry['NAME'] = muni_geometry['NAME'].str.replace('/','_')

# load the lv_grid - municipality mapping
with open (dict_path+'dict_municipality_grid_LV.json') as f:
    dict_municipality_grid = json.load(f)
dict_municipality_grid={k.replace('/','_'): v for k, v in dict_municipality_grid.items()}
keys = list(dict_municipality_grid.keys())

# load PV_data
PV_data = pd.read_csv(os.path.join(script_path, 'PV_input', 'PV_data', 'rooftop_PV_CH_annual_by_building.csv'))
PV_data['geometry'] = [Point(xy) for xy in zip(PV_data.XCOORD, PV_data.YCOORD)]
PV_data = gpd.GeoDataFrame(PV_data, crs='EPSG:21781')
PV_data = PV_data.to_crs('EPSG:2056')
PV_data['split'] = False

number_prec = []
future_data={}

for key in keys:
    # Find PV_data in the municipality
    muni = muni_geometry[muni_geometry['NAME'] == key]
    if len(dict_municipality_grid[key]) == 0:
        continue

    # Extract all numbers for the municipality
    numbers = [int(item.split('-')[0]) for item in dict_municipality_grid[key]]

    # Filter out numbers that have already appeared
    new_numbers = [num for num in numbers if num not in number_prec]

    if not new_numbers:
        print('Error, all grid numbers have already been used for municipality:', key)
        continue

    # Select the number that occurs most frequently
    number = str(max(set(new_numbers), key=new_numbers.count))
    if int(number) in number_prec:
        print('Error, grid number is in more than one municipality for municipality:', key)
    number_prec.append(int(number))

    PV_data_muni = PV_data[PV_data['split'] == False].copy()
    PV_data_muni = PV_data_muni[PV_data_muni.within(muni.geometry.values[0])]
    index_muni = PV_data_muni.index

    # Save the PV data to the corresponding file
    PV_data_muni.to_csv(os.path.join(save_path, number + '_PV.csv'), index=False)
    print(len(PV_data_muni), 'PV data in', key, number, '(', keys.index(key) + 1, '/', len(keys), ')')

    # Check for other numbers in the municipality and add their data to the main number file
    for num in numbers:
        if num != int(number):
            if num in future_data:
                future_data[num] = pd.concat([future_data[num], PV_data_muni])
            else:
                future_data[num] = PV_data_muni

# Process future data
for num, data in future_data.items():
    other_file_path = os.path.join(save_path, str(num) + '_PV.csv')
    if os.path.exists(other_file_path):
        other_data = pd.read_csv(other_file_path)
        combined_data = pd.concat([other_data, data])
        combined_data.to_csv(other_file_path, index=False)
        print(f'Added future data to {num}_PV.csv')
    else:
        data.to_csv(other_file_path, index=False)
        print(f'Created {num}_PV.csv with future data')

    
    

    
    

    
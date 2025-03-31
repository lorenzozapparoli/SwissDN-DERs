import numpy as np
import pandas as pd
import os
import geopandas as gpd
import json
from scipy.spatial import ConvexHull
from shapely.geometry import Point
from shapely.geometry import Polygon
from scipy.spatial import Voronoi
from geovoronoi import points_to_coords
import warnings
from mpi4py import MPI
warnings.filterwarnings('ignore')


# This script is adapted from the PV_potential_calculator.py script written by Lorenzo.
'''Description of PV data:
https://zenodo.org/records/3609833
GWR_EGID: building ID
XCOORD: x coordinate
YCOORD: y coordinate
EPV_kWh_a: annually PV potential kWh/year
EPV_kWh_a_std uncertainty of annually PV potential kWh/year
Gt_kWh_m2_a: tilted radiation on the surface of the building kWh/m2/year
Gt_kWh_m2_a_std: uncertainty of tilted radiation on the surface of the building kWh/m2/year
APV: available PV area m2
APV_std: uncertainty of available PV area m2
APV_ratio: ratio of APV to the total roof area
EPV_kWh_m2roof_a: annual PV potential per m2 of suitable roof area
'''

BUFFER_DISTANCE = 50
script_path = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_path, 'PV_input', 'PV_data')
grid_path = 'LV/'
dict_path = 'data_processing/'
data_path = os.path.join(script_path, 'PV_input', 'PV_split')
with open(os.path.join(dict_path, 'dict_folder.json')) as f:
    dict_folder = json.load(f)
len_dict = len(dict_folder)

files = os.listdir(data_path)
keys = [f.split('_')[0] for f in files]
keys = list(set(keys))
keys = sorted(keys)

        
def PV_data_preprocessing(path):
    PV_data = pd.read_csv(path)
    PV_data['geometry'] = [Point(xy) for xy in zip(PV_data.XCOORD, PV_data.YCOORD)]
    PV_data = gpd.GeoDataFrame(PV_data, crs='EPSG:21781')
    PV_data = PV_data.to_crs('EPSG:2056')
    PV_data['LV_grid'] = '-1'
    PV_data['LV_osmid'] = -1
    PV_data['distance'] = float('inf')  # Initialize distance column with infinity
    PV_data = PV_data[['SB_UUID', 'geometry', 'LV_grid', 'LV_osmid', 'EPV_kWh_a', 'distance']]
    return PV_data

def load_grid_data(grid_id, municipality):
    lv_node_name = grid_id+"_nodes"
    lv_node_gpd=gpd.read_file(grid_path+dict_folder[municipality]+'/'+lv_node_name)
    try:
        lv_node_gpd['osmid'] = lv_node_gpd['osmid'].astype(int)
        lv_node_gpd['consumers'] = lv_node_gpd['consumers'].astype(bool)
        lv_node_gpd['source'] = lv_node_gpd['source'].astype(bool)
    except:
        print('Error in data type conversion')
    if 'el_dmd' not in lv_node_gpd.columns:
        lv_node_gpd['el_dmd'] = 0
    else:
        lv_node_gpd=lv_node_gpd[lv_node_gpd['el_dmd'].apply(lambda x: isinstance(x, (int, float)))] 
        lv_node_gpd.drop(lv_node_gpd[lv_node_gpd['el_dmd']==0].index, inplace=True)
        lv_node_gpd.reset_index(drop=True, inplace=True)
    return lv_node_gpd
    
def create_convex_hull(lv_node_gpd):
    # create convex hull for the grids 
    hull = ConvexHull([list(point) for point in lv_node_gpd.geometry.apply(lambda x: (x.x,x.y))])
    hull_points = [lv_node_gpd.geometry.apply(lambda x: (x.x,x.y))[i] for i in hull.vertices]
    polygon = Polygon(hull_points)
    buffered_polygon = polygon.buffer(BUFFER_DISTANCE)
    buffered_hull_points = np.array(buffered_polygon.exterior.coords)
    return buffered_polygon, buffered_hull_points

def find_PV_within_hull(building, hull):
    points_within_hull = building
    # points_within_hull = building[((building['LV_grid'] == '-1') | (building['LV_grid'] == -1))&(building['LV_osmid'] == -1)]
    points_within_hull = points_within_hull[points_within_hull.geometry.apply(lambda x: hull.contains(x))] 
    return points_within_hull

def voronoi_allocation(PV_data, grid_id, lv_node_gpd):
    buffered_polygon, buffered_hull_points = create_convex_hull(lv_node_gpd)
    points_within_hull = find_PV_within_hull(PV_data, buffered_polygon)
    points_within_hull = points_within_hull.reset_index(drop=True)
    if len(points_within_hull) == 0:
        print('No points within the hull.')
    coords = points_to_coords(lv_node_gpd.geometry)
    coords = np.append(coords, buffered_hull_points, axis=0)
    vor = Voronoi(coords)
    regions = vor.regions
    vertices = vor.vertices
    point_region = vor.point_region
    for i in range(len(lv_node_gpd)):
        cor = coords[i]
        region = regions[point_region[i]]
        if -1 in region:
            region.remove(-1)
        region_vertices = vertices[region]
        for j in range(len(points_within_hull)):
            bd = points_within_hull.iloc[j]
            if bd['geometry'].within(Polygon(region_vertices)):
                distance = bd['geometry'].distance(Point(cor))
                if distance < points_within_hull.at[j, 'distance']:
                    points_within_hull.at[j, 'LV_osmid'] = lv_node_gpd.iloc[i]['osmid']
                    points_within_hull.at[j, 'LV_grid'] = grid_id
                    points_within_hull.at[j, 'distance'] = distance
    return points_within_hull


def find_PV_for_2_point(PV, lv_node_gpd):
    # find the buildings in the circle with radius 20m around 2 points
    PV_section = pd.DataFrame()
    for i in range(len(lv_node_gpd)):
        point = lv_node_gpd.geometry.iloc[i]
        points_within_radius = PV[PV.geometry.apply(lambda x: point.distance(x) <= BUFFER_DISTANCE)]
        PV_section = pd.concat([PV_section, points_within_radius])

    # Remove duplicates by keeping the row with the lowest distance for each SB_UUID
    PV_section = PV_section.loc[PV_section.groupby('SB_UUID')['distance'].idxmin()].reset_index(drop=True)
    return PV_section

def allocation_for_2node(PV_data, grid_id, lv_node_gpd):
    points_within_hull = find_PV_for_2_point(PV_data, lv_node_gpd)
    if points_within_hull.empty:
        print('No PV found in the grid')
        PV_data = pd.DataFrame()
        return PV_data
    for _, node in lv_node_gpd.iterrows():
        distances = points_within_hull.geometry.apply(lambda g: node.geometry.distance(g))
        for idx, distance in distances.items():
            if distance < points_within_hull.at[idx, 'distance']:
                points_within_hull.at[idx, 'LV_osmid'] = node['osmid']
                points_within_hull.at[idx, 'LV_grid'] = grid_id
                points_within_hull.at[idx, 'distance'] = distance
    return points_within_hull

def merge_update(PV, PV_part):
    cols_to_merge = ['SB_UUID', 'LV_grid', 'LV_osmid', 'distance']
    PV = pd.merge(PV, PV_part[cols_to_merge], how='left', on='SB_UUID', suffixes=('', '_updated'))
    PV['LV_grid'] = PV.apply(lambda row: row['LV_grid_updated'] if row['distance_updated'] < row['distance'] else row['LV_grid'], axis=1)
    PV['LV_osmid'] = PV.apply(lambda row: row['LV_osmid_updated'] if row['distance_updated'] < row['distance'] else row['LV_osmid'], axis=1)
    PV['distance'] = PV.apply(lambda row: row['distance_updated'] if row['distance_updated'] < row['distance'] else row['distance'], axis=1)
    PV = PV.reset_index(drop=True)
    PV = PV.loc[PV.groupby('SB_UUID')['distance'].idxmin()]
    PV = PV.drop(columns=['LV_grid_updated', 'LV_osmid_updated', 'distance_updated'])
    return PV
    
def process_municipality(key):
    """Process a single municipality."""
    try:
        path_municipality = os.path.join(data_path, f'{key}_PV.csv')
        buildings = PV_data_preprocessing(path_municipality)
    except:
        print(f'Error in loading the data for {key}')
        with open(os.path.join(save_path, 'error_log.txt'), 'a') as f:
            f.write(f'Error in loading the data for {key}\n')
        return pd.DataFrame()
    print(f"Processing municipality {key} ({list(dict_folder.keys()).index(key)+1}/{len_dict})")
    path = os.path.join(grid_path, dict_folder[key])
    grid_ids = list(set([str(f.split('.')[0][:-6]) for f in os.listdir(path) if f.startswith(key + '-')]))
    # print(f"There are {len(grid_ids)} grids in this municipality.")
    for grid_id in grid_ids:
        lv_node_gpd = load_grid_data(grid_id, key)
        if lv_node_gpd.shape[0] > 2:
            try:
                building_partly = voronoi_allocation(buildings, grid_id, lv_node_gpd)
            except:
                building_partly = allocation_for_2node(buildings, grid_id, lv_node_gpd)
        elif lv_node_gpd.shape[0]<=2 and lv_node_gpd.shape[0]>0:
            building_partly = allocation_for_2node(buildings, grid_id, lv_node_gpd)
        else:
            print('No nodes in the grid')
            continue
        if building_partly.empty:
            print('No buildings found in the grid', key, grid_id)
            continue
        building_partly_save = building_partly[(building_partly['LV_grid'] != '-1') & (building_partly['LV_osmid'] != -1)]
        buildings = merge_update(buildings, building_partly_save)
    return buildings

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        keys_split = np.array_split(keys, size)
    else:
        keys_split = None

    keys_split = comm.scatter(keys_split, root=0)
    # keys_split = ['1086']
    partial_results = pd.DataFrame()
    for key in keys_split:
        buildings = process_municipality(key)
        partial_results = pd.concat([partial_results, buildings]).reset_index(drop=True)
        partial_results = partial_results.loc[partial_results.groupby('SB_UUID')['distance'].idxmin()]

    # print("Here")
    if rank != 0:
        comm.send(partial_results, dest=0)
    else:
        results = partial_results
        for i in range(1, size):
            received_results = comm.recv(source=i)
            results = pd.concat([results, received_results])

        # Remove duplicates by keeping the row with the lowest distance for each SB_UUID
        results = results.reset_index(drop=True)
        results = results.loc[results.groupby('SB_UUID')['distance'].idxmin()]

        results.to_csv(os.path.join(save_path, 'PV_allocation_LV.csv'), index=False)
        
    comm.Barrier()
    print(f"Process {rank} finished.")
        
        


        
    

"""
Author: Lorenzo Zapparoli
Institution: ETH Zurich
Date: 15/03/2025

Introduction:
This script, `HP_building_split.py`, is designed to split the buildings from the Swiss building registry into municipalities. The purpose of this splitting is to ease the processing of building data by dividing it into smaller, more manageable subsets based on geographical boundaries. The script uses convex hulls around grid nodes to identify buildings within each municipality and assigns them accordingly.

The script leverages MPI for parallel processing, allowing multiple processes to handle different municipalities simultaneously. The results are saved as separate CSV files for each municipality in the `Building_split` directory.

Usage:
1. Ensure the required input files (building registry, grid node data, and municipality dictionary) are available in the specified directories.
2. Run the script using MPI to process the data in parallel.
3. The output files will be saved in the `HP_input/Buildings_sata/Building_split` directory.

Dependencies:
- pandas
- numpy
- geopandas
- shapely
- mpi4py
- scipy.spatial.ConvexHull
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from mpi4py import MPI

# Constants and paths
BUFFER_DISTANCE = 100
script_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_path, 'HP_input', 'Buildings_data')
building_split_path = os.path.join(data_path, 'Building_split')
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grid_path = os.path.join(base_path, 'Grids', 'LV')
dict_path = os.path.join(base_path, 'Grids', 'Additional_files')

# Ensure output directory exists
os.makedirs(building_split_path, exist_ok=True)

# Load building data
buildings_data = pd.read_csv(
    os.path.join(data_path, 'Buildings_data.csv'),
    sep=',',
    low_memory=False
)
buildings_data = gpd.GeoDataFrame(
    buildings_data,
    crs='EPSG:2056',
    geometry=gpd.points_from_xy(buildings_data.GKODE, buildings_data.GKODN)
)

# Initialize flag for buildings assignment
flag = np.zeros(len(buildings_data))

# Load dictionary and municipality data
with open(os.path.join(dict_path, 'dict_folder.json')) as f:
    dict_folder = json.load(f)
keys = list(dict_folder.keys())
len_dict = len(dict_folder)

municipality_names = pd.read_csv(os.path.join(dict_path, 'dict_grid_municipality.csv'))
municipality_names['municipality'] = municipality_names['municipality'].str.replace('/', '_')


def create_convex_hull(lv_node_gpd):
    """
    Create a convex hull polygon around grid nodes, with a buffer.

    Args:
        lv_node_gpd (GeoDataFrame): Grid node geometries.

    Returns:
        buffered_polygon (Polygon): Buffered convex hull polygon.
        buffered_hull_points (ndarray): Coordinates of the buffered polygon exterior.
    """

    points = [point for point in lv_node_gpd.geometry]

    if len(points) < 3:
        # If there are fewer than 3 points, create a buffer around each point and combine them
        buffered_points = [point.buffer(BUFFER_DISTANCE) for point in points]
        combined_polygon = buffered_points[0]
        for buffered_point in buffered_points[1:]:
            combined_polygon = combined_polygon.union(buffered_point)
        buffered_hull_points = np.array(combined_polygon.exterior.coords)
        return combined_polygon, buffered_hull_points

    hull = ConvexHull([(point.x, point.y) for point in points])
    hull_points = [points[i] for i in hull.vertices]
    polygon = Polygon(hull_points)
    buffered_polygon = polygon.buffer(BUFFER_DISTANCE)
    buffered_hull_points = np.array(buffered_polygon.exterior.coords)

    return buffered_polygon, buffered_hull_points


def find_building_within_hull(building, hull):
    """
    Find buildings within a convex hull.

    Args:
        building (GeoDataFrame): Buildings GeoDataFrame.
        hull (Polygon): Convex hull polygon.
    
    Returns:
        points_within_hull (GeoDataFrame): Buildings within the hull.
    """

    points_within_hull = building.loc[flag == 0]
    points_within_hull = points_within_hull[points_within_hull.geometry.apply(hull.contains)]
    return points_within_hull


# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Scatter keys to all processes
if rank == 0:
    keys_split = np.array_split(keys, size)
else:
    keys_split = None

keys_split = comm.scatter(keys_split, root=0)

# Each process processes its assigned keys
partial_results = pd.DataFrame()
for idx, key in enumerate(keys_split):
    path = os.path.join(grid_path, dict_folder[key])

    # Retrieve grid IDs for the current key
    grid_ids = list(set([
        str(f.split('.')[0][:-6]) for f in os.listdir(path) if f.startswith(key + '-')
    ]))

    # Aggregate grid nodes
    node_total = gpd.GeoDataFrame()
    for grid_id in grid_ids:
        lv_node_name = f"{grid_id}_nodes"
        lv_node_gpd = gpd.read_file(os.path.join(grid_path, dict_folder[key], lv_node_name))
        node_total = pd.concat([node_total, lv_node_gpd])
    node_total.reset_index(drop=True, inplace=True)

    # Create convex hull and find buildings within
    buffered_polygon, _ = create_convex_hull(node_total)
    buildings_section = find_building_within_hull(buildings_data, buffered_polygon)

    # Update flag and save results
    if buildings_section.empty:
        print(f"No buildings found in {key}")
        continue
    output_file = os.path.join(building_split_path, f"{key}_buildings.csv")
    buildings_section.to_csv(output_file, index=False)
    print(f"FINISH: ({idx + 1}/{len(keys_split)})")

comm.Barrier()
print(f"Process {rank} finished.")

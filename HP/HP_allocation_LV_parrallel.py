import numpy as np
import pandas as pd
import os
import geopandas as gpd
import json
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from scipy.spatial import Voronoi
from geovoronoi import points_to_coords
import warnings
from mpi4py import MPI
warnings.filterwarnings('ignore')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_processors = comm.Get_size()

script_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_path, 'HP_input','Buildings_data', 'Building_split_check')
grid_path = 'LV/'
dict_path = 'data_processing/'
save_path = os.path.join(script_path,'HP_output', 'HP_allocation_LV')
with open(os.path.join(dict_path, 'dict_folder.json')) as f:
    dict_folder = json.load(f)
len_dict = len(dict_folder)
keys = list(dict_folder.keys())
BUFFER_DISTANCE = 5

def load_grid_data(grid_id, municipality):
    """
    Load LV node data for a specific grid and municipality.

    Args:
        grid_id (str): The unique identifier for the grid.
        municipality (str): The name of the municipality.

    Returns:
        GeoDataFrame: Processed GeoDataFrame containing LV node data.
    """
    lv_node_name = grid_id + "_nodes"
    lv_node_gpd = gpd.read_file(grid_path + dict_folder[municipality] + '/' + lv_node_name)

    # Convert columns to appropriate data types if they exist
    try:
        lv_node_gpd['osmid'] = lv_node_gpd['osmid'].astype(int)
        lv_node_gpd['consumers'] = lv_node_gpd['consumers'].astype(bool)
        lv_node_gpd['source'] = lv_node_gpd['source'].astype(bool)
    except:
        print('Error in data type conversion')

    # Ensure the 'el_dmd' column exists and filter rows with non-zero values
    if 'el_dmd' not in lv_node_gpd.columns:
        lv_node_gpd['el_dmd'] = 0
    else:
        lv_node_gpd = lv_node_gpd[lv_node_gpd['el_dmd'].apply(lambda x: isinstance(x, (int, float)))]
        lv_node_gpd.drop(lv_node_gpd[lv_node_gpd['el_dmd'] == 0].index, inplace=True)
        lv_node_gpd.reset_index(drop=True, inplace=True)

    return lv_node_gpd

def create_convex_hull(lv_node_gpd):
    """
    Create a convex hull around LV nodes and buffer it.

    Args:
        lv_node_gpd (GeoDataFrame): GeoDataFrame containing LV node geometries.

    Returns:
        Tuple:
            Polygon: Buffered convex hull polygon.
            np.ndarray: Exterior coordinates of the buffered convex hull.
    """
    # Construct the convex hull using the node coordinates
    hull = ConvexHull([list(point) for point in lv_node_gpd.geometry.apply(lambda x: (x.x, x.y))])
    hull_points = [lv_node_gpd.geometry.apply(lambda x: (x.x, x.y))[i] for i in hull.vertices]

    # Create a polygon from hull points and buffer it
    polygon = Polygon(hull_points)
    buffered_polygon = polygon.buffer(BUFFER_DISTANCE)

    # Extract the exterior coordinates of the buffered polygon
    buffered_hull_points = np.array(buffered_polygon.exterior.coords)

    return buffered_polygon, buffered_hull_points

def find_building_within_hull(building, hull):
    """
    Find buildings located within a given convex hull.

    Args:
        building (GeoDataFrame): GeoDataFrame of building data with geometries.
        hull (Polygon): Polygon object representing the convex hull.

    Returns:
        GeoDataFrame: Subset of buildings within the convex hull.
    """
    # Filter buildings that are not yet assigned to any LV grid
    points_within_hull = building[((building['LV_grid'] == '-1') | (building['LV_grid'] == -1)) & (building['LV_osmid'] == -1)]

    # Check if building geometries are within the hull
    points_within_hull = points_within_hull[points_within_hull.geometry.apply(lambda x: hull.contains(x))]

    return points_within_hull

def building_allocation(buildings, grid_id, lv_node_gpd):
    """
    Allocate buildings to a grid using convex hull and Voronoi partitioning.
    Args:
        buildings (GeoDataFrame): GeoDataFrame containing building data.
        grid_id (str): The unique identifier for the grid.
        lv_node_gpd (GeoDataFrame): GeoDataFrame containing LV node geometries.
    Returns:
        GeoDataFrame: Updated GeoDataFrame with building assignments to LV grids.
    """
    # Create a buffered convex hull for the LV nodes
    buffered_polygon, buffered_hull_points = create_convex_hull(lv_node_gpd)

    # Find buildings within the convex hull
    points_within_hull = find_building_within_hull(buildings, buffered_polygon)
    points_within_hull = points_within_hull.reset_index(drop=True)

    if len(points_within_hull) == 0:
        print('No points within the hull.')
        return points_within_hull

    # Partition all the LV nodes using Voronoi
    coords = points_to_coords(lv_node_gpd.geometry)
    coords = np.append(coords, buffered_hull_points, axis=0)
    vor = Voronoi(coords)
    regions = vor.regions
    vertices = vor.vertices
    point_region = vor.point_region

    # Assign buildings to the closest Voronoi region
    for i in range(len(lv_node_gpd)):
        cor = coords[i]
        region = regions[point_region[i]]
        if -1 in region:
            region.remove(-1)
        region_vertices = vertices[region]

        for j in range(len(points_within_hull)):
            bd = points_within_hull.iloc[j]
            if bd['geometry'].within(Polygon(region_vertices)):
                points_within_hull.at[j, 'LV_osmid'] = lv_node_gpd.iloc[i]['osmid']
                points_within_hull.at[j, 'LV_grid'] = grid_id

    print('Building allocation for grid', grid_id, 'is done.')
    return points_within_hull


def find_building_for_2_point(building, lv_node_gpd):
    """
    For grid with 2 nodes, find buildings within a 20-meter radius around each node.

    Args:
        building (GeoDataFrame): GeoDataFrame containing building data with geometries.
        lv_node_gpd (GeoDataFrame): GeoDataFrame containing LV node geometries.

    Returns:
        GeoDataFrame: Subset of buildings located within the 20-meter radius of any LV node.
    """
    buildings_section = pd.DataFrame()
    for i in range(len(lv_node_gpd)):
        point = lv_node_gpd.geometry.iloc[i]
        # Find buildings within a 20-meter radius of the current node
        points_within_radius = building[building.geometry.apply(lambda x: point.distance(x) <= 20)]
        buildings_section = pd.concat([buildings_section, points_within_radius])
    return buildings_section

def allocation_for_2node(building, grid_id, lv_node_gpd):
    """
    Allocate buildings to a grid based on proximity to LV nodes (for grids with <=2 nodes).

    Args:
        building (GeoDataFrame): GeoDataFrame containing building data with geometries.
        grid_id (str): The unique identifier for the grid.
        lv_node_gpd (GeoDataFrame): GeoDataFrame containing LV node geometries.

    Returns:
        GeoDataFrame: Updated GeoDataFrame with building assignments to LV grids and nodes.
    """
    # Find buildings within the radius of LV nodes
    points_within_hull = find_building_for_2_point(building, lv_node_gpd)
    if points_within_hull.empty:
        print('No buildings found in the grid')
        building = pd.DataFrame()
        return building

    # Assign each building to the nearest node
    for _, node in lv_node_gpd.iterrows():
        distances = points_within_hull.geometry.apply(lambda g: node.geometry.distance(g))
        min_idx = distances.idxmin()  # Index of the nearest building
        building.loc[min_idx, 'LV_osmid'] = node['osmid']
        building.loc[min_idx, 'LV_grid'] = grid_id
    return building

def merge_update(building, building_part):
    """
    Merge updated building data back into the main building GeoDataFrame.

    Args:
        building (GeoDataFrame): Original GeoDataFrame of buildings.
        building_part (GeoDataFrame): GeoDataFrame with updated building assignments.

    Returns:
        GeoDataFrame: Merged GeoDataFrame with updated LV grid and osmid assignments.
    """
    cols_to_merge = ['EGID', 'LV_grid', 'LV_osmid']
    building = pd.merge(building, building_part[cols_to_merge], how='left', on='EGID', suffixes=('', '_updated'))

    # Combine updated values with original values
    building['LV_grid'] = building['LV_grid_updated'].combine_first(building['LV_grid'])
    building['LV_osmid'] = building['LV_osmid_updated'].combine_first(building['LV_osmid'])

    # Drop temporary columns used during merging
    building = building.drop(columns=['LV_grid_updated', 'LV_osmid_updated'])
    return building

def process_municipality(key):
    """
    Process building data for a single municipality and assign buildings to grids.

    Args:
        key (str): The municipality identifier (e.g., a municipality name or code).

    Returns:
        GeoDataFrame: Processed GeoDataFrame with building assignments for the municipality.
    """
    try:
        # Load building data for the municipality
        buildings = pd.read_csv(os.path.join(data_path, f'{key}_buildings.csv'))
        buildings = gpd.GeoDataFrame(buildings, crs='EPSG:2056', geometry=gpd.points_from_xy(buildings.GKODE, buildings.GKODN))
        buildings['LV_grid'] = '-1'  # Initialize LV grid assignments
        buildings['LV_osmid'] = -1  # Initialize LV osmid assignments
    except Exception as e:
        print(f'Error in loading the data for {key}: {e}')
        return pd.DataFrame()

    print(f"Processing municipality {key} ({list(dict_folder.keys()).index(key) + 1}/{len_dict})")
    path = os.path.join(grid_path, dict_folder[key])
    
    # Extract unique grid IDs for the municipality
    grid_ids = list(set([str(f.split('.')[0][:-6]) for f in os.listdir(path) if f.startswith(key + '-')]))
    print(f"There are {len(grid_ids)} grids in this municipality.")

    for grid_id in grid_ids:
        # Load LV node data for the grid
        lv_node_gpd = load_grid_data(grid_id, key)

        # Allocate buildings based on the number of nodes in the grid
        if lv_node_gpd.shape[0] > 2:
            try:
                building_partly = building_allocation(buildings, grid_id, lv_node_gpd)
            except Exception as e:
                print(f"Error in convex hull allocation for grid {grid_id}: {e}")
                building_partly = allocation_for_2node(buildings, grid_id, lv_node_gpd)
        elif 0 < lv_node_gpd.shape[0] <= 2:
            building_partly = allocation_for_2node(buildings, grid_id, lv_node_gpd)
        else:
            print('No nodes in the grid')
            continue

        if building_partly.empty:
            print('No buildings found in the grid')
            continue

        # Merge updated building data back into the main building dataset
        buildings = merge_update(buildings, building_partly)

    return buildings

if __name__ == '__main__':
    if rank == 0:
        keys_split = np.array_split(keys, num_processors)
    else:
        keys_split = None

    # Scatter the keys to all processes
    keys_split = comm.scatter(keys_split, root=0)
    print(f"Rank {rank} received keys: {keys_split}")

    # Each process processes its assigned keys
    partial_results = pd.DataFrame()
    for key in keys_split:
        buildings = process_municipality(key)
        partial_results = pd.concat([partial_results, buildings])

    # Send results back to rank 0
    if rank != 0:
        comm.send(partial_results, dest=0)
    else:
        results = partial_results
        for i in range(1, num_processors):
            received_results = comm.recv(source=i)
            results = pd.concat([results, received_results])
        results = results[results['LV_osmid'] != -1]
        results.to_csv(os.path.join(save_path, 'Building_allocation_LV.csv'), index=False)

    comm.Barrier()
    print(f"Process {rank} finished.")
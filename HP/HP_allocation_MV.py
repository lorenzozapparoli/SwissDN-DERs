"""
Author: Lorenzo Zapparoli
Institution: ETH Zurich
Date: 15/03/2025

Introduction:
This script, `HP_allocation_MV.py`, is designed to allocate buildings from the Swiss building registry to Medium Voltage (MV) grids. The allocation is based on proximity to MV grid nodes and uses convex hulls and Voronoi partitioning to assign buildings to grids. The script processes data for multiple MV grids in parallel using MPI, ensuring efficient handling of large datasets.

The script identifies buildings within the convex hull of MV nodes, assigns them to the closest grid node, and calculates distances. For grids with fewer than three nodes, a simpler proximity-based allocation is used. The results are saved as a CSV file containing the building-to-grid assignments.

Usage:
1. Ensure the required input files (building registry, MV grid node data, and MV grid identifiers) are available in the specified directories.
2. Run the script using MPI to process the data in parallel.
3. The output file will be saved in the `HP/HP_input/Buildings_data` directory.

Dependencies:
- pandas
- numpy
- geopandas
- shapely
- scipy.spatial.ConvexHull
- geovoronoi
- mpi4py
- json
"""

# this is the python script for allocating heat pumps to MV loads
import numpy as np
import pandas as pd
import os
import geopandas as gpd
import json
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from scipy.spatial import Voronoi, voronoi_plot_2d
from geovoronoi import voronoi_regions_from_coords, points_to_coords
import warnings
from mpi4py import MPI
warnings.filterwarnings('ignore')


class HP():
    def __init__(self):
        self.BUFFER_DISTANCE = 3000
        self.script_path = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.script_path, 'HP_input', 'Buildings_data')
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.grid_path = os.path.join(base_path, 'Grids', 'MV')
        self.MV_id = dict()
        self.id = str
        self.save_path = os.path.join(self.script_path, 'HP_output', 'HP_allocation_MV')

    def load_building_data(self):
        buildings = pd.read_csv(self.data_path+'/Buildings_data.csv')
        # Remove entries with PRT above the 95th percentile
        prt_threshold = buildings['PRT'].quantile(0.95)
        buildings = buildings[buildings['PRT'] >= prt_threshold]
        buildings = gpd.GeoDataFrame(buildings, crs='EPSG:2056', geometry=gpd.points_from_xy(buildings.GKODE, buildings.GKODN))
        buildings['MV_grid'] = '-1'
        buildings['MV_osmid'] = -1
        print('Transform the building data to the correct coordinate system.')
        return buildings

    def load_grid_data(self):
        # load mv and lv node and edge data
        mv_node_name = self.id + "_nodes"
        mv_node_gpd = gpd.read_file(os.path.join(self.grid_path, mv_node_name))
        mv_node_gpd['osmid'] = mv_node_gpd['osmid'].astype(int)
        mv_node_gpd['consumers'] = mv_node_gpd['consumers'].astype(bool)
        mv_node_gpd['source'] = mv_node_gpd['source'].astype(bool)
        mv_node_gpd.drop(mv_node_gpd[mv_node_gpd['el_dmd'] == 0].index, inplace=True)
        mv_node_gpd.reset_index(drop=True, inplace=True)
        print('Finish loading grid data {}.'.format(self.id)+" ("+str(list(self.MV_id).index(self.id)+1)+"/"+str(len(self.MV_id))+")")
        return mv_node_gpd

    def create_convex_hull(self,mv_node_gpd):
        # create convex full for the grids
        hull = ConvexHull([list(point) for point in mv_node_gpd.geometry.apply(lambda x: (x.x,x.y))])
        hull_points = [mv_node_gpd.geometry.apply(lambda x: (x.x,x.y))[i] for i in hull.vertices]
        polygon = Polygon(hull_points)
        buffered_polygon = polygon.buffer(self.BUFFER_DISTANCE)
        buffered_hull_points = np.array(buffered_polygon.exterior.coords)
        return buffered_polygon, buffered_hull_points

    def find_building_within_hull(self, building, hull):
        points_within_hull = building[(building['MV_grid'] == '-1') | (building['MV_osmid'] == -1)]
        points_within_hull = points_within_hull[points_within_hull.geometry.apply(lambda x: hull.contains(x))]
        return points_within_hull

    def building_allocation(self, buildings):
        # This function do the voronoi partitioning for the nodes and allocate the buildings to the nodes

        mv_node_gpd = self.load_grid_data()
        try:
            buffered_polygon, buffered_hull_points = self.create_convex_hull(mv_node_gpd)
            points_within_hull = self.find_building_within_hull(buildings, buffered_polygon)
            points_within_hull = points_within_hull.reset_index(drop=True)
        except:
            points_within_hull = pd.DataFrame()
            return points_within_hull
        if len(points_within_hull) == 0:
            print('No points within the hull.')
            points_within_hull = pd.DataFrame()
            return points_within_hull
        coords = points_to_coords(mv_node_gpd.geometry)
        coords = np.append(coords, buffered_hull_points, axis=0)
        vor = Voronoi(coords)
        regions = vor.regions
        vertices = vor.vertices
        point_region = vor.point_region

        for i in range(len(mv_node_gpd)):
            cor = coords[i]
            region = regions[point_region[i]]
            if -1 in region:
                region.remove(-1)
            region_vertices = vertices[region]

            for j in range(len(points_within_hull)):
                bd = points_within_hull.iloc[j]
                if bd['geometry'].within(Polygon(region_vertices)):
                    points_within_hull.at[j, 'MV_osmid'] = mv_node_gpd.iloc[i]['osmid']
                    points_within_hull.at[j, 'MV_grid'] = self.id
        return points_within_hull


def merge_update(building, building_part):
    cols_to_merge = ['EGID', 'MV_grid', 'MV_osmid']
    building = pd.merge(building, building_part[cols_to_merge], how='left', on='EGID',suffixes=('', '_updated'))
    building['MV_grid'] = building['MV_grid_updated'].combine_first(building['MV_grid'])
    building['MV_osmid'] = building['MV_osmid_updated'].combine_first(building['MV_osmid'])
    building = building.drop(columns=['MV_grid_updated', 'MV_osmid_updated'])
    return building


if __name__ == "__main__":
    HP_instance = HP()
    script_path = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dict_path = os.path.join(base_path, 'Grids', 'Additional_files')
    save_path = os.path.join(script_path, 'HP_input', 'Buildings_data', 'Building_allocation_MV.csv')
    with open(os.path.join(dict_path, 'list_test_id_MV.json'), 'r') as f:
        HP_instance.MV_id = json.load(f)

    buildings = HP_instance.load_building_data()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Split the MV_id list among the available processes
    MV_id_split = np.array_split(HP_instance.MV_id, size)
    local_MV_id = MV_id_split[rank]

    for id in local_MV_id:
        print('Processing grid {}...'.format(id))
        HP_instance.id = id
        building_partly = HP_instance.building_allocation(buildings)
        if not building_partly.empty:
            buildings = merge_update(buildings, building_partly)
            print(len(buildings[(buildings['MV_grid'] != '-1') & (buildings['MV_osmid'] != -1)]))

    buildings_save = buildings[(buildings['MV_grid'] != '-1') & (buildings['MV_osmid'] != -1)]

    # Gather the results from all processes
    all_buildings_save = comm.gather(buildings_save, root=0)

    if rank == 0:
        final_buildings_save = pd.concat(all_buildings_save, ignore_index=True)
        final_buildings_save = final_buildings_save.drop_duplicates(subset='EGID', keep='first')
        final_buildings_save = final_buildings_save.groupby('EGID').apply(lambda x: x.sample(1)).reset_index(
            drop=True)
        final_buildings_save.to_csv(save_path, index=False)
        print("MV data has been processed and saved.")

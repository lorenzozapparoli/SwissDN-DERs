"""
Author: Lorenzo Zapparoli
Institution: ETH Zurich
Date: 15/03/2025

Introduction:
This script, `PV_allocation_MV.py`, is designed to allocate photovoltaic (PV) systems to Medium Voltage (MV) grids. The allocation is based on proximity to MV grid nodes and uses convex hulls and Voronoi partitioning to assign PV systems to grids. The script processes data for multiple MV grids in parallel using MPI, ensuring efficient handling of large datasets.

The script identifies PV systems within the convex hull of MV nodes, assigns them to the closest grid node, and calculates distances. The results are saved as a CSV file containing the PV-to-grid assignments.

Usage:
1. Ensure the required input files (PV data, MV grid node data, and MV grid identifiers) are available in the specified directories.
2. Run the script using MPI to process the data in parallel.
3. The output file will be saved in the `PV_input/PV_data` directory.

Dependencies:
- pandas
- numpy
- geopandas
- shapely
- scipy.spatial.ConvexHull
- geovoronoi
- mpi4py
- json
- warnings
"""

import numpy as np
import pandas as pd
import os
import geopandas as gpd
import json
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from geovoronoi import points_to_coords
from shapely.geometry import Point
import warnings
from scipy.spatial import Voronoi
from mpi4py import MPI
warnings.filterwarnings('ignore')


class MV_allocation():
    def __init__(self):
        """
        Initializes the `MV_allocation` class.

        Description:
        - Sets up paths for input and output directories.
        - Defines buffer distance for convex hull creation.
        """
        self.BUFFER_DISTANCE = 3000
        self.script_path = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.script_path, 'PV_input', 'PV_data')
        self.base_path = os.path.dirname(self.script_path)
        self.grids_path = os.path.join(self.base_path, 'Grids')
        self.grid_path = os.path.join(self.base_path, 'Grids', 'MV')
        self.MV_id = dict()
        self.id = str
        self.save_path = os.path.join(self.script_path, 'PV_output', 'HP_allocation_MV')
    
    def load_building_data(self):
        """
        Loads and preprocesses PV building data.

        Returns:
            GeoDataFrame: Processed GeoDataFrame containing PV building data.

        Description:
        - Filters PV systems based on the 95th percentile of annual energy production.
        - Converts data to the appropriate coordinate system.
        """
        buildings = pd.read_csv(os.path.join(self.data_path, 'rooftop_PV_CH_annual_by_building.csv'))
        EPV_threshold = buildings['EPV_kWh_a'].quantile(0.95)
        buildings = buildings[buildings['EPV_kWh_a'] >= EPV_threshold]
        buildings['geometry'] = [Point(xy) for xy in zip(buildings.XCOORD, buildings.YCOORD)]
        buildings = gpd.GeoDataFrame(buildings, crs='EPSG:21781')
        buildings = buildings.to_crs('EPSG:2056')
        buildings = buildings[['SB_UUID', 'geometry', 'EPV_kWh_a']]
        buildings['MV_grid'] = '-1'
        buildings['MV_osmid'] = -1
        return buildings

    def load_grid_data(self):
        """
        Loads MV grid node data for a specific grid.

        Returns:
            GeoDataFrame: Processed GeoDataFrame containing MV node data.

        Description:
        - Reads MV node data from the corresponding file.
        - Filters out nodes with zero electrical demand.
        """
        MV_path = 'PV/MV/'
        mv_node_name = self.id+"_nodes"
        mv_node_gpd=gpd.read_file(MV_path+mv_node_name)
        mv_node_gpd['osmid'] = mv_node_gpd['osmid'].astype(int)
        mv_node_gpd['consumers'] = mv_node_gpd['consumers'].astype(bool)
        mv_node_gpd['source'] = mv_node_gpd['source'].astype(bool)
        mv_node_gpd.drop(mv_node_gpd[mv_node_gpd['el_dmd']==0].index, inplace=True)
        mv_node_gpd.reset_index(drop=True, inplace=True)
        print('Finish loading grid data {}.'.format(self.id)+" ("+str(list(self.MV_id).index(self.id)+1)+"/"+str(len(self.MV_id))+")")
        return mv_node_gpd
    

    def create_convex_hull(self,mv_node_gpd):
        """
        Creates a convex hull around MV nodes and buffers it.

        Args:
            mv_node_gpd (GeoDataFrame): GeoDataFrame containing MV node geometries.

        Returns:
            Tuple:
                Polygon: Buffered convex hull polygon.
                np.ndarray: Exterior coordinates of the buffered convex hull.

        Description:
        - Calculates the convex hull for MV nodes.
        - Buffers the convex hull to include nearby PV systems.
        """
        # create convex full for the grids 
        hull = ConvexHull([list(point) for point in mv_node_gpd.geometry.apply(lambda x: (x.x,x.y))])
        hull_points = [mv_node_gpd.geometry.apply(lambda x: (x.x,x.y))[i] for i in hull.vertices]
        polygon = Polygon(hull_points)
        buffered_polygon = polygon.buffer(self.BUFFER_DISTANCE)
        buffered_hull_points = np.array(buffered_polygon.exterior.coords)
        return buffered_polygon, buffered_hull_points

    def find_building_within_hull(self, building, hull):
        """
        Finds PV systems located within a given convex hull.

        Args:
            building (GeoDataFrame): GeoDataFrame of PV building data.
            hull (Polygon): Polygon object representing the convex hull.

        Returns:
            GeoDataFrame: Subset of PV systems within the convex hull.

        Description:
        - Filters PV systems that are not yet assigned to any MV grid.
        - Checks if each PV system's geometry is contained within the convex hull.
        """
        points_within_hull = building[(building['MV_grid'] == '-1') | (building['MV_osmid'] == -1)]
        points_within_hull = points_within_hull[points_within_hull.geometry.apply(lambda x: hull.contains(x))] 
        return points_within_hull

    def building_allocation(self, buildings):
        """
        Allocates PV systems to a grid using convex hull and Voronoi partitioning.

        Args:
            buildings (GeoDataFrame): GeoDataFrame containing PV building data.

        Returns:
            GeoDataFrame: Updated GeoDataFrame with PV assignments to MV grids.

        Description:
        - Creates a convex hull around MV nodes and identifies PV systems within it.
        - Uses Voronoi partitioning to assign PV systems to the closest MV node.
        - Updates PV data with grid and node assignments.
        """
        # This function do the voronoi partitioning for the nodes and allocate the buildings to the nodes
        
        mv_node_gpd = self.load_grid_data()
        buffered_polygon, buffered_hull_points = self.create_convex_hull(mv_node_gpd)
        points_within_hull = self.find_building_within_hull(buildings, buffered_polygon)
        points_within_hull = points_within_hull.reset_index(drop=True)
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
        
        # print('Allocating buildings to the nodes...')
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
        # print('Building allocation is completed.')
        return points_within_hull


def merge_update(building, building_part):
    """
    Merges updated PV data back into the main PV GeoDataFrame.

    Args:
        building (GeoDataFrame): Original GeoDataFrame of PV systems.
        building_part (GeoDataFrame): GeoDataFrame with updated PV assignments.

    Returns:
        GeoDataFrame: Merged GeoDataFrame with updated MV grid and osmid assignments.

    Description:
    - Updates PV assignments based on the closest distance.
    - Ensures no duplicate assignments for the same PV system.
    """
    cols_to_merge = ['SB_UUID', 'MV_grid', 'MV_osmid']
    building = pd.merge(building, building_part[cols_to_merge], how='left', on='SB_UUID',suffixes=('', '_updated'))
    building['MV_grid'] = building['MV_grid_updated'].combine_first(building['MV_grid'])
    building['MV_osmid'] = building['MV_osmid_updated'].combine_first(building['MV_osmid'])
    building = building.drop(columns=['MV_grid_updated', 'MV_osmid_updated'])
    return building


if __name__ == "__main__":
    """
    Main execution block.

    Description:
    - Initializes MPI and splits MV grid IDs among processes.
    - Allocates PV systems to MV grids in parallel.
    - Gathers results from all processes and saves the final allocation to a CSV file.
    """
    PV = MV_allocation()
    with open(os.path.join(PV.grids_path, 'Additional_files', 'list_test_id_MV.json'), 'r') as f:
        PV.MV_id = json.load(f)
    buildings = PV.load_building_data()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # subtract the MV_id with index 821, this grid only has 1 node
    PV.MV_id = PV.MV_id[0:821] + PV.MV_id[822:]

    # Split the MV_id list among the available processes
    MV_id_split = np.array_split(PV.MV_id, size)
    local_MV_id = MV_id_split[rank]

    for id in local_MV_id:
        print('Processing grid {}...'.format(id))
        PV.id = id
        building_partly = PV.building_allocation(buildings)
        if not building_partly.empty:
            buildings = merge_update(buildings, building_partly)
            print(len(buildings[(buildings['MV_grid'] != '-1') & (buildings['MV_osmid'] != -1)]))

    buildings_save = buildings[(buildings['MV_grid'] != '-1') & (buildings['MV_osmid'] != -1)]

    # Gather the results from all processes
    all_buildings_save = comm.gather(buildings_save, root=0)

    if rank == 0:
        final_buildings_save = pd.concat(all_buildings_save, ignore_index=True)
        # Check for multiple entries with the same SB_UUID and randomly select one to keep
        final_buildings_save = final_buildings_save.drop_duplicates(subset='SB_UUID', keep='first')
        final_buildings_save = final_buildings_save.groupby('SB_UUID').apply(lambda x: x.sample(1)).reset_index(
            drop=True)
        final_buildings_save.to_csv(PV.data_path + '/PV_allocation_MV.csv', index=False)
        print("MV data has been processed and saved.")

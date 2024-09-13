import numpy as np
import pandas as pd
import os
import pandapower as pp
import geopandas as gpd
import json
from scipy.spatial import ConvexHull
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.affinity import scale
from scipy.spatial import Voronoi, voronoi_plot_2d
from geovoronoi import voronoi_regions_from_coords, points_to_coords
from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area
import tqdm
import shutil
import warnings
from shapely import wkt
import seaborn as sns
import matplotlib.pyplot as plt
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

class PV_demand_generation():
    def __init__(self):
        self.BUFFER_DISTANCE = 3000
        self.building_data_path = 'PV_data/rooftop_PV_CH_annual_by_building.csv'
        self.PV_time_series_path = 'PV_data/rooftop_PV_CH_EPV_W_by_building.csv'
        self.LV_id = dict()
        self.MV_id = dict()
        self.folder_lv = dict()
        self.dict_mv_lv = dict()
        self.id = str
        self.save_path = 'PV_data/'
        
    def PV_data_preprocessing(self):
        PV_data = pd.read_csv(self.building_data_path)
        PV_data['geometry'] = [Point(xy) for xy in zip(PV_data.XCOORD, PV_data.YCOORD)]
        PV_data = gpd.GeoDataFrame(PV_data, crs='EPSG:21781')
        PV_data = PV_data.to_crs('EPSG:2056')
        PV_data['MV_grid'] = -1
        PV_data['MV_osmid'] = -1
        PV_data = PV_data[['SB_UUID', 'geometry', 'MV_grid', 'MV_osmid']]
        # save the processed PV data in json format
        PV_data.to_csv('PV_data/PV_Building_processed.csv', index=False)
        return PV_data

    def load_grid_data(self):
        # load mv and lv node and edge data
        MV_path = 'MV/'
        mv_node_name = self.id+"_nodes"
        mv_edge_name = self.id+"_edges"
        mv_node_gpd=gpd.read_file(MV_path+mv_node_name)
        mv_edge_gpd=gpd.read_file(MV_path+mv_edge_name)
        mv_node_gpd['osmid'] = mv_node_gpd['osmid'].astype(int)
        mv_node_gpd['consumers'] = mv_node_gpd['consumers'].astype(bool)
        mv_node_gpd['source'] = mv_node_gpd['source'].astype(bool)
        mv_node_gpd.drop(mv_node_gpd[mv_node_gpd['el_dmd']==0].index, inplace=True)
        mv_node_gpd.reset_index(drop=True, inplace=True)
       
        print('Finish loading grid data {}.'.format(self.id)+" ("+str(list(self.MV_id).index(self.id)+1)+"/"+str(len(self.MV_id))+")")
        return mv_node_gpd, mv_edge_gpd
    

    def create_convex_hull(self,mv_node_gpd):
        # create convex full for the grids 
        hull = ConvexHull([list(point) for point in mv_node_gpd.geometry.apply(lambda x: (x.x,x.y))])
        hull_points = [mv_node_gpd.geometry.apply(lambda x: (x.x,x.y))[i] for i in hull.vertices]
        polygon = Polygon(hull_points)
        buffered_polygon = polygon.buffer(self.BUFFER_DISTANCE)
        buffered_hull_points = np.array(buffered_polygon.exterior.coords)
        return buffered_polygon, buffered_hull_points

    def find_PV_within_hull(self, PV_building, hull):
        # if all the value in the column 'MV_grid' is -1, then pv_points_within_hull = PV_building[PV_building['MV_grid']==-1]
        # if there are some values in the column 'MV_grid' is not -1, then pv_points_within_hull = PV_building[PV_building['MV_grid']!='-1']
        pv_points_within_hull = PV_building[PV_building['MV_grid']=='-1']
        pv_points_within_hull = pv_points_within_hull[pv_points_within_hull.geometry.apply(lambda x: hull.contains(x))] 
        return pv_points_within_hull

    def PV_allocation(self, PV_building, show_plot=False):
        # This function do the voronoi partitioning for the nodes and allocate the PV to the nodes
        
        mv_node_gpd, mv_edge_gpd = self.load_grid_data()
        buffered_polygon, buffered_hull_points = self.create_convex_hull(mv_node_gpd)
        pv_points_within_hull = self.find_PV_within_hull(PV_building, buffered_polygon)
        pv_points_within_hull = pv_points_within_hull.reset_index(drop=True)
        if len(pv_points_within_hull) == 0:
            print('No PV points within the hull.')
        # partition all the mv nodes
        coords = points_to_coords(mv_node_gpd.geometry)
        coords = np.append(coords, buffered_hull_points, axis=0)
        vor = Voronoi(coords)
        regions = vor.regions
        vertices = vor.vertices
        point_region = vor.point_region
        
        print('Allocating PV to the nodes...')
        for i in tqdm.tqdm(range(len(mv_node_gpd))):
            cor = coords[i]
            region = regions[point_region[i]]
            if -1 in region:
                region.remove(-1)
            region_vertices = vertices[region]

            for j in range(len(pv_points_within_hull)):
                pv = pv_points_within_hull.iloc[j]
                if pv['geometry'].within(Polygon(region_vertices)):
                    pv_points_within_hull.at[j, 'MV_osmid'] = mv_node_gpd.iloc[i]['osmid']
                    pv_points_within_hull.at[j, 'MV_grid'] = self.id
        print('PV allocation is completed.')

        if show_plot:
            fig, ax = plt.subplots()
            mv_edge_gpd.plot(ax=ax, color='k', linewidth=1)
            voronoi_plot_2d(vor, ax=ax, show_vertices=False, point_size=3, line_colors='grey', line_width=0.5,
                        line_alpha=0.6,show_points=False)
            # if pv_points_within_hull['MV_grid']==-1, plot it in grey, otherwise plot it in darkorange
            pv_points_within_hull[pv_points_within_hull['MV_grid']=='-1'].plot(ax=ax, color='grey', markersize=0.5)
            pv_points_within_hull[pv_points_within_hull['MV_grid']!='-1'].plot(ax=ax, color='slateblue', markersize=0.5)
            mv_node_gpd.plot(ax=ax, color='lightcoral', markersize=5, zorder=5)
            plt.xlim([buffered_polygon.bounds[0]-1500, buffered_polygon.bounds[2]+1500])
            plt.ylim([buffered_polygon.bounds[1]-1500, buffered_polygon.bounds[3]+1500])
            plt.savefig('PV_data/allocation_by_MV/{}_PV_allocation.png'.format(self.id), dpi=300)
            plt.show()
        return pv_points_within_hull
    
    def demand_assignment(self):
        # This function assign each grid with the PV demand
        # Merge the PV building allocated to the same MV osmid
        pass

if __name__ == "__main__":
    PV_demand_allocation = PV_demand_generation()
    with open ('data_processing/dict_mv_lv.json') as f:
        PV_demand_allocation.dict_mv_lv = json.load(f)
    '''with open ('data_processing/file_folder_lv.json') as f:
        PV_demand_allocation.folder_lv = json.load(f)
    with open ('data_processing/list_test_id_LV.json') as f:
        PV_demand_allocation.LV_id = json.load(f)'''
    with open ('data_processing/list_test_id_MV.json') as f:
        PV_demand_allocation.MV_id = json.load(f)

    try:
        print('Loading the processed PV data...')
        PV_building = pd.read_csv('PV_data/PV_Building_processed.csv')
        PV_building['geometry'] = PV_building['geometry'].apply(wkt.loads)
        PV_building = gpd.GeoDataFrame(PV_building, crs='EPSG:2056', geometry=PV_building.geometry)

    except:
        PV_building = PV_demand_allocation.PV_data_preprocessing()
        print('Transform the PV data to the correct coordinate system.')

    for id in PV_demand_allocation.MV_id[822:]:
        PV_demand_allocation.id = id
        PV_partly = PV_demand_allocation.PV_allocation(PV_building, show_plot=False)
        PV_building = pd.merge(PV_building, PV_partly, how='left', on='SB_UUID',suffixes=('', '_updated'))
        PV_building['MV_grid'] = PV_building['MV_grid_updated'].combine_first(PV_building['MV_grid'])
        PV_building['MV_osmid'] = PV_building['MV_osmid_updated'].combine_first(PV_building['MV_osmid'])
        PV_building['geometry'] = PV_building['geometry_updated'].combine_first(PV_building['geometry'])
        PV_building = PV_building.drop(columns=['MV_grid_updated', 'MV_osmid_updated', 'geometry_updated'])
        index = PV_demand_allocation.MV_id.index(id)
        if index%100 == 0:
            PV_building.to_csv('PV_data/PV_Building_processed.csv', index=False)
            print("saving the changes of allocation to the csv file." +" ("+str(list(PV_demand_allocation.MV_id).index(id)+1)+"/"+str(len(PV_demand_allocation.MV_id))+")")

    PV_building.to_csv('PV_data/PV_Building_processed.csv', index=False)
    print("saving the changes of allocation to the csv file." +" ("+str(list(PV_demand_allocation.MV_id).index(id)+1)+"/"+str(len(PV_demand_allocation.MV_id))+")")


        
    

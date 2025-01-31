# this is the python script for allocating heat pumps to MV loads
import numpy as np
import pandas as pd
import os
import geopandas as gpd
import json
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from geovoronoi import points_to_coords
import tqdm
from shapely.geometry import Point
import warnings
from shapely import wkt
from scipy.spatial import Voronoi
warnings.filterwarnings('ignore')

dict_path = 'data_processing/'

class MV_allocation():
    def __init__(self):
        self.BUFFER_DISTANCE = 3000
        self.script_path = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.script_path, 'PV_input', 'PV_data')
        self.grid_path = 'MV/'
        self.MV_id = dict()
        self.id = str
        self.save_path = os.path.join(self.script_path, 'PV_output','HP_allocation_MV') 
    
    def load_building_data(self):
        try:
            print('Loading the processed building data...')
            buildings = pd.read_csv(self.data_path+'/rooftop_PV_CH_annual_by_building_MV_processed.csv')
            buildings['geometry'] = buildings['geometry'].apply(wkt.loads)
            buildings = gpd.GeoDataFrame(buildings, crs='EPSG:2056', geometry=buildings.geometry)
            print('Finish loading the processed building data from the csv file.')
        except:
            buildings = pd.read_csv(os.path.join(self.data_path, 'rooftop_PV_CH_annual_by_building_MV.csv'))
            buildings['geometry'] = [Point(xy) for xy in zip(buildings.XCOORD, buildings.YCOORD)]
            buildings = gpd.GeoDataFrame(buildings, crs='EPSG:21781')
            buildings = buildings.to_crs('EPSG:2056')
            buildings['MV_grid'] = '-1'
            buildings['MV_osmid'] = -1
            buildings = buildings[['SB_UUID', 'geometry', 'LV_grid', 'LV_osmid','EPV_kWh_a']]
            print('Transform the building data to the correct coordinate system.')
        return buildings

    def load_grid_data(self):
        # load mv and lv node and edge data
        MV_path = 'MV/'
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
        
        print('Allocating buildings to the nodes...')
        for i in tqdm.tqdm(range(len(mv_node_gpd))):
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
        print('Building allocation is completed.')
        return points_within_hull
    
def merge_update(building, building_part):
    cols_to_merge = ['SB_UUID', 'MV_grid', 'MV_osmid']
    building = pd.merge(building, building_part[cols_to_merge], how='left', on='SB_UUID',suffixes=('', '_updated'))
    building['MV_grid'] = building['MV_grid_updated'].combine_first(building['MV_grid'])
    building['MV_osmid'] = building['MV_osmid_updated'].combine_first(building['MV_osmid'])
    building = building.drop(columns=['MV_grid_updated', 'MV_osmid_updated'])
    return building

if __name__ == "__main__":
    PV = MV_allocation()
    with open ('data_processing/list_test_id_MV.json', 'r') as f:
        PV.MV_id = json.load(f)
    buildings = PV.load_building_data()
    # subtract the MV_id with index 821, this grid only has 1 node
    PV.MV_id = PV.MV_id[0:821]+PV.MV_id[822:]
    for id in PV.MV_id: 
        PV.id = id
        building_partly = PV.building_allocation(buildings)
        if building_partly.empty:
            continue
        buildings = merge_update(buildings, building_partly)
        index = PV.MV_id.index(id)
        if index%200 == 0:
            buildings.to_csv(PV.data_path+'/rooftop_PV_CH_annual_by_building_MV_processed.csv', index=False)
            print("saving the changes of allocation to the csv file." +" ("+str(list(PV.MV_id).index(id)+1)+"/"+str(len(PV.MV_id))+")")

    buildings.to_csv(PV.data_path+'/rooftop_PV_CH_annual_by_building_MV_processed.csv', index=False)
    print("saving the changes of allocation to the csv file." +" ("+str(list(PV.MV_id).index(id)+1)+"/"+str(len(PV.MV_id))+")")
    
  
    
    
    




# This is the script to do voronoi partitioning of all the MV trafos in Switzerland
import pandas as pd
import os
import geopandas as gpd
import json
from geovoronoi import voronoi_regions_from_coords
import tqdm
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

with open('data_processing/list_test_id_MV.json', 'r') as f:
    list_test_id_MV = json.load(f)

def generate_MV_trafos(MV_ids=list_test_id_MV):
    try:
        MV_trafos = gpd.read_file('data_processing/MV_trafos.geojson')
        MV_trafos = MV_trafos.to_crs(epsg=2056)
    except:
        MV_trafos = pd.DataFrame()
        for id in tqdm.tqdm(MV_ids):
            nodes = gpd.read_file(f'MV/{id}_nodes')
            trafo_row = nodes[nodes['source']==True]
            trafo_row['id'] = id
            MV_trafos = pd.concat([MV_trafos, trafo_row])
        MV_trafos = MV_trafos.reset_index(drop=True)
    return MV_trafos

def load_boundary():
    boundary = gpd.read_file('data_processing/canton_union.geojson')
    boundary = boundary.to_crs(epsg=2056)
    boundary_union = boundary.unary_union
    return boundary, boundary_union

def voronoi_partitioning(nodes, boundary, show_plot=False,save_file=False):
    region_polys, region_pts = voronoi_regions_from_coords(nodes["geometry"], boundary)
    region_polys = gpd.GeoSeries(region_polys)
    region_pts = pd.Series(region_pts, name='node_id')
    region_pts = region_pts.astype(str)
    region_pts= region_pts.str.split('[',expand=True).drop(columns=[0])
    region_pts= region_pts[1].str.split(']',expand=True).drop(columns=[1])
    region_pts= region_pts.rename(columns={0:'idx'})
    gdf_shape = gpd.GeoDataFrame(geometry=region_polys)
    gdf_shape = gdf_shape.join(region_pts)
    gdf_shape = gdf_shape.astype({'idx': 'int64'})
    gdf_shape = gdf_shape.set_index('idx')
    gdf_shape = gdf_shape.join(nodes.geometry,rsuffix='_r')
    gdf_shape = gdf_shape.rename(columns={'geometry_r':'trafo_location'})
    gdf_shape = gdf_shape.join(nodes['id'])
    gdf_shape = gdf_shape.set_index('id')
    gdf_shape = gdf_shape.sort_index()
    gdf_shape = gdf_shape.reset_index()
    
    if show_plot:
        ax = gdf_shape['geometry'].plot(color='skyblue', edgecolor='black', alpha=0.5, figsize=(10, 8))
        gdf_shape['trafo_location'].plot(ax=ax, color='coral', marker='o', markersize=2, label = 'MV trafo')
        plt.title('Voronoi partitioning of MV trafos')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.show()

    if save_file:
        gdf_shape['trafo_location']=gdf_shape['trafo_location'].to_wkt()
        gdf_shape.to_file('MV_trafo_partitioning.geojson', driver='GeoJSON',crs='EPSG:2056')
    return gdf_shape

if __name__ == '__main__':
    MV_trafos = generate_MV_trafos()
    boundary, boundary_union = load_boundary()
    gdf_shape = voronoi_partitioning(MV_trafos, boundary_union, show_plot=True, save_file=True)
    print("Successfully partitioned MV trafos!")

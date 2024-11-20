import os
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
from joblib import Parallel, delayed
from tqdm import tqdm  

# Paths
script_path = os.path.dirname(os.path.abspath(__file__))
temperature_path = os.path.join(script_path, 'Temperature_data')
buildings_path = os.path.join(script_path, 'Buildings_data')

# Load building and station data
building_data = pd.read_csv(os.path.join(buildings_path, 'buildings_data_processed_3000.csv'))
building_data['geometry'] = building_data['geometry'].apply(wkt.loads)
building_data = gpd.GeoDataFrame(building_data, crs='EPSG:2056', geometry=building_data.geometry)
building_data['station'] = -1

stations = pd.read_csv(os.path.join(temperature_path, 'station_locations.csv'))
stations['geometry'] = [Point(x, y) for x, y in zip(stations['X'], stations['Y'])]
stations = gpd.GeoDataFrame(stations, crs='EPSG:2056', geometry=stations.geometry)

# Function to process a chunk of data
def find_closest_station(building):
    dist = stations.distance(building.geometry)
    station_idx = dist.idxmin()
    return stations.iloc[station_idx]['station']

num_cores = os.cpu_count()  
building_data['station'] = Parallel(n_jobs=num_cores)(
    delayed(find_closest_station)(building) for building in tqdm(building_data.itertuples(), total=len(building_data))
)

building_data.to_csv(os.path.join(buildings_path, 'buildings_data_with_stations.csv'), index=False)

    


    

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import tqdm

base_path = os.path.dirname(os.path.abspath(__file__))
buffer_distance = '3000'
mode = 'full' 
PV_building = pd.read_csv(base_path+'/PV_data/PV_Building_processed_'+buffer_distance+'.csv')
PV_building = PV_building[PV_building['MV_grid']!='-1']
PV_building['MV_osmid'] = PV_building['MV_osmid'].astype(int)
PV_time_series = pd.read_csv(base_path+'/PV_data/rooftop_PV_CH_EPV_W_by_building.csv')
PV_std = pd.read_csv(base_path+'/PV_data/rooftop_PV_CH_EPV_W_std_by_building.csv')
#PV_time_series_area = pd.read_csv('PV_data/rooftop_PV_CH_Gt_W_m2_by_building.csv')

def calculate_daily_average(df):
    # calculate the daily average of 12 days hourly data
    non_time_columns = ['SB_UUID']
    time_columns = [col for col in df.columns if col not in non_time_columns]
    df_time_data = df[time_columns].T
    df_time_data.index = pd.to_datetime(df_time_data.index, errors='coerce')
    df_time_data = df_time_data.dropna()
    df_hourly_avg = df_time_data.groupby(df_time_data.index.hour).mean()
    df_final = pd.concat([df[non_time_columns], df_hourly_avg.T], axis=1)
    return df_final

def mapping(PV_building, df):
    mv_grid_dict = dict(zip(PV_building['SB_UUID'], PV_building['MV_grid']))
    mv_osmid_dict = dict(zip(PV_building['SB_UUID'], PV_building['MV_osmid']))
    df['MV_grid'] = df['SB_UUID'].map(mv_grid_dict)
    df['MV_osmid'] = df['SB_UUID'].map(mv_osmid_dict)
    df = df[df['MV_grid'].notna()]
    return df

def calculate_unmapped(df_time_series_avg):
    # calculate the unmapped rate of energy
    cols = df_time_series_avg.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    df_time_series_avg = df_time_series_avg[cols]
    df_time_series_avg['Max_demand'] = df_time_series_avg.iloc[:,3:].max(axis=1)
    df_time_series_avg['avg_demand'] = df_time_series_avg.iloc[:,3:].mean(axis=1)
    unmapped_avg_demand = df_time_series_avg[(df_time_series_avg['MV_grid']=='-1')|(df_time_series_avg['MV_grid']==-1)]['avg_demand'].sum()
    total_avg_demand = df_time_series_avg['avg_demand'].sum()
    print('The total average demand is: '+str(total_avg_demand))
    print('The total average unmapped demand is: '+str(unmapped_avg_demand))
    unmapped_rate = unmapped_avg_demand/total_avg_demand
    return unmapped_rate
 
def calculate_unmapped_building(df=PV_building):
    # calculate the unmapped rate of buildings
    mapped_buildings = df[(df['MV_grid'] != '-1') & (df['MV_grid'] != -1)]
    unmapped_buildings = df[(df['MV_grid'] == '-1') | (df['MV_grid'] == -1)]
    print('The number of mapped buildings is: '+str(len(mapped_buildings)))
    print('The number of total buildings is: '+str(len(df)))
    unmapped_rate = len(unmapped_buildings)/len(df)
    return unmapped_rate, len(mapped_buildings)

def fill_missing_hour(df):
    # Some hours are not included in the data, fill them with zeros
    df.set_index('SB_UUID', inplace=True)
    df.columns = pd.to_datetime(df.columns)
    unique_days = df.columns.normalize().unique()
    full_columns = pd.DatetimeIndex([])  # Start with an empty DatetimeIndex
    for day in unique_days:
        full_columns = full_columns.append(pd.date_range(day, periods=24, freq='H'))
    df_full = df.reindex(columns=full_columns, fill_value=0)
    df_full.reset_index(inplace=True)
    return df_full

def area_calculation(id, PV_generation_data, irradiance_data):
    # calculate the area of PV panels in each node
    P_data =pd.DataFrame(columns=['grid_id','MV_osmid', 'P_installed (kWp)'])
    # delete the columns with all zeros
    PV_generation_data = PV_generation_data.loc[:, (PV_generation_data != 0).any(axis=0)]
    irradiance_data = irradiance_data.loc[:, (irradiance_data != 0).any(axis=0)]
    area = PV_generation_data.iloc[:,1:]/(irradiance_data.iloc[:,1:]*0.138)
    area = area.replace(0, np.nan)
    area = area.fillna(area.mean(axis=1))
    area['mean'] = area.mean(axis=1)
    P_installed = 285*area['mean']/(1.6*10**3) #kWp
    P_data['grid_id'] = id
    P_data['MV_osmid'] = PV_generation_data['MV_osmid']
    P_data['P_installed (kWp)'] = P_installed
    return P_data
    

def allocation(PV_building, df_time_series_avg, df_std_avg, id, buffer_distance):
    pv_part = PV_building[PV_building['MV_grid'] == id]
    pv_time_series_part = df_time_series_avg[df_time_series_avg['MV_grid'] == id]
    pv_std_part = df_std_avg[df_std_avg['MV_grid'] == id]
    pv_part = pv_part.reset_index(drop=True)

    pv_time_series_part = pv_time_series_part.reset_index(drop=True)
    pv_time_series_part = pv_time_series_part.drop("MV_grid", axis=1)
    data2 = pv_time_series_part.drop("SB_UUID", axis=1).groupby("MV_osmid", dropna=False).sum()
    data2 = data2.reset_index(inplace=False)
    
    pv_std_part = pv_std_part.reset_index(drop=True)
    pv_std_part = pv_std_part.drop("MV_grid", axis=1)
    data3 = pv_std_part.drop("SB_UUID", axis=1).groupby("MV_osmid", dropna=False).sum()
    data3 = data3.reset_index(inplace=False)

    # create folder "allocation results" if not exist
    if not os.path.exists(base_path+'/PV_allocation_results/'):
        os.makedirs(base_path+'/PV_allocation_results/')
    # create folder for each MV grid
    if not os.path.exists(base_path+'/PV_allocation_results/'+ str(id)):
        os.makedirs(base_path+'/PV_allocation_results/' + str(id))
    # save data2 and data3 as csv files, named by MV grid id+demand or std
    data2.to_pickle(base_path+'/PV_allocation_results/'+ str(id) + '/' + str(id) + '_generation.pkl')
    data3.to_pickle(base_path+'/PV_allocation_results/'+ str(id) + '/' + str(id) + '_std.pkl')
    #P_data.to_csv('PV_allocation_results/'+buffer_distance+'/' + str(id) + '/' + str(id) + '_P_installed.csv', index=False)
    
    
if mode == 'avg':
    PV_time_series = calculate_daily_average(PV_time_series)
    PV_std = calculate_daily_average(PV_std)
    #PV_time_series_area = calculate_daily_average(PV_time_series_area)
    df_time_series = mapping(PV_building, PV_time_series)
    df_std = mapping(PV_building, PV_std)
    #df_time_series_area = mapping(PV_building, PV_time_series_area)

elif mode == 'full':
    PV_time_series = fill_missing_hour(PV_time_series)
    PV_std = fill_missing_hour(PV_std)
    #PV_time_series_area = fill_missing_hour(PV_time_series_area)
    df_time_series = mapping(PV_building, PV_time_series)
    df_std = mapping(PV_building, PV_std)
    #df_time_series_area = mapping(PV_building, PV_time_series_area)

MV_grids_ids = PV_building['MV_grid'].unique()
# rearrange the order of MV grid ids to start with the smallest id
MV_grids_ids = np.sort(MV_grids_ids)

#P_total = pd.DataFrame(columns=['grid_id','MV_osmid', 'P_installed (kWp)'])
#for id in tqdm.tqdm(MV_grids_ids):
    #allocation(PV_building, df_time_series, df_std, id, buffer_distance)
    #P_total = pd.concat([P_total, P_data], axis=0)
#P_total.to_csv('PV_allocation_results/'+buffer_distance+'/P_total.csv', index=False)
unmapped_rate = calculate_unmapped(df_time_series)
print('The energy unmapped rate is: ' + str(unmapped_rate))
unmapped_rate_building, mapped_number = calculate_unmapped_building()
print('The building unmapped rate is: ' + str(unmapped_rate_building)+', and the number of mapped buildings is: '+str(mapped_number))

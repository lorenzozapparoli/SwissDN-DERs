import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import tqdm
buffer_distance = '5000'
mode = 'full' # 'avg' or 'full'
PV_building = pd.read_csv('PV_data/PV_Building_processed_'+buffer_distance+'.csv')
PV_building['MV_osmid'] = PV_building['MV_osmid'].astype(int)
PV_time_series = pd.read_csv('PV_data/rooftop_PV_CH_EPV_W_by_building.csv')
PV_std = pd.read_csv('PV_data/rooftop_PV_CH_EPV_W_std_by_building.csv')

def calculate_daily_average(df):
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
    return df

def calculate_unmapped(df_time_series_avg):
    cols = df_time_series_avg.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    df_time_series_avg = df_time_series_avg[cols]
    df_time_series_avg['Max_demand'] = df_time_series_avg.iloc[:,3:].max(axis=1)
    df_time_series_avg['avg_demand'] = df_time_series_avg.iloc[:,3:].mean(axis=1)
    mapped_avg_demand = df_time_series_avg[df_time_series_avg['MV_grid']!='-1']['avg_demand'].sum()
    unmapped_avg_demand = df_time_series_avg[df_time_series_avg['MV_grid']=='-1']['avg_demand'].sum()
    total_avg_demand = df_time_series_avg['avg_demand'].sum()
    unmapped_rate = unmapped_avg_demand/total_avg_demand
    return unmapped_rate
 
def calculate_unmapped_building(df=PV_building):
    mapped_buildings = df[df['MV_grid'] != '-1']
    unmapped_buildings = df[df['MV_grid'] == '-1']
    unmapped_rate = len(unmapped_buildings)/len(df)
    return unmapped_rate, len(mapped_buildings)

def fill_missing_hour(df):
    df.set_index('SB_UUID', inplace=True)
    df.columns = pd.to_datetime(df.columns)
    unique_days = df.columns.normalize().unique()
    full_columns = pd.DatetimeIndex([])  # Start with an empty DatetimeIndex
    for day in unique_days:
        full_columns = full_columns.append(pd.date_range(day, periods=24, freq='H'))
    df_full = df.reindex(columns=full_columns, fill_value=0)
    df_full.reset_index(inplace=True)
    return df_full
    

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
    if not os.path.exists('PV_allocation_results/'+buffer_distance):
        os.makedirs('PV_allocation_results/'+buffer_distance)
    # create folder for each MV grid
    if not os.path.exists('PV_allocation_results/'+buffer_distance+'/' + str(id)):
        os.makedirs('PV_allocation_results/'+buffer_distance+'/' + str(id))
    # save data2 and data3 as csv files, named by MV grid id+demand or std
    data2.to_pickle('PV_allocation_results/'+buffer_distance+'/' + str(id) + '/' + str(id) + '_demand.pkl')
    data3.to_pickle('PV_allocation_results/'+buffer_distance+'/' + str(id) + '/' + str(id) + '_std.pkl')
    
if mode == 'avg':
    PV_time_series = calculate_daily_average(PV_time_series)
    PV_std = calculate_daily_average(PV_std)
    df_time_series = mapping(PV_building, PV_time_series)
    df_std = mapping(PV_building, PV_std)
elif mode == 'full':
    PV_time_series = fill_missing_hour(PV_time_series)
    PV_std = fill_missing_hour(PV_std)
    df_time_series = mapping(PV_building, PV_time_series)
    df_std = mapping(PV_building, PV_std)

MV_grids_ids = PV_building['MV_grid'].unique()
MV_grids_ids = MV_grids_ids[MV_grids_ids != '-1']
# rearrange the order of MV grid ids to start with the smallest id
MV_grids_ids = np.sort(MV_grids_ids)
for id in tqdm.tqdm(MV_grids_ids):
    allocation(PV_building, df_time_series, df_std, id, buffer_distance)
unmapped_rate = calculate_unmapped(df_time_series)
print('The energy unmapped rate is: ' + str(unmapped_rate))
unmapped_rate_building, mapped_number = calculate_unmapped_building()
print('The building unmapped rate is: ' + str(unmapped_rate_building)+', and the number of mapped buildings is: '+str(mapped_number))

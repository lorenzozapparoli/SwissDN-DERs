"""
Author: Alfredo Oneto
Institution: ETH Zurich
Date: 15/03/2025

Introduction:
This script, `EV_allocator.py`, is designed to allocate Electric Vehicle (EV) charging demand to Low Voltage (LV) grids. It processes EV charging data, maps it to municipalities using BFS codes, and generates profiles for hourly charging power, flexible energy, and power bounds. The script also calculates the share of EV demand for each LV grid node and saves the results for the years 2030, 2040, and 2050.

The script uses penetration rates to scale the EV demand for future years and ensures that the allocation is saved in a structured format for further analysis or simulations.

Usage:
1. Ensure the required input files (EV data, municipality data, and grid data) are available in the specified directories.
2. Run the script to generate EV demand profiles and allocations for LV grids.
3. The output files will be saved in the `EV_output` directory.

Dependencies:
- pandas
- geopandas
- tqdm
- json
- os
"""

import os
import pandas as pd
import geopandas as gpd
import json
import tqdm


def map_BFS(data, zones):
    """
    This function maps the BFS numbers to the input data, which is a dataframe with the first column containing the names of the zones.
    data: DataFrame for the EV data (CP, FE, PD, PU)
    zones: DataFrame for the zones, which contains the BFS numbers and names.
    The function returns a DataFrame with the BFS numbers mapped to the input data.
    """
    df = data.copy()
    # Maps the 'BFS_NUMMER' using the 'NAME' column of the zones DataFrame.
    df['BFS_NUMMER'] = zones.set_index('NAME').loc[df[0].values].BFS_NUMMER.values
    # The name column, which is column 0, is removed
    df = df.drop(columns=[0])
    # The columns of the dataframe are reodered, such that the BFS_NUMMER column is the first column, followed by the rest of the columns.
    # The following columns are sorted in ascending order according to the time they represent.
    df = df[['BFS_NUMMER'] + [col for col in df.columns if col != 'BFS_NUMMER']]
    # The BFS_NUMMER column is renamed to BFS_municipality_code and converted to integer type.
    df.rename(columns={'BFS_NUMMER': 'BFS_municipality_code'}, inplace=True)
    df['BFS_municipality_code'] = df['BFS_municipality_code'].astype('int')
    # The dataframe is sorted by BFS_municipality_code number in ascending order and the index is reset.
    df = df.sort_values('BFS_municipality_code')
    df.reset_index(drop=True, inplace=True)
    # Sanity check for NaN values in the BFS_municipality_code column.
    if df['BFS_municipality_code'].isna().sum() > 0:
        print('There are NaN values in the BFS_municipality_code column')
    return df


def generate_profiles(mapped_CP, mapped_FE, mapped_PD, mapped_PU, penetration, save_path):
    """
    This function generates the profiles for the mapped data and saves them in the output folder.
    mapped_CP: DataFrame for the CP data
    mapped_FE: DataFrame for the FE data
    mapped_PD: DataFrame for the PD data
    mapped_PU: DataFrame for the PU data
    penetration: Dictionary with the penetration rates for each year
    save_path: Path to the output folder
    """
    # we generate one file with mapped_PD, mapped_PU, and mapped_CP
    # As all the three have the same BFS numbers and columns, we concatenate them and assign a new column name with "Upper", "Lower", and "Base". 
    # "Upper" corresponds to PU, "Lower" corresponds to PD, and "Base" corresponds to CP.
    mapped_PU.insert(1, 'Type', 'Upper')
    mapped_PD.insert(1, 'Type', 'Lower')
    mapped_CP.insert(1, 'Type', 'Base')
    Power_mapped = pd.concat([mapped_PU, mapped_PD, mapped_CP])
    Power_mapped.sort_values('BFS_municipality_code', inplace=True)
    Power_mapped.reset_index(drop=True, inplace=True)

    # The profiles are shifted to start on a Monday at 00:00
    # The first 48 values of the last week of the year are added to the dataframe
    Power_two_days_last_week = Power_mapped[list(range(8760 - 168, 8760 - 168 + 48))].copy()
    # We rename the columns to [8761, 8762, ..., 8812] and add them to the dataframe
    Power_two_days_last_week.rename(columns={i: i + 169 for i in range(8760 - 168, 8760 - 168 + 48)}, inplace=True)
    # Power_two_days_last_week is added to Power_mapped
    Power_mapped = pd.concat([Power_mapped, Power_two_days_last_week], axis=1)
    # The columns [1, 2, ..., 48] are dropped from the dataframe
    Power_mapped = Power_mapped.drop(list(range(1, 49)), axis=1)
    # The columns are renamed to [1, 2, ..., 8760]
    Power_mapped.rename(columns={i: i - 48 for i in range(49, 8761 + 48)}, inplace=True)
    # The flexible energy profiles are shifted to start on a Monday at 00:00
    # The first two days of the last week of the year are added to the dataframe
    Fe_two_days_last_week = mapped_FE[list(range(365 - 7, 365 - 7 + 2))].copy()
    # We rename the columns to [366, 367] and add them to the dataframe
    Fe_two_days_last_week.rename(columns={i: i + 8 for i in range(365 - 7, 365 - 7 + 2)}, inplace=True)
    # Fe_two_days_last_week is added to mapped_FE
    mapped_FE = pd.concat([mapped_FE, Fe_two_days_last_week], axis=1)
    # The columns [1,2] are dropped from the dataframe
    mapped_FE = mapped_FE.drop(list(range(1, 3)), axis=1)
    # The columns are renamed to [1, 2, ..., 365]
    mapped_FE.rename(columns={i: i - 2 for i in range(3, 365 + 3)}, inplace=True)

    # The profiles are adjusted according to the penetration rates for each year.
    Power_by_year = {year: Power_mapped.copy() for year in penetration.keys()}
    for year, penetration_rate in penetration.items():
        print(year)
        Power_by_year[year][list(range(1,8761))] = Power_by_year[year][list(range(1,8761))].multiply(penetration_rate)
        Power_by_year[year][list(range(1,8761))] = Power_by_year[year][list(range(1,8761))].round(2)
        Power_by_year[year].to_csv(os.path.join(save_path, f'{year}/EV_power_profiles_LV.csv'), index=False)

    FE_by_year = {year: mapped_FE.copy() for year in penetration.keys()}
    for year, penetration_rate in penetration.items():
        print(year)
        FE_by_year[year][list(range(1,366))] = FE_by_year[year][list(range(1,366))].multiply(penetration_rate)
        FE_by_year[year][list(range(1,366))] = FE_by_year[year][list(range(1,366))].round(2)
        FE_by_year[year].to_csv(os.path.join(save_path, f'{year}/EV_flexible_energy_profiles_LV.csv'), index=False)
    return


def concat_all_grids(grid_ids, path):
    """
    This function concatenates all the grids in the given path and returns a dataframe with the grid name, node name, and demand.
    grid_ids: List of grid ids
    path: Path to the folder containing the grids
    The function returns a dataframe with the grid name, node name, and demand.
    """
    node_total_list = []
    for n in range(len(grid_ids)):
        node_id = grid_ids[n]+"_nodes"
        node = gpd.read_file(path+node_id)
        if 'el_dmd' not in node.columns:
            node['el_dmd'] = 0
        dmd = node['el_dmd']
        if 'osmid' not in node.columns:
            continue    
        node_name = node['osmid']
        node_total_list.append(pd.DataFrame({'grid_name':grid_ids[n], 'node_name':node_name, 'dmd':dmd}))
    if len(node_total_list) == 0:
        return pd.DataFrame()
    node_total = pd.concat(node_total_list, ignore_index=True)
    node_total['dmd'] = node_total['dmd'].astype(float)
    node_total['EV_share'] = node_total['dmd']/node_total['dmd'].sum()
    return node_total


def get_allocation(path_base, path_grid, processing_dictionary):
    """
    This function generates the allocation for EV consumption in the LV grids.
    path_base: Path to the base folder
    processing_dictionary: Dictionary with the processing information for the grids
    The function saves the allocation in the output folder.
    """
    keys = list(processing_dictionary.keys())
    LV_data_path = os.path.join(path_grid, 'LV/')
    save_path = os.path.join(path_base, 'EV_output')

    node_profiles_list = []
    for key in tqdm.tqdm(keys):
        path = LV_data_path+processing_dictionary[key]+'/'
        grid_ids = list(set([f[:-6] for f in os.listdir(path) if f.startswith(key+'-')]))
        node_total = concat_all_grids(grid_ids,path)
        if node_total.empty:
            print('Empty grid:', key)
            continue

        node_total['BFS_municipality_code'] = int(key)
        node_profiles_list.append(node_total)

    node_profiles = pd.concat(node_profiles_list, ignore_index=True)
    node_profiles['LV_grid'] = node_profiles['grid_name'].astype(str)
    node_profiles['LV_osmid'] = node_profiles['node_name'].astype(str)
    node_profiles.sort_values(['BFS_municipality_code','grid_name','node_name'], ascending=True, inplace=True)
    node_profiles_reduced = node_profiles[['LV_grid', 'LV_osmid', 'EV_share']].copy()
    node_profiles_reduced.to_csv(os.path.join(save_path, '2030/EV_allocation_LV.csv'), index=False)
    node_profiles_reduced.to_csv(os.path.join(save_path, '2040/EV_allocation_LV.csv'), index=False)
    node_profiles_reduced.to_csv(os.path.join(save_path, '2050/EV_allocation_LV.csv'), index=False)
    return


if __name__ == '__main__':

    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, 'EV_input')

    # read the municipalities
    zones = gpd.read_file(os.path.join(data_path, 'Municipalities_2022_01_crs2056.geojson'))
    zones = zones[(zones['ICC'] == 'CH') & (zones['EINWOHNERZ'] > 0)].copy()

    # read the EV data
    CP = pd.read_csv(os.path.join(data_path, 'Municipality_hourly_charging_power_2050.csv'), skiprows=1, header=None)
    FE = pd.read_csv(os.path.join(data_path, 'Municipality_daily_flexible_charging_energy_2050.csv'), skiprows=1, header=None)
    PD = pd.read_csv(os.path.join(data_path, 'Municipality_hourly_lower_charging_power_bound_2050.csv'), skiprows=1, header=None)
    PU = pd.read_csv(os.path.join(data_path, 'Municipality_hourly_upper_charging_power_bound_2050.csv'), skiprows=1, header=None)

    # The BFS numbers are mapped to the input data, to assign the municipalities to the correct zones.
    CP_mapped = map_BFS(CP, zones)
    FE_mapped = map_BFS(FE, zones)
    PD_mapped = map_BFS(PD, zones)
    PU_mapped = map_BFS(PU, zones)

    penetration_rates = {'2030': 0.15, '2040': 0.6, '2050': 1.0}
    base_save_path = os.path.join(base_path, 'EV_output')

    # we generate the profiles for the mapped data and save them in the output folder.
    generate_profiles(CP_mapped, FE_mapped, PD_mapped, PU_mapped, penetration_rates, base_save_path)

    grids_path = os.path.join(os.path.dirname(base_path), 'Grids')
    with open(os.path.join(grids_path, 'Additional_files', 'dict_folder.json')) as f:
        dictionary_processing = json.load(f)
    # we generate the allocation for EV consumption in the LV grids
    get_allocation(base_path, grids_path, dictionary_processing)

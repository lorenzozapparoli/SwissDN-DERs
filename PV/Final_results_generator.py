"""
Author: Lorenzo Zapparoli
Institution: ETH Zurich
Date: 15/03/2025

Introduction:
This script, `Final_results_generator.py`, processes photovoltaic (PV) data to allocate PV systems to Low Voltage (LV) and Medium Voltage (MV) grids for the years 2030, 2040, and 2050. The allocation is based on predefined parameters such as PV penetration, nodal power limits, and building efficiency factors for each year. The script ensures that the allocation adheres to nodal power constraints and distinguishes between LV and MV grids.

The script reads PV data, filters it based on specific criteria, and performs sampling to allocate PV systems. It processes LV and MV grids separately, grouping data by grid nodes and saving the results for each simulation year.

Usage:
1. Ensure the required input files (PV data and allocation files) are available in the `PV_input/PV_data` directory.
2. Run the script to generate PV allocation results for LV and MV grids for 2030, 2040, and 2050.
3. The output files will be saved in the `PV_output` directory under subfolders for each simulation year.

Dependencies:
- pandas
- numpy
- os
- warnings
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

script_path = os.path.dirname(os.path.abspath(__file__))
generation_ts = pd.read_csv(os.path.join(script_path, 'PV_input', 'PV_data', 'rooftop_PV_CH_EPV_W_by_building.csv'))
deviation_ts = pd.read_csv(
    os.path.join(script_path, 'PV_input', 'PV_data', 'rooftop_PV_CH_EPV_W_std_by_building.csv'))
irradiance_ts = pd.read_csv(
    os.path.join(script_path, 'PV_input', 'PV_data', 'rooftop_PV_CH_Gt_W_m2_by_building.csv'))
# Scenarios data for 2030, 2040 and 2050 (https://nexus-e.org/eip-eth-collaboration-rethink-future-swiss-electricity-supply/)
parameters_dict = {2030: {'PV_penetration': 9.41/34.64},
                   2040: {'PV_penetration': 15.69/34.64},
                   2050: {'PV_penetration': 34.64/34.64}}


def data_aggregate_LV(building_data_part):
    """
    Aggregates building data for LV grids.

    Args:
        building_data_part (DataFrame): DataFrame containing building data.

    Returns:
        DataFrame: Aggregated building data grouped by LV grid and osmid.

    Description:
    - Drops unnecessary columns and groups data by LV grid and osmid.
    - Sums numerical columns for aggregation.
    """

    building_copy = building_data_part.copy()
    building_copy = building_copy.drop(columns=['geometry', 'SB_UUID'])

    # Aggregate other building data by summing numerical columns
    building_data_0 = (
        building_copy.groupby(['LV_grid', 'LV_osmid'], as_index=False).sum())
    # Map the most frequent T_PROFILE back to the grouped data
    building_data_0 = building_data_0.set_index(['LV_grid', 'LV_osmid'])
    building_data_0 = building_data_0.reset_index()
    return building_data_0


def data_aggregate_MV(building_data_part):
    """
    Aggregates building data for MV grids.

    Args:
        building_data_part (DataFrame): DataFrame containing building data.

    Returns:
        DataFrame: Aggregated building data grouped by MV grid and osmid.

    Description:
    - Drops unnecessary columns and groups data by MV grid and osmid.
    - Sums numerical columns for aggregation.
    """

    building_copy = building_data_part.copy()
    building_copy = building_copy.drop(columns=['geometry', 'SB_UUID'])

    # Aggregate other building data by summing numerical columns
    building_data_0 = (
        building_copy.groupby(['MV_grid', 'MV_osmid'], as_index=False).sum())
    # Map the most frequent T_PROFILE back to the grouped data
    building_data_0 = building_data_0.set_index(['MV_grid', 'MV_osmid'])
    building_data_0 = building_data_0.reset_index()
    return building_data_0


def fill_missing_hour(data):
    """
    Fills missing hourly data in the time series.

    Args:
        data (DataFrame): DataFrame containing time series data.

    Returns:
        DataFrame: DataFrame with missing hours filled with zeros.

    Description:
    - Ensures all hours are present in the time series.
    - Fills missing hours with zero values.
    """
    df = data.copy()
    df.set_index('SB_UUID', inplace=True)
    df.columns = pd.to_datetime(df.columns)
    unique_days = df.columns.normalize().unique()
    full_columns = pd.DatetimeIndex([])
    for day in unique_days:
        full_columns = full_columns.append(pd.date_range(day, periods=24, freq='H'))
    df_full = df.reindex(columns=full_columns, fill_value=0)
    df_full.reset_index(inplace=True)
    return df_full


def mapping(PV_building, df, type):
    """
    Maps grid and osmid information to the data.

    Args:
        PV_building (DataFrame): DataFrame containing PV building data.
        df (DataFrame): DataFrame to map grid and osmid information.
        type (str): Type of grid ('LV' or 'MV').

    Returns:
        DataFrame: Updated DataFrame with grid and osmid mappings.

    Description:
    - Maps grid and osmid columns from PV building data to the input DataFrame.
    - Filters out rows with missing grid information.
    """

    grid_dict = dict(zip(PV_building['SB_UUID'], PV_building[f'{type}_grid']))
    osmid_dict = dict(zip(PV_building['SB_UUID'], PV_building[f'{type}_osmid']))
    df[f'{type}_grid'] = df['SB_UUID'].map(grid_dict)
    df[f'{type}_osmid'] = df['SB_UUID'].map(osmid_dict)
    df = df[df[f'{type}_grid'].notna()]
    return df


def area_calculation(id, PV_generation_data, irradiance_data, type):
    """
    Calculates the installed PV area and power.

    Args:
        id (str): Identifier for the data.
        PV_generation_data (DataFrame): DataFrame containing PV generation data.
        irradiance_data (DataFrame): DataFrame containing irradiance data.
        type (str): Type of grid ('LV' or 'MV').

    Returns:
        DataFrame: DataFrame with calculated installed power (kWp).

    Description:
    - Calculates the PV area based on generation and irradiance data.
    - Computes the installed power in kWp.
    """

    osmid = PV_generation_data[f'{type}_osmid']
    grid = PV_generation_data[f'{type}_grid']
    P_data = pd.DataFrame(columns=[f'{type}_grid', f'{type}_osmid', 'P_installed (kWp)'])
    PV_generation_data = PV_generation_data.loc[:, (PV_generation_data != 0).any(axis=0)]
    irradiance_data = irradiance_data.loc[:, (irradiance_data != 0).any(axis=0)]
    area = PV_generation_data.iloc[:, 2:] / (irradiance_data.iloc[:, 2:] * 0.138)
    area = area.replace(0, np.nan)
    area = area.fillna(area.mean(axis=1))
    area['mean'] = area.mean(axis=1)
    P_installed = 285 * area['mean'] / (1.640 * 0.992 * 10 ** 3)  # kWp
    P_data[f'{type}_grid'] = PV_generation_data[f'{type}_grid']
    try:
        P_data[f'{type}_osmid'] = PV_generation_data[f'{type}_osmid']
    except:
        P_data[f'{type}_osmid'] = osmid
    try:
        P_data[f'{type}_grid'] = PV_generation_data[f'{type}_grid']
    except:
        P_data[f'{type}_grid'] = grid
    P_data['P_installed (kWp)'] = P_installed
    return P_data


def allocation(PV_building, df_time_series_avg, df_std_avg, df_ir, type):
    """
    Allocates PV generation and standard deviation to grids.

    Args:
        PV_building (DataFrame): DataFrame containing PV building data.
        df_time_series_avg (DataFrame): Average time series data.
        df_std_avg (DataFrame): Standard deviation data.
        df_ir (DataFrame): Irradiance data.
        type (str): Type of grid ('LV' or 'MV').

    Returns:
        Tuple: DataFrames for generation, standard deviation, and installed power.

    Description:
    - Aggregates time series, standard deviation, and irradiance data by grid.
    - Calculates installed power for each grid.
    """

    pv_time_series_part = df_time_series_avg
    pv_std_part = df_std_avg
    pv_ir_part = df_ir

    pv_time_series_part = pv_time_series_part.reset_index(drop=True)
    data2 = pv_time_series_part.drop('SB_UUID', axis=1).groupby([f'{type}_grid', f'{type}_osmid'], dropna=False).sum().reset_index(inplace=False)

    pv_std_part = pv_std_part.reset_index(drop=True)
    data3 = pv_std_part.drop('SB_UUID', axis=1).groupby([f'{type}_grid', f'{type}_osmid'], dropna=False).sum().reset_index(inplace=False)

    pv_ir_part = pv_ir_part.reset_index(drop=True)
    data4 = pv_ir_part.drop('SB_UUID', axis=1).groupby([f'{type}_grid', f'{type}_osmid'], dropna=False).max().reset_index(inplace=False)

    # if f['{type}_osmid'] not in the first column of data2, data4, put them in the first column
    columns_order = [f'{type}_grid', f'{type}_osmid'] + [col for col in data2.columns if col not in [f'{type}_grid', f'{type}_osmid']]
    data2 = data2[columns_order]
    columns_order = [f'{type}_grid', f'{type}_osmid'] + [col for col in data3.columns if col not in [f'{type}_grid', f'{type}_osmid']]
    data3 = data3[columns_order]
    columns_order = [f'{type}_grid', f'{type}_osmid'] + [col for col in data2.columns if col not in [f'{type}_grid', f'{type}_osmid']]
    data3 = data3[columns_order]

    P_data = area_calculation(id, data2, data4, type)
    return data2, data3, P_data


def process_LV_allocation(previous_year_data=None, simulation_year=2030):
    """
    Processes PV allocation for LV grids.

    Args:
        previous_year_data (DataFrame, optional): Data from the previous year.
        simulation_year (int): Year for the simulation (2030, 2040, or 2050).

    Returns:
        DataFrame: Sampled LV allocation data for the simulation year.

    Description:
    - Filters and samples LV allocation data based on PV penetration.
    - Maps time series, standard deviation, and irradiance data to grids.
    - Saves results for generation, standard deviation, and installed power.
    """

    print('computing allocation for LV grids for year:', simulation_year)

    PV_penetration = parameters_dict[simulation_year]['PV_penetration']
    max_nodal_power = 100  # kW
    LV_allocation = pd.read_csv(os.path.join(script_path, 'PV_input', 'PV_data', 'PV_allocation_LV.csv'))
    EPV_threshold = LV_allocation['EPV_kWh_a'].quantile(0.95)
    LV_allocation = LV_allocation[LV_allocation['EPV_kWh_a'] <= EPV_threshold]
    LV_allocation = LV_allocation[LV_allocation['LV_grid'] != '-1']

    if previous_year_data is not None:
        LV_previous = previous_year_data
        LV_sampled = LV_previous.copy()

        LV_remaining = LV_allocation[~LV_allocation['SB_UUID'].isin(LV_previous['SB_UUID'])]
        LV_additional = LV_remaining.sample(frac=((PV_penetration - len(LV_previous) / len(LV_allocation)) / (1 - len(LV_previous) / len(LV_allocation))), random_state=1)
        print('LV_previous_share:', len(LV_previous) / len(LV_allocation))
        LV_sampled = pd.concat([LV_sampled, LV_additional])
        print(len(LV_sampled) / len(LV_allocation))
    else:
        LV_sampled = LV_allocation.sample(frac=PV_penetration, random_state=1)

    PV_time_series = fill_missing_hour(generation_ts.copy())
    PV_std = fill_missing_hour(deviation_ts.copy())
    PV_time_series_area = fill_missing_hour(irradiance_ts.copy())

    df_time_series = mapping(LV_sampled, PV_time_series, 'LV')
    df_std = mapping(LV_sampled, PV_std, 'LV')
    df_time_series_area = mapping(LV_sampled, PV_time_series_area, 'LV')

    Generation_data, Std_data, P_installed_data = allocation(LV_sampled, df_time_series, df_std, df_time_series_area, 'LV')

    valid_nodes = P_installed_data[P_installed_data['P_installed (kWp)'] <= max_nodal_power]
    valid_nodes_set = set(valid_nodes[['LV_grid', 'LV_osmid']].apply(tuple, axis=1))

    Generation_data = Generation_data[Generation_data[['LV_grid', 'LV_osmid']].apply(tuple, axis=1).isin(valid_nodes_set)]
    Std_data = Std_data[Std_data[['LV_grid', 'LV_osmid']].apply(tuple, axis=1).isin(valid_nodes_set)]
    P_installed_data = P_installed_data[P_installed_data[['LV_grid', 'LV_osmid']].apply(tuple, axis=1).isin(valid_nodes_set)]

    Generation_data.to_csv(os.path.join(script_path, 'PV_output', str(simulation_year), 'LV_generation.csv'), index=False)
    Std_data.to_csv(os.path.join(script_path, 'PV_output', str(simulation_year), 'LV_std.csv'), index=False)
    P_installed_data.to_csv(os.path.join(script_path, 'PV_output', str(simulation_year), 'LV_P_installed.csv'), index=False)

    return LV_sampled


def process_MV_allocation(previous_year_data=None, simulation_year=2030):
    """
    Processes PV allocation for MV grids.

    Args:
        previous_year_data (DataFrame, optional): Data from the previous year.
        simulation_year (int): Year for the simulation (2030, 2040, or 2050).

    Returns:
        DataFrame: Sampled MV allocation data for the simulation year.

    Description:
    - Filters and samples MV allocation data based on PV penetration.
    - Maps time series, standard deviation, and irradiance data to grids.
    - Saves results for generation, standard deviation, and installed power.
    """

    print('computing allocation for MV grids for year:', simulation_year)

    PV_penetration = parameters_dict[simulation_year]['PV_penetration']
    max_nodal_power = 1000  # kW
    MV_allocation = pd.read_csv(os.path.join(script_path, 'PV_input', 'PV_data', 'PV_allocation_MV.csv'))
    EPV_threshold = MV_allocation['EPV_kWh_a'].quantile(0.95)
    MV_allocation = MV_allocation[MV_allocation['EPV_kWh_a'] <= EPV_threshold]
    MV_allocation = MV_allocation[MV_allocation['MV_grid'] != '-1']

    if previous_year_data is not None:
        MV_previous = previous_year_data
        MV_sampled = MV_previous.copy()

        MV_remaining = MV_allocation[~MV_allocation['SB_UUID'].isin(MV_previous['SB_UUID'])]
        MV_additional = MV_remaining.sample(frac=((PV_penetration - len(MV_previous) / len(MV_allocation)) / (1 - len(MV_previous) / len(MV_allocation))), random_state=1)
        MV_sampled = pd.concat([MV_sampled, MV_additional])
    else:
        MV_sampled = MV_allocation.sample(frac=PV_penetration, random_state=1)

    PV_time_series = fill_missing_hour(generation_ts.copy())
    PV_std = fill_missing_hour(deviation_ts.copy())
    PV_time_series_area = fill_missing_hour(irradiance_ts.copy())

    df_time_series = mapping(MV_sampled, PV_time_series, 'MV')
    df_std = mapping(MV_sampled, PV_std, 'MV')
    df_time_series_area = mapping(MV_sampled, PV_time_series_area, 'MV')

    Generation_data, Std_data, P_installed_data = allocation(MV_sampled, df_time_series, df_std, df_time_series_area, 'MV')

    valid_nodes = P_installed_data[P_installed_data['P_installed (kWp)'] <= max_nodal_power]
    valid_nodes_set = set(valid_nodes[['MV_grid', 'MV_osmid']].apply(tuple, axis=1))

    Generation_data = Generation_data[Generation_data[['MV_grid', 'MV_osmid']].apply(tuple, axis=1).isin(valid_nodes_set)]
    Std_data = Std_data[Std_data[['MV_grid', 'MV_osmid']].apply(tuple, axis=1).isin(valid_nodes_set)]
    P_installed_data = P_installed_data[P_installed_data[['MV_grid', 'MV_osmid']].apply(tuple, axis=1).isin(valid_nodes_set)]

    Generation_data.to_csv(os.path.join(script_path, 'PV_output', str(simulation_year), 'MV_generation.csv'), index=False)
    Std_data.to_csv(os.path.join(script_path, 'PV_output', str(simulation_year), 'MV_std.csv'), index=False)
    P_installed_data.to_csv(os.path.join(script_path, 'PV_output', str(simulation_year), 'MV_P_installed.csv'), index=False)

    return MV_sampled


if __name__ == '__main__':
    """
    Main execution block.

    Description:
    - Processes PV allocation for LV and MV grids for 2030, 2040, and 2050.
    - Saves results for each simulation year in the `PV_output` directory.
    """

    data_2030_LV = process_LV_allocation(simulation_year=2030)
    data_2040_LV = process_LV_allocation(previous_year_data=data_2030_LV, simulation_year=2040)
    data_2050_LV = process_LV_allocation(previous_year_data=data_2040_LV, simulation_year=2050)

    data_2030_MV = process_MV_allocation(simulation_year=2030)
    data_2040_MV = process_MV_allocation(previous_year_data=data_2030_MV, simulation_year=2040)
    data_2050_MV = process_MV_allocation(previous_year_data=data_2040_MV, simulation_year=2050)

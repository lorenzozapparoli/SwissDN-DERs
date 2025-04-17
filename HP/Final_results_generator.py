"""
Author: Lorenzo Zapparoli
Institution: ETH Zurich
Date: 15/03/2025

Introduction:
This script, `Final_results_generator.py`, processes building data to allocate heat pumps (HP) to Low Voltage (LV) and Medium Voltage (MV) grids for the years 2030, 2040, and 2050. The allocation is based on predefined parameters such as HP share, Coefficient of Performance (COP), and building efficiency factors for each year. The script ensures that the allocation adheres to nodal power limits and distinguishes between residential and commercial buildings.

The script reads building data, filters it based on specific criteria, and performs sampling to allocate heat pumps. It processes LV and MV grids separately, grouping data by grid nodes and saving the results for each simulation year.

Usage:
1. Ensure the required input files (building data and allocation files) are available in the `HP_input/Buildings_data` directory.
2. Run the script to generate heat pump allocation results for LV and MV grids for 2030, 2040, and 2050.
3. The output files will be saved in the `HP_output` directory under subfolders for each simulation year.

Dependencies:
- pandas
- numpy
- os
- warnings
"""

import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings("ignore")

# Read the CSV file
script_path = os.path.dirname(os.path.abspath(__file__))
buildings_info = pd.read_csv(os.path.join(script_path, 'HP_input', 'Buildings_data', 'Buildings data Switzerland', 'gebaeude_batiment_edificio.csv'),
                             sep="\t", low_memory=False)
parameters_dict = {2030: {'HP_share_commercial': 0.1394, 'HP_share_residential': 0.360, 'COP': 3.49, 'Building_efficiency_factor': 0.904066},
                   2040: {'HP_share_commercial': 0.2129, 'HP_share_residential': 0.520, 'COP': 3.79, 'Building_efficiency_factor': 0.784556},
                   2050: {'HP_share_commercial': 0.2830, 'HP_share_residential': 0.650, 'COP': 4.12, 'Building_efficiency_factor': 0.704088}}


def process_lv_allocation(previous_year_data=None, simulation_year=2030):
    print('computing allocation for LV grids for year:', simulation_year)

    HP_share_commercial = parameters_dict[simulation_year]['HP_share_commercial']
    HP_share_residential = parameters_dict[simulation_year]['HP_share_residential']
    HP_COP = parameters_dict[simulation_year]['COP']
    HP_building_efficiency_factor = parameters_dict[simulation_year]['Building_efficiency_factor']
    max_nodal_power = 100  # kW
    # Read the CSV file
    builidngs_df = pd.read_csv(os.path.join(script_path, 'HP_input', 'Buildings_data', 'Buildings_data.csv'))
    df = pd.read_csv(os.path.join(script_path, 'HP_input', 'Buildings_data', 'Building_allocation_LV.csv'))
    # Remove entries with PRT above the 95th percentile
    prt_threshold = builidngs_df['PRT'].quantile(0.95)
    df_filtered = df[df['PRT'] <= prt_threshold]
    df_filtered['COP'] = HP_COP

    # Drop entries with LV_grid or LV_osmid equal to -1
    df_filtered = df_filtered[(df_filtered['LV_grid'] != '-1') & (df_filtered['LV_osmid'] != -1)]

    # Add GKLAS column to df_filtered by matching EGID
    df_filtered = df_filtered.merge(buildings_info[['EGID', 'GKLAS']], on='EGID', how='left')

    # Classify buildings as residential or commercial
    residential_gklas = [11, 111, 1121, 1122, 1130, 1211, 1212, 1110]
    df_filtered['Building_Type'] = np.where(df_filtered['GKLAS'].isin(residential_gklas), 'Residential', 'Commercial')

    # Sample the commercial and residential buildings
    if previous_year_data is not None:
        LV_previous = previous_year_data
        LV_sampled = LV_previous.copy()

        LV_remaining = df_filtered[~df_filtered['EGID'].isin(LV_previous['EGID'])]
        commercial_buildings = LV_remaining[LV_remaining['Building_Type'] == 'Commercial']
        residential_buildings = LV_remaining[LV_remaining['Building_Type'] == 'Residential']
        residential_share_old = len(LV_previous[LV_previous['Building_Type'] == 'Residential']) / len(
            df_filtered[df_filtered['Building_Type'] == 'Residential'])
        commercial_share_old = len(LV_previous[LV_previous['Building_Type'] == 'Commercial']) / len(
            df_filtered[df_filtered['Building_Type'] == 'Commercial'])
        print('Residential: ', residential_share_old, 'Commercial: ', commercial_share_old)
        # residential_share_old = len(LV_previous[LV_previous['Building_Type'] == 'Residential']) / len(LV_previous)
        # commercial_share_old = len(LV_previous[LV_previous['Building_Type'] == 'Commercial']) / len(LV_previous)
        additional_commercial = commercial_buildings.sample(
            frac=((HP_share_commercial - commercial_share_old) / (1 - commercial_share_old)), random_state=1)
        additional_residential = residential_buildings.sample(
            frac=((HP_share_residential - residential_share_old) / (1 - residential_share_old)), random_state=1)
        df_filtered = pd.concat([LV_sampled, additional_commercial, additional_residential])
    else:
        commercial_buildings = df_filtered[df_filtered['Building_Type'] == 'Commercial']
        residential_buildings = df_filtered[df_filtered['Building_Type'] == 'Residential']
        sampled_commercial = commercial_buildings.sample(frac=HP_share_commercial, random_state=1)
        sampled_residential = residential_buildings.sample(frac=HP_share_residential, random_state=1)
        df_filtered = pd.concat([sampled_commercial, sampled_residential])

    # Drop unnecessary columns
    df_previous = df_filtered.copy()
    df_filtered = df_filtered.drop(columns=['EGID', 'GKODN', 'GKODE', 'GEBF', 'GAREA', 'ISHP', 'geometry', 'distance',
                                            'GKLAS', 'Building_Type'])

    # Convert PRT and HBLD to kW
    df_filtered['PRT'] = df_filtered['PRT'] / 10 ** 3 * HP_building_efficiency_factor  # Convert PRT to kW
    df_filtered['HBLD'] = df_filtered['HBLD'] / 10 ** 3 * HP_building_efficiency_factor  # Convert HBLD to kW/K

    # Rename columns for clarity
    df_filtered.rename(columns={
        'PRT': 'PRT_kW',
        'CBLD': 'CBLD_KWh/K',
        'HBLD': 'HBLD_kW/K'
    }, inplace=True)

    # Reorder columns to make LV_grid and LV_osmid the first two columns
    columns_order = ['LV_grid', 'LV_osmid', 'CBLD_KWh/K', 'HBLD_kW/K', 'PRT_kW', 'COP', 'T_PROFILE']
    df_filtered = df_filtered[columns_order]

    # Group by LV_grid and LV_osmid, summing the relevant columns and taking the most frequent T_PROFILE
    df_grouped = df_filtered.groupby(['LV_grid', 'LV_osmid']).agg({
        'PRT_kW': 'sum',
        'CBLD_KWh/K': 'sum',
        'HBLD_kW/K': 'sum',
        'COP': 'mean',
        'T_PROFILE': lambda x: x.mode()[0]
    }).reset_index()

    df_grouped = df_grouped[df_grouped['PRT_kW'] <= max_nodal_power]

    # Save the processed dataframes to CSV files
    df_grouped.to_csv(os.path.join(script_path, 'HP_output', str(simulation_year), 'LV_heat_pump_allocation.csv'), index=False)

    print("LV Data has been processed and saved.")
    return df_previous


def process_mv_allocation(previous_year_data=None, simulation_year=2030):
    print('computing allocation for MV grids for year:', simulation_year)

    HP_share_commercial = parameters_dict[simulation_year]['HP_share_commercial']
    HP_share_residential = parameters_dict[simulation_year]['HP_share_residential']
    HP_COP = parameters_dict[simulation_year]['COP']
    HP_building_efficiency_factor = parameters_dict[simulation_year]['Building_efficiency_factor']
    max_nodal_power = 1000  # kW
    df = pd.read_csv(os.path.join(script_path, 'HP_input', 'Buildings_data', 'Building_allocation_MV.csv'))

    # Remove entries with PRT above the 95th percentile
    prt_threshold = df['PRT'].quantile(0.95)
    df_filtered = df[df['PRT'] <= prt_threshold]

    # Drop entries with MV_grid or MV_osmid equal to -1
    df_filtered = df_filtered[(df_filtered['MV_grid'] != '-1') & (df_filtered['MV_osmid'] != -1)]

    # Add GKLAS column to df_filtered by matching EGID
    df_filtered = df_filtered.merge(buildings_info[['EGID', 'GKLAS']], on='EGID', how='left')

    # Classify buildings as residential or commercial
    residential_gklas = [11, 111, 1121, 1122, 1130, 1211, 1212, 1110]
    df_filtered['Building_Type'] = np.where(df_filtered['GKLAS'].isin(residential_gklas), 'Residential', 'Commercial')

    # Sample the commercial and residential buildings
    if previous_year_data is not None:
        LV_previous = previous_year_data
        LV_sampled = LV_previous.copy()

        LV_remaining = df_filtered[~df_filtered['EGID'].isin(LV_previous['EGID'])]
        commercial_buildings = LV_remaining[LV_remaining['Building_Type'] == 'Commercial']
        residential_buildings = LV_remaining[LV_remaining['Building_Type'] == 'Residential']
        residential_share_old = len(LV_previous[LV_previous['Building_Type'] == 'Residential']) / len(df_filtered[df_filtered['Building_Type'] == 'Residential'])
        commercial_share_old = len(LV_previous[LV_previous['Building_Type'] == 'Commercial']) / len(df_filtered[df_filtered['Building_Type'] == 'Commercial'])
        print('Residential: ', residential_share_old, 'Commercial: ', commercial_share_old)
        # residential_share_old = len(LV_previous[LV_previous['Building_Type'] == 'Residential']) / len(LV_previous)
        # commercial_share_old = len(LV_previous[LV_previous['Building_Type'] == 'Commercial']) / len(LV_previous)
        additional_commercial = commercial_buildings.sample(frac=((HP_share_commercial - commercial_share_old) / (1 - commercial_share_old)), random_state=1)
        additional_residential = residential_buildings.sample(frac=((HP_share_residential - residential_share_old) / (1 - residential_share_old)), random_state=1)
        df_filtered = pd.concat([LV_sampled, additional_commercial, additional_residential])
    else:
        commercial_buildings = df_filtered[df_filtered['Building_Type'] == 'Commercial']
        residential_buildings = df_filtered[df_filtered['Building_Type'] == 'Residential']
        sampled_commercial = commercial_buildings.sample(frac=HP_share_commercial, random_state=1)
        sampled_residential = residential_buildings.sample(frac=HP_share_residential, random_state=1)
        df_filtered = pd.concat([sampled_commercial, sampled_residential])

    # # Sample the commercial and residential buildings
    # commercial_buildings = df_filtered[df_filtered['Building_Type'] == 'Commercial']
    # residential_buildings = df_filtered[df_filtered['Building_Type'] == 'Residential']
    #
    # sampled_commercial = commercial_buildings.sample(frac=HP_share_commercial, random_state=1)
    # sampled_residential = residential_buildings.sample(frac=HP_share_residential, random_state=1)

    # df_filtered = pd.concat([sampled_commercial, sampled_residential])
    df_previous = df_filtered.copy()
    df_filtered['COP'] = HP_COP

    # Drop unnecessary columns
    df_filtered = df_filtered.drop(columns=['EGID', 'GKODN', 'GKODE', 'GEBF', 'GAREA', 'ISHP', 'geometry', 'GKLAS', 'Building_Type'])

    # Convert PRT and HBLD to kW
    df_filtered['PRT'] = df_filtered['PRT'] / 10 ** 3 * HP_building_efficiency_factor  # Convert PRT to kW
    df_filtered['HBLD'] = df_filtered['HBLD'] / 10 ** 3 * HP_building_efficiency_factor  # Convert HBLD to kW/K

    # Rename columns for clarity
    df_filtered.rename(columns={
        'PRT': 'PRT_kW',
        'CBLD': 'CBLD_KWh/K',
        'HBLD': 'HBLD_kW/K'
    }, inplace=True)

    # Reorder columns to make MV_grid and MV_osmid the first two columns
    columns_order = ['MV_grid', 'MV_osmid', 'CBLD_KWh/K', 'HBLD_kW/K', 'PRT_kW', 'COP', 'T_PROFILE']
    df_filtered = df_filtered[columns_order]

    # Group by MV_grid and MV_osmid, summing the relevant columns and taking the most frequent T_PROFILE
    df_grouped = df_filtered.groupby(['MV_grid', 'MV_osmid']).agg({
        'PRT_kW': 'sum',
        'CBLD_KWh/K': 'sum',
        'HBLD_kW/K': 'sum',
        'COP': 'mean',
        'T_PROFILE': lambda x: x.mode()[0]
    }).reset_index()

    # Save the processed dataframes to CSV files
    df_grouped = df_grouped[df_grouped['PRT_kW'] <= max_nodal_power]
    df_grouped.to_csv(os.path.join(script_path, 'HP_output', str(simulation_year), 'MV_heat_pump_allocation.csv'), index=False)

    print("MV data has been processed and saved.")

    return df_previous


if __name__ == "__main__":
    data_2030_LV = process_lv_allocation(simulation_year=2030)
    data_2040_LV = process_lv_allocation(previous_year_data=data_2030_LV, simulation_year=2040)
    data_2050_LV = process_lv_allocation(previous_year_data=data_2040_LV, simulation_year=2050)

    data_2030_MV = process_mv_allocation(simulation_year=2030)
    data_2040_MV = process_mv_allocation(previous_year_data=data_2030_MV, simulation_year=2040)
    data_2050_MV = process_mv_allocation(previous_year_data=data_2040_MV, simulation_year=2050)

"""
Author: Alfredo Oneto
Institution: ETH Zurich
Date: 15/03/2025

Introduction:
This script, `validity_data_consistency.py`, is designed to validate and ensure the consistency of data related to Distributed Energy Resources (DERs) such as PV systems, heat pumps (HP), battery energy storage systems (BESS), and electric vehicles (EVs). It performs checks on data types, shapes, and values, ensuring that the data adheres to expected formats and constraints. The script also verifies micro-level consistency for energy profiles and flexible energy allocations.

The script processes data for multiple simulation years (2030, 2040, 2050) and validates the relationships between installed capacities, generation, and flexibility profiles. It ensures that the data is free of missing values and adheres to predefined constraints.

Usage:
1. Ensure the required input files are organized in folders corresponding to DERs and simulation years.
2. Run the script to validate the data consistency and integrity.
3. Outputs are printed to the console, highlighting any errors or inconsistencies.

Dependencies:
- pandas
- numpy
- tqdm
- os
"""

import pandas as pd
from pandas.api.types import is_object_dtype
import os
import numpy as np
from tqdm import tqdm

def read_files(file_names):
    # we read the files 
    dfs = {}
    for file in tqdm(file_names):
        # we read the file
        df = pd.read_csv(file)
        # we get the columns with object type
        cols = [col for col in df.columns if is_object_dtype(df[col])]
        # we convert the columns to string type
        for col in cols:
            df[col] = df[col].astype('string')
        der, year, type_file = file.split('\\')
        # we add the dataframe to the dictionary
        dfs[(der, year, type_file)] = df
    return dfs

def value_types_and_shape(dfs):
    """
    The expected output types are:
    - 16 x 3 string columns: 
        - 1 x 3 for LV_P_installed.csv
        - 1 x 3 for LV_generation.csv
        - 1 x 3 for LV_std.csv
        - 1 x 3 for MV_P_installed.csv
        - 1 x 3 for MV_generation.csv
        - 1 x 3 for MV_std.csv
        - 1 x 3 for BESS_allocation_LV.csv
        - 1 x 3 for BESS_allocation_MV.csv
        - 2 x 3 for LV_heat_pump_allocation.csv
        - 2 x 3 for MV_heat_pump_allocation.csv
        - 1 x 3 for EV_allocation.csv
        - 1 x 3 for EV_power_profiles.csv
        - 1 x 3 for LV_basicload_shares.csv.

    The other columns are numeric columns.
    """

    # we get the types of the dataframe
    cols, dtypes = [], []
    der_list, year_list, type_file_list = [], [], []
    for key, df_temp in dfs.items():
        der, year, type_file = key
        # we get the name of the columns
        cols += df_temp.columns.tolist()
        # we get the types of the columns
        dtypes += [str(t) for t in df_temp.dtypes.tolist()]
        # we get the identifiers of the file
        der_list += [der] * len(df_temp.columns)
        year_list += [year] * len(df_temp.columns)
        type_file_list += [type_file] * len(df_temp.columns)
    # we create a dataframe with the types of the columns
    df_types = pd.DataFrame({'column': cols, 'type': dtypes, 'der': der_list, 'year': year_list, 'type_file': type_file_list})

    print('\n---------------------------------------------------')
    # we check the non float64 and int64 columns
    # the string types must be 16 * 3 
    if df_types[~df_types['type'].isin(['float64', 'int64'])].shape[0] == 16 * 3:
        print('\nThe identifier string columns are correct, and equal to 16 * 3 in total')
    else:
        print('\nERROR: The identifiers are not correct')

    # we check if the other columns are numeric
    if (df_types.shape[0] - df_types[df_types['type'].isin(['float64', 'int64'])].shape[0]) == df_types[~df_types['type'].isin(['float64', 'int64'])].shape[0]:
        print('\nThe rest of the columns are numeric')
    else:
        print('\nERROR: The rest of the columns are not numeric')

    # we print the shape of the dataframe to check if the expected number of columns is correct
    der_list, year_list, type_list, rows_list, column_list = [], [], [], [], []
    for key, df_temp in dfs.items():
        der, year, type_file = key
        # we get the shape of the dataframe
        der_list.append(der)
        year_list.append(year)
        type_list.append(type_file)
        rows_list.append(df_temp.shape[0])
        column_list.append(df_temp.shape[1])
    # we create a dataframe with the shape of the dataframes
    df_shape = pd.DataFrame({'der': der_list, 'year': year_list, 'type_file': type_list, 'columns': column_list, 'rows': rows_list})

    """
    The correct unique number of columns are (as shown in the Supplementary material): correct_columns = [3, 4, 6, 7, 290, 366, 8760, 8761, 8762].
    Total unique number of columns: 9    
    """
    print('\n---------------------------------------------------')

    correct_columns = [3, 4, 6, 7, 290, 366, 8760, 8761, 8762]

    print('Correct number of columns: ', correct_columns) # we check the number of columns
    print('\nCorrect unique number of columns: ', len(correct_columns)) # we check the number of columns

    print('\nNumber of columns in the files: ', sorted([int(c) for c in df_shape['columns'].unique()])) # we check the number of columns
    print('\nUnique number of columns: ', len(df_shape['columns'].unique())) # we check the number of columns

    if sorted(df_shape['columns'].unique()) == correct_columns:
        print('\nThe number of columns is correct')
    else:
        print('\nERROR: The number of columns is not correct')

    """
    The correct unique number of rows are (as shown in the Supplementary material): 
    correct_rows = [1, 440, 2148, 3625, 6444, 7498, 11202, 11551, 13726, 13784, 14759, 15429, 19452, 
                                       151034, 384288, 481318, 630171, 758118, 876594, 998977, 1065403, 1427096, 2525530]

    Total unique number of rows: 24
    """
    print('\n---------------------------------------------------')

    correct_rows = [1, 440, 2148, 3625, 6444, 7449, 11202, 11551, 13481, 13726, 14759, 15429, 19452,
                                       151034, 384287, 481318, 630171, 758118, 876594, 998961, 1065403, 1427096, 2525530]
    print('Correct number of rows: ', correct_rows) # we check the number of rows
    print('\nCorrect unique number of rows: ', len(correct_rows)) # we check the number of rows 

    print('\nNumber of rows in the files: ',sorted([int(c) for c in df_shape['rows'].unique()]))
    print('\nUnique number of rows: ', len(df_shape['rows'].unique())) # we check the number of rows

    if sorted(df_shape['rows'].unique()) == correct_rows:
        print('\nThe number of rows is correct')
    else:
        print('\nERROR: The number of rows is not correct')

    return

def check_none_values(dfs):
    print('\n---------------------------------------------------')
    # we check if there are none values
    none_values = False
    for key, df_temp in dfs.items():
        der, year, type_file = key
        # we check if there are none values
        if df_temp.isnull().sum().sum() > 0:
            none_values = True
            print(f'\nERROR: The file {key} has none values')
    if not(none_values):
        print('\nThe files have no none values')
    return 

def micro_level_consistency(dfs, epsilon):
    print('\n---------------------------------------------------')

    years = sorted(list(set([k[1] for k in df_files.keys()])))
    for year in years:
        LV_installed_geq_gen = (dfs['01_PV', year, 'LV_generation.csv'].iloc[:,2:].max(axis=1) <= dfs['01_PV', year, 'LV_P_installed.csv']['P_installed_kW']).all()
        print(f'\nLV installed PV power is equal or greater than the generation at every node for year {year}: {LV_installed_geq_gen}')
        MV_installed_geq_gen = (dfs['01_PV', year, 'MV_generation.csv'].iloc[:,2:].max(axis=1) <= dfs['01_PV', year, 'MV_P_installed.csv']['P_installed_kW']).all()
        print(f'\nMV installed PV power is equal or greater than the generation at every node for year {year}: {MV_installed_geq_gen}', '\n')
    
    print('\n---------------------------------------------------')

    for year in years:
        upper = dfs['04_EV', year, 'EV_power_profiles_LV.csv'][dfs['04_EV', '2030', 'EV_power_profiles_LV.csv']['Profile_type'] == 'Upper'].iloc[:,2:].values
        base = dfs['04_EV',year, 'EV_power_profiles_LV.csv'][dfs['04_EV', '2030', 'EV_power_profiles_LV.csv']['Profile_type'] == 'Base'].iloc[:,2:].values
        lower = dfs['04_EV', year, 'EV_power_profiles_LV.csv'][dfs['04_EV', '2030', 'EV_power_profiles_LV.csv']['Profile_type'] == 'Lower'].iloc[:,2:].values
        flexibility = dfs['04_EV', year, 'EV_flexible_energy_profiles_LV.csv'].iloc[:,1:].values
        max_deviation_per_hour = np.maximum(np.abs(upper - base), np.abs(lower - base))
        # we get sum per day, considering that the shape is (n_bfs, n_hours) with n_hours = 8760. We have to sum every 24 hours
        max_deviation_per_day = max_deviation_per_hour.reshape(-1, 365, 24)
        sum_per_day = max_deviation_per_day.sum(axis=2)
        # we check that the flexible profiles are equal or lower than the maximum deviation per day
        valid_flexibility = (flexibility <= sum_per_day + epsilon).all()
        print(f'\nThe upper profile is valid for year {year}: {(upper >= base).all()}')
        print(f'\nThe lower profile is valid for year {year}: {(lower <= base).all()}')
        print(f'\nThe flexibility is valid for year {year}: {valid_flexibility}', '\n')

    return

if __name__ == "__main__":

    # as the data was rounded, we check that the flexibility is lower or equal to the sum per day + epsilon
    epsilon = 5

    # we get the name of the files 
    folders = os.listdir()
    folders = [folder for folder in folders if '.' not in folder and any(['0{}'.format(i) in folder for i in range(1,6)])] # we read the folders of DERs and load
    # files per folder
    years = ['2030', '2040', '2050']
    files = []
    for folder in folders:
        for year in years:
            # we get the name of the files 
            files_list = os.listdir(os.path.join(folder, year))
            files += [os.path.join(folder, year, file) for file in files_list if file.endswith('.csv')]
    
    print("Reading the files...")
    # we read the files
    df_files = read_files(files) 
    
    print("Validating the shape of the files and the types of the columns...")
    # we check the types of the columns and the shape of the dataframes
    value_types_and_shape(df_files)
    
    print('\n---------------------------------------------------')
    # we check if there are none values in the dataframes
    print("Checking for none values...")
    check_none_values(df_files)

    print('\n---------------------------------------------------')
    # we check for micro level consistency
    print("Checking for micro level consistency...")
    micro_level_consistency(df_files, epsilon)
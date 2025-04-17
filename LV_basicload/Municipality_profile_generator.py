"""
This code has been created with the objective of inferring the demand profiles for the RRE synthetic networks
starting from publicly available databases. The script reads the dataframe containing all buildings information
and the demand profiles, then classifies for each municipality the buildings and counts how many are of each load class.
Then, it computes the weighted average normalized load profile for residential and commercial LV loads for 
every municipality. 
Then, the demand composition is taken from a raster and applied to the network nodes weighting residential and commercial
consumption.

Run with mpi as: mpiexec -n 1 python Synthetic_networks\Demand_calculator\Municipality_profile_generator.py

Input:
gebaeude_batiment_edificio.csv
residential_hour_profile_EMPA.csv
commercial_hour_profile_EMPA.csv

Output:
normalized_profiles.pkl
residential_nested_dict.pkl
Municipality -> Category -> Construction period, Field: (n_buildings)
commercial_nested_dict.pkl

Created by Lorenzo Zapparoli, 24/6/2024
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from mpi4py import MPI
import pickle

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define the paths of input data, and output
script_path = os.path.dirname(os.path.abspath(__file__))
buildings_path = os.path.join(script_path, "LV_basicload_input", "Buildings_data")
demand_path = os.path.join(script_path, "LV_basicload_input", "JASM_EMPA_data")
save_path = os.path.join(script_path, "Municipality_profiles")
buildings_file_path = os.path.join(buildings_path, "gebaeude_batiment_edificio.csv")
residential_demand_file_path = os.path.join(demand_path, "residential_hour_profile_EMPA.csv")
commercial_demand_file_path = os.path.join(demand_path, "commercial_hour_profile_EMPA.csv")

# Read the buildings database on rank 0 and broadcast it to all ranks
if rank == 0:
    buildings = pd.read_csv(buildings_file_path, sep="\t", low_memory=False)
else:
    buildings = None
buildings = comm.bcast(buildings, root=0)

# Filter the buildings
buildings = buildings[(buildings['GSTAT'] == 1004) & (buildings['GAREA'].notna()) & (buildings['GAREA'] > 5) & (buildings['GKODN'].notna()) & (buildings['GKODE'].notna())]

# Fill the missing year with the average building year
average_construction_year = buildings['GBAUJ'].mean()
buildings['GBAUJ'] = buildings['GBAUJ'].fillna(average_construction_year)

# Assume buildings without number of floors have one floor
buildings['GASTW'] = buildings['GASTW'].fillna(1)

# Assume buildings without number of apartments have one per floor
buildings['GANZWHG'] = buildings['GANZWHG'].fillna(buildings['GASTW'])

# For buildings with missing GEBF, use base area times number of floors
buildings['GEBF'] = buildings['GEBF'].fillna(buildings['GAREA'] * buildings['GASTW'])

# Read the demand attributes database on rank 0 and broadcast to all ranks
if rank == 0:
    residential_demand = pd.read_csv(residential_demand_file_path)
    commercial_demand = pd.read_csv(commercial_demand_file_path)
else:
    residential_demand = None
    commercial_demand = None
residential_demand = comm.bcast(residential_demand, root=0)
commercial_demand = comm.bcast(commercial_demand, root=0)

# Define dictionaries for building categories translation
JASM_GKLAS_translation_dictionary_residential = {
    'MFH': [11, 111, 1121, 1122, 1130, 1211, 1212],
    'SFH': [1110]
}
JASM_GKLAS_translation_dictionary_commercial = {
    'Hospitals': [1264],
    'Offices': [122, 1220],
    'Restaurants': [1231],
    'Schools': [1262, 1263],
    'Shops': [1230]
}


def parse_construction_period(period_str):
    """
    Parse a construction period string into a numeric range.
    
    Args:
    period_str (str): The period string to parse.
    
    Returns:
    np.ndarray: An array of years within the period.
    """

    if 'Before' in period_str or 'before' in period_str:
        year = int(period_str.split(' ')[1])
        return np.arange(0, year + 1)
    else:
        start_year, end_year = map(int, period_str.split('-'))
        return np.arange(start_year, end_year + 1)


def create_period_dict(df):
    """
    Create a dictionary mapping period strings to numpy ranges.
    
    Args:
    df (DataFrame): DataFrame containing the construction periods.
    
    Returns:
    dict: A dictionary mapping period strings to numpy ranges.
    """

    unique_periods = df['Construction_period'].unique()
    period_dict = {period: parse_construction_period(period) for period in unique_periods}
    return period_dict


# Create dictionaries mapping period strings to numpy ranges
commercial_period_dict = create_period_dict(commercial_demand)
residential_period_dict = create_period_dict(residential_demand)


def classify_building_type(gklas, translation_dict):
    """
    Classify a building type based on GKLAS code.
    
    Args:
    gklas (int): GKLAS code of the building.
    translation_dict (dict): Dictionary for translating GKLAS codes to building types.
    
    Returns:
    str: The building type.
    """

    for building_type, gklas_codes in translation_dict.items():
        if gklas in gklas_codes:
            return building_type
    return None


def create_nested_dict(buildings, translation_dict, period_dict):
    """
    Create nested dictionaries for residential and commercial buildings.
    
    Args:
    buildings (DataFrame): DataFrame containing buildings data.
    translation_dict (dict): Dictionary for translating GKLAS codes to building types.
    period_dict (dict): Dictionary mapping period strings to numpy ranges.
    
    Returns:
    dict: A nested dictionary categorizing buildings by municipality, type, and construction period.
    """

    nested_dict = {}
    
    for _, row in buildings.iterrows():
        municipality = row['GGDENR']
        building_type = classify_building_type(row['GKLAS'], translation_dict)
        construction_year = int(row['GBAUJ'])
        
        if building_type is None:
            continue
        
        period_str = None
        for period_key, period_range in period_dict.items():
            if construction_year in period_range:
                period_str = period_key
                break
        
        if period_str is None:
            continue
        
        if municipality not in nested_dict:
            nested_dict[municipality] = {}
        
        if building_type not in nested_dict[municipality]:
            nested_dict[municipality][building_type] = {}
        
        if period_str not in nested_dict[municipality][building_type]:
            nested_dict[municipality][building_type][period_str] = 0
        
        nested_dict[municipality][building_type][period_str] += 1
    
    return nested_dict


# Scatter the buildings data among all ranks
buildings_split = np.array_split(list(buildings.index), size)
local_buildings = comm.scatter(buildings_split, root=0)

# Each rank creates a nested dictionary for its portion of buildings
local_residential_nested_dict = create_nested_dict(buildings.loc[local_buildings, :], JASM_GKLAS_translation_dictionary_residential, residential_period_dict)
local_commercial_nested_dict = create_nested_dict(buildings.loc[local_buildings, :], JASM_GKLAS_translation_dictionary_commercial, commercial_period_dict)

# Gather all nested dictionaries at rank 0
residential_nested_dicts = comm.gather(local_residential_nested_dict, root=0)
commercial_nested_dicts = comm.gather(local_commercial_nested_dict, root=0)

# Lets put everything together
if rank == 0:
    # Combine nested dictionaries from all ranks
    residential_nested_dict = {}
    commercial_nested_dict = {}
    
    for d in residential_nested_dicts:
        for key, value in d.items():
            if key not in residential_nested_dict:
                residential_nested_dict[key] = value
            else:
                for subkey, subvalue in value.items():
                    if subkey not in residential_nested_dict[key]:
                        residential_nested_dict[key][subkey] = subvalue
                    else:
                        for subsubkey, subsubvalue in subvalue.items():
                            if subsubkey not in residential_nested_dict[key][subkey]:
                                residential_nested_dict[key][subkey][subsubkey] = subsubvalue
                            else:
                                residential_nested_dict[key][subkey][subsubkey] += subsubvalue
    
    for d in commercial_nested_dicts:
        for key, value in d.items():
            if key not in commercial_nested_dict:
                commercial_nested_dict[key] = value
            else:
                for subkey, subvalue in value.items():
                    if subkey not in commercial_nested_dict[key]:
                        commercial_nested_dict[key][subkey] = subvalue
                    else:
                        for subsubkey, subsubvalue in subvalue.items():
                            if subsubkey not in commercial_nested_dict[key][subkey]:
                                commercial_nested_dict[key][subkey][subsubkey] = subsubvalue
                            else:
                                commercial_nested_dict[key][subkey][subsubkey] += subsubvalue
    
    # Save the combined dictionaries to files
    with open(os.path.join(save_path, 'residential_nested_dict.pkl'), 'wb') as f:
        pickle.dump(residential_nested_dict, f)
    
    with open(os.path.join(save_path, 'commercial_nested_dict.pkl'), 'wb') as f:
        pickle.dump(commercial_nested_dict, f)
else:
    residential_nested_dict = None
    commercial_nested_dict = None

# Broadcast the combined dictionaries to all ranks
residential_nested_dict = comm.bcast(residential_nested_dict, root=0)
commercial_nested_dict = comm.bcast(commercial_nested_dict, root=0)


def create_normalized_profile(municipality, building_types, demand_data, period_dict, building_category):
    """
    Create a normalized load profile for a municipality and building category.
    
    Args:
    municipality (int): Municipality identifier.
    building_types (dict): Dictionary of building types and periods for the municipality.
    demand_data (DataFrame): DataFrame containing demand data.
    period_dict (dict): Dictionary mapping period strings to numpy ranges.
    building_category (str): The building category ('residential' or 'commercial').
    
    Returns:
    tuple: (municipality, building_category, normalized profile)
    """

    profile = np.zeros(8760)
    
    for building_type, periods in building_types.items():
        for period_str, count in periods.items():
            demand_profile = demand_data[
                (demand_data['Building_type'] == building_type) &
                (demand_data['Construction_period'] == period_str) &
                (demand_data['retrofit_scenario'] == 'Full retrofit')
            ]['Electricity demand [J](Hourly) '].values
            
            if len(demand_profile) > 0:
                demand_profile = np.array(demand_profile) / (1000 * 3600) # From J to kWh
                profile += count * demand_profile
    
    if profile.max() > 0:
        profile /= profile.max()
    
    return (municipality, building_category, profile.tolist())

# Scatter the municipalities among all ranks
municipalities = list(residential_nested_dict.keys())
municipalities_split = np.array_split(municipalities, size)
local_municipalities = comm.scatter(municipalities_split, root=0)

# Each rank creates normalized profiles for its portion of municipalities
local_profiles_dict = {}

if rank == 0:
    iterable = tqdm(local_municipalities)
else:
    iterable = local_municipalities
    
for municipality in iterable:
    if municipality in residential_nested_dict:
        result = create_normalized_profile(municipality, residential_nested_dict[municipality], residential_demand, residential_period_dict, 'residential')
        if result[0] not in local_profiles_dict:
            local_profiles_dict[result[0]] = {}
        local_profiles_dict[result[0]][result[1]] = result[2]
    
    if municipality in commercial_nested_dict:
        result = create_normalized_profile(municipality, commercial_nested_dict[municipality], commercial_demand, commercial_period_dict, 'commercial')
        if result[0] not in local_profiles_dict:
            local_profiles_dict[result[0]] = {}
        local_profiles_dict[result[0]][result[1]] = result[2]

# Gather all profiles dictionaries at rank 0
all_profiles_dicts = comm.gather(local_profiles_dict, root=0)

# We put everything back together
if rank == 0:
    # Combine profiles dictionaries from all ranks
    profiles_dict = {}
    for d in all_profiles_dicts:
        for key, value in d.items():
            if key not in profiles_dict:
                profiles_dict[key] = value
            else:
                profiles_dict[key].update(value)
                
    # Save the combined profiles to a file
    with open(os.path.join(save_path, 'municipality_normalized_profiles.pkl'), 'wb') as f:
        pickle.dump(profiles_dict, f)

    # Prepare data for CSV files
    residential_data = {}
    commercial_data = {}

    for municipality, categories in profiles_dict.items():
        if 'residential' in categories:
            residential_data[municipality] = categories['residential']
        if 'commercial' in categories:
            commercial_data[municipality] = categories['commercial']

    # Convert to DataFrames
    residential_df = pd.DataFrame.from_dict(residential_data, orient='index').transpose()
    commercial_df = pd.DataFrame.from_dict(commercial_data, orient='index').transpose()

    # Save to CSV files
    residential_df.to_csv(os.path.join(save_path, 'residential_profiles.csv'), index=False)
    commercial_df.to_csv(os.path.join(save_path, 'commercial_profiles.csv'), index=False)

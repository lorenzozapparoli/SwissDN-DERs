"""
Author: Lorenzo Zapparoli
Institution: ETH Zurich
Date: 15/03/2025

Introduction:
This script, `Result_condenser.py`, processes Low Voltage (LV) grid data to generate and save scaled electricity demand profiles and demand shares for residential and commercial categories. The script reads data from multiple LV grid files, aggregates the required information, and saves the results in specified output directories for the years 2030, 2040, and 2050.

Usage:
1. Ensure the required input files (LV grid data and municipality profiles) are available in the specified directories.
2. Run the script to generate scaled profiles and demand shares.
3. The output files will be saved in the `LV_basicload_output` directory.

Dependencies:
- pandas
- geopandas
- os
"""

import os
import pandas as pd
import geopandas as gpd


def generate_profiles_shares():
    """
    Processes LV grid data to generate demand shares for residential and commercial categories.

    Description:
    - Iterates through LV grid folders and reads node data from GeoJSON files.
    - Extracts residential and commercial demand percentages and associates them with grid nodes.
    - Aggregates the data into a single DataFrame and saves it as a CSV file in the output directories for 2030, 2040, and 2050.
    """

    # Define the path to the LV folder
    path = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(path)
    lv_grids_path = os.path.join(base_path, 'Grids', 'LV')

    # Initialize an empty list to store data from each file
    data_list = []

    # Iterate through all folders and files in the LV directory
    for root, dirs, files in os.walk(lv_grids_path):
        for dir in dirs:
            folder_path = os.path.join(lv_grids_path, dir)
            print(f"Processing folder: {folder_path}")
            for file in os.listdir(folder_path):
                if file.endswith('_nodes'):
                    # Construct the full file path
                    file_path = os.path.join(folder_path, file)

                    # Read the GeoJSON file
                    gdf = gpd.read_file(file_path)

                    # Extract the grid code from the file name
                    grid_code = file.split('_nodes')[0]

                    # Extract the required columns and add the grid code
                    df = gdf[['osmid', 'com_percentage', 'res_percentage']].copy()
                    df['LV_grid'] = grid_code
                    df.rename(columns={'osmid': 'LV_osmid'}, inplace=True)

                    # Append the DataFrame to the list
                    data_list.append(df)

    # Concatenate all DataFrames into a single DataFrame
    result_df = pd.concat(data_list, ignore_index=True)

    # Reorder the columns
    result_df = result_df[['LV_grid', 'LV_osmid', 'com_percentage', 'res_percentage']]

    # Define the output directories for the scaled profiles
    output_dirs = ['LV_basicload_output/2030', 'LV_basicload_output/2040',
                   'LV_basicload_output/2050']

    # # Create the output directory if it doesn't exist
    # output_dir = 'Concatenated_results'
    # os.makedirs(output_dir, exist_ok=True)

    # Save the DataFrame to a CSV file
    # Save the scaled profiles to the specified folders
    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, 'LV_basicload_shares.csv')
        result_df.to_csv(output_file_path, index=False)

    print(f"Data has been successfully saved")


def generate_profiles():
    """
    Scales residential and commercial demand profiles using a calibration factor.

    Description:
    - Reads residential and commercial profiles from CSV files.
    - Applies a calibration factor to scale the profiles.
    - Saves the scaled profiles to the output directories for 2030, 2040, and 2050.
    """

    # Define the calibration factor
    calibration_factor = 0.73

    # Define the paths to the commercial and residential profiles
    commercial_profiles_path = 'LV_basicload_output/municipality_profiles/Commercial_profiles.csv'
    residential_profiles_path = 'LV_basicload_output/municipality_profiles/Residential_profiles.csv'

    # Read the commercial and residential profiles
    commercial_profiles_df = pd.read_csv(commercial_profiles_path)
    residential_profiles_df = pd.read_csv(residential_profiles_path)

    # Scale the profiles by the calibration factor
    scaled_commercial_profiles_df = commercial_profiles_df * calibration_factor
    scaled_residential_profiles_df = residential_profiles_df * calibration_factor

    # Define the output directories for the scaled profiles
    output_dirs = ['LV_basicload_output/2030', 'LV_basicload_output/2040', 'LV_basicload_output/2050']

    # Save the scaled profiles to the specified folders
    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)
        scaled_commercial_profiles_df.to_csv(os.path.join(output_dir, 'Commercial_profiles.csv'), index=False)
        scaled_residential_profiles_df.to_csv(os.path.join(output_dir, 'Residential_profiles.csv'), index=False)

    print("Scaled profiles have been successfully saved to the specified folders.")


if __name__ == '__main__':
    # generate_profiles_shares()
    generate_profiles()

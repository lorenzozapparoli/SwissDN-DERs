"""
Author: Lorenzo Zapparoli
Institution: ETH Zurich
Date: 15/03/2025

Introduction:
This script, `MV_profile_generator.py`, processes Medium Voltage (MV) load data to generate hourly load profiles for the years 2015 to 2023. The script aligns each year's data to start on the first Monday and extracts 8760-hour profiles (one year of hourly data). It calculates the average yearly profile and the standard deviation for each hour, normalizing the results to the maximum value of the original dataset.

The processed profiles are saved as CSV files for further analysis or use in simulations.

Usage:
1. Ensure the input file `Zurich_2015-2024.csv` is available in the `MV_basicload_input` directory.
2. Run the script to generate normalized MV load profiles.
3. The output files will be saved in the `MV_basicload_output` directory.

Dependencies:
- pandas
- os
"""

import pandas as pd
import os

# Load the data
base_path = os.getcwd()
data_output_path = os.path.join(base_path, 'MV_basicload_output')
data_path = os.path.join(base_path, 'MV_basicload_input', 'Zurich_2015-2024.csv')
data = pd.read_csv(data_path)

# Convert the timestamp to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%Y-%m-%dT%H:%M:%S%z', errors='coerce', utc=True)
data['Timestamp'] = data['Timestamp'] + pd.DateOffset(hours=1)

# Group the 15-minute steps into hourly data
data = data.set_index('Timestamp').resample('h').mean().reset_index()


# Add necessary columns
data['Year'] = data['Timestamp'].dt.year
data['DayOfWeek'] = data['Timestamp'].dt.dayofweek


def extract_yearly_profile(df):
    """
    Aligns the data for a specific year to start on the first Monday and extracts the 8760-hour profile.

    Args:
        df (DataFrame): DataFrame containing the load data for a specific year.

    Returns:
        DataFrame: DataFrame containing the 8760-hour profile for the year.

    Description:
    - Identifies the first Monday in the data.
    - Extracts data starting from the first Monday for the next 8760 hours (one year).
    """

    first_monday = df[df['DayOfWeek'] == 0].iloc[0]['Timestamp']
    df = df[(df['Timestamp'] >= first_monday) & (df['Timestamp'] < first_monday + pd.Timedelta(hours=8760))]
    return df


profiles = []
for year in range(2015, 2024):
    yearly_data = data[data['Year'] >= year]
    if not yearly_data.empty:
        profile = extract_yearly_profile(yearly_data)
        if len(profile) == 8760:
            profiles.append(profile['Value_NE5'].values)

# Convert profiles to DataFrame
profiles_df = pd.DataFrame(profiles).T

# Compute the average yearly profile
average_profile = profiles_df.mean(axis=1)

# Compute the standard deviation for each hour
std_profile = profiles_df.std(axis=1)

# Normalize to the maximum value of the original dataframe
max_value = data['Value_NE5'].max()
average_profile = average_profile / max_value
std_profile = std_profile / max_value

# Save the results to CSV files
average_profile.to_csv(os.path.join(data_output_path, 'MV_load_profile.csv'), index=False, header=['Power_pu'])
# std_profile.to_csv(os.path.join(data_output_path, 'MV_load_profile_std.csv'), index=False, header=['Power_pu'])

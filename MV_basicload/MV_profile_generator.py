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

# Function to align each year to start on the first Monday and extract the 8760-hour profile
def extract_yearly_profile(df):
    first_monday = df[df['DayOfWeek'] == 0].iloc[0]['Timestamp']
    df = df[(df['Timestamp'] >= first_monday) & (df['Timestamp'] < first_monday + pd.Timedelta(hours=8760))]
    return df

# Extract profiles for each year from 2015 to 2023
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
std_profile.to_csv(os.path.join(data_output_path, 'MV_load_profile_std.csv'), index=False, header=['Power_pu'])

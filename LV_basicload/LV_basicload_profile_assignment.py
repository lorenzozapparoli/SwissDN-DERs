import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpi4py import MPI
import tqdm
import warnings
import json
import os
import geopandas as gpd
import pickle
warnings.filterwarnings('ignore')

# Define the path to store municipality profiles
profle_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LV_basicload_output', 'municipality_profiles')

def fill_all_municipalities():
    """
    Fill missing municipality profiles with averages:
    - Loads residential and commercial profiles.
    - Identifies and fills columns with zero values using the average of non-empty columns.
    - Ensures all keys (municipalities) are included, adding missing ones with the average profile.
    - Saves updated profiles to CSV files.
    """
    with open(profle_path+'/dict_folder.json') as f:
        folders = json.load(f)
    res_profile = pd.read_csv(os.path.join(profle_path,'residential_profiles.csv'))
    com_profile = pd.read_csv(os.path.join(profle_path,'commercial_profiles.csv'))
    cols = res_profile.columns
    not_included = []
    empty_columns = []
    keys =folders.keys()
    average_res = pd.DataFrame()
    average_com = pd.DataFrame()
    print('Filling the empty columns with the average of the other columns')
    for column in tqdm.tqdm(res_profile.columns):
        if res_profile[column].sum() == 0 or com_profile[column].sum() == 0:
            empty_columns.append(column)
        else:
            average_res[column] = res_profile[column]
            average_com[column] = com_profile[column]
    average_res = average_res.mean(axis=1)
    average_com = average_com.mean(axis=1)
    for column in empty_columns:
        res_profile[column] = average_res
        com_profile[column] = average_com
    print('check if all the keys are included')
    for key in tqdm.tqdm(keys):
        if key in cols:
            continue
        else:
            not_included.append(key)
            res_profile[key] = average_res
            com_profile[key] = average_com
    res_profile.to_csv(os.path.join(profle_path,'residential_profiles.csv'),index=False)
    com_profile.to_csv(os.path.join(profle_path,'commercial_profiles.csv'),index=False)
    print('Successfully saved the profiles')
    
def profile_prep(profile):
    """
    Prepare a profile by replacing empty columns with the average of other columns:
    - Identifies columns with all zero values.
    - Calculates the mean across non-empty columns.
    - Replaces zero-filled columns with the calculated average.
    """
    empty_columns = []
    profile_averages = pd.DataFrame()
    for column in profile.columns:
        if profile[column].sum() == 0:
            empty_columns.append(column)
        else:
            profile_averages[column] = profile[column]
    average_profile = profile_averages.mean(axis=1)
    for column in empty_columns:
        profile[column] = average_profile

def divide_into_days(profile):
    """
    Split a time series profile into daily data:
    - Reshapes the profile data into a matrix with 24 columns (hours per day).
    """
    days = profile.values.reshape(-1, 24)
    return days

def find_typical_days(days, max_k=4):
    """
    Identify typical daily profiles using K-means clustering:
    - Finds the optimal number of clusters (k) based on silhouette score.
    - Returns cluster centers (typical days), counts of days in each cluster, the optimal k, and the score.
    """
    best_k = 2
    best_score = -1
    best_kmeans = None
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(days)
        score = silhouette_score(days, labels)
        if score > best_score:
            best_k = k
            best_score = score
            best_kmeans = kmeans
    typical_days = best_kmeans.cluster_centers_
    typical_days = pd.DataFrame(typical_days)
    counts = np.bincount(best_kmeans.labels_)
    return typical_days, counts, best_k, best_score

def process_columns(columns, profile):
    """
    Process each column of the profile to extract typical daily profiles:
    - Divides each column's data into days.
    - Finds typical days using clustering and stores them along with cluster counts.
    - Returns a dictionary containing profiles and counts for each column.
    """
    new_profile = {}
    pbar = tqdm.tqdm(total=len(columns))
    for column in columns:
        days = divide_into_days(profile[column])
        typical_days, counts, best_k, best_score = find_typical_days(days)
        flattened_typical_days = typical_days.values.flatten()
        new_profile[column] = {
            'profile': [flattened_typical_days],
            'counts': [counts]
        }
        pbar.update(1)
    pbar.close()
    return new_profile

def get_typical_commercial_profile():
    """
    Generate and save typical commercial profiles using K-means clustering:
    - Prepares the commercial profile data.
    - Processes each column to extract typical daily profiles.
    - Saves the result to a pickle file for future use.
    """
    com_profile = pd.read_csv('Demand_calculator/commercial_profiles.csv')
    commercial_profile = pd.DataFrame(com_profile)
    profile_prep(commercial_profile)
    columns = list(commercial_profile.columns)
    new_profile = process_columns(columns, commercial_profile)
    with open('municipality_profiles/commercial_profile_k-means.pkl', 'wb') as f:
        pickle.dump(new_profile, f)

def get_daily_com_profile():
    """
    Generate and save daily average commercial profiles:
    - Prepares the commercial profile data.
    - Calculates the average daily profile for each column.
    - Saves the result to a CSV file.
    """
    com_profile = pd.read_csv('Demand_calculator/commercial_profiles.csv')
    commercial_profile = pd.DataFrame(com_profile)
    profile_prep(commercial_profile)
    columns = list(commercial_profile.columns)
    new_data = {}
    for column in tqdm.tqdm(columns):
        days = divide_into_days(commercial_profile[column])
        new_data[column] = days.mean(axis=0)
    commercial_new = pd.DataFrame(new_data)
    commercial_new.to_csv('Demand_calculator/commercial_profiles_24h.csv')

def get_daily_res_profile():
    """
    Generate and save daily average residential profiles:
    - Prepares the residential profile data.
    - Calculates the average daily profile for each column.
    - Saves the result to a CSV file.
    """
    new_data = {}
    res_profile = pd.read_csv('Demand_calculator/residential_profiles.csv')
    residential_profile = pd.DataFrame(res_profile)
    profile_prep(residential_profile)
    columns = list(residential_profile.columns)
    for column in tqdm.tqdm(columns):
        days = divide_into_days(residential_profile[column])
        new_data[column] = days.mean(axis=0)
    residential_new = pd.DataFrame(new_data)
    residential_new.to_csv('Demand_calculator/residential_profiles_24h.csv')

def assign_to_node_yearly(grid_id):
    """
    Assign yearly profiles to nodes in a grid:
    - Loads commercial and residential profiles.
    - Constructs yearly profiles for both types by repeating daily profiles.
    - Assigns combined profiles (weighted by percentages) to nodes in the grid.
    - Saves the node data with assigned profiles to disk.
    """
    with open('municipality_profiles/dict_folder.json') as f:
        folders = json.load(f)
    # Load profiles
    with open('municipality_profiles/commercial_profile_k-means.pkl', 'rb') as f:
        commercial_profile_kmean = pickle.load(f)
    residential_profile = pd.read_csv('municipality_profiles/residential_profiles_24h.csv', index_col=0)
    commercial_profile = pd.read_csv('municipality_profiles/commercial_profiles_24h.csv', index_col=0)
    # Construct yearly profiles
    commercial_profile_day = commercial_profile_kmean[grid_id]['profile'].reshape(-1, 24)
    counts = commercial_profile_kmean[grid_id]['counts']
    repeated_profiles = [profile for profile, count in zip(commercial_profile_day, counts) for _ in range(count)]
    np.random.shuffle(repeated_profiles)
    commercial_profile_year = np.concatenate(repeated_profiles)
    residential_profile_day = np.array(residential_profile[grid_id])
    residential_profile_year = np.tile(residential_profile_day, 365)
    # Process grid nodes
    path = 'LV/' + folders[grid_id] + '/'
    grid_ids = list(set([str(f.split('.')[0][:-6]) for f in os.listdir(path) if f.startswith(grid_id + '-')]))
    for n in tqdm.tqdm(range(len(grid_ids))):
        i = grid_ids[n]
        node_id = i + "_nodes"
        nodes = gpd.read_file('LV/' + folders[grid_id] + '/' + node_id)
        nodes['profile'] = {}
        for iter, row in nodes.iterrows():
            combined_profile = row['res_percentage'] * residential_profile_year + row['com_percentage'] * commercial_profile_year
            nodes.at[iter, 'profile'] = combined_profile
        # Save node data with profiles
        if not os.path.exists('LV_with_profiles/' + folders[grid_id]):
            os.makedirs('LV_with_profiles/' + folders[grid_id])
        save_path = 'LV_with_profiles/' + folders[grid_id] + '/' + i + '_nodes'
        nodes.to_pickle(save_path)

if __name__ == "__main__":
    # Main script execution
    get_typical_commercial_profile()
    get_daily_res_profile()
    fill_all_municipalities()
    '''
    for key in tqdm.tqdm(folders.keys()):
        assign_to_node_yearly(key)
    '''


        




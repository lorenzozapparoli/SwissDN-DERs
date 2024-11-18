import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import pickle
import tqdm
import warnings
warnings.filterwarnings('ignore')
'''
This script is used to calculate the error between the actual commercial profile 
and the clustered profile. The error is calculated as the sum of the absolute error
'''

# Load the data
with open('municipality_profiles/commercial_profile_k-means.pkl', 'rb') as f:
    profile_clustered = pickle.load(f)
profile_original = pd.read_csv('municipality_profiles/commercial_profiles.csv')

municipalities = profile_clustered.keys()
municipalities = np.array(list(municipalities))
Results = pd.DataFrame(columns=['municipality', 'RMSE', 'Error'])

for i in tqdm.tqdm(range(len(municipalities))):
    municipality = municipalities[i]
    profile_c = profile_clustered[municipality]['profile'].reshape(-1, 24)
    count = profile_clustered[municipality]['counts']
    repeated_profile = [profile for profile, count in zip(profile_c, count) for _ in range(count)]
    repeated_profile = np.array(repeated_profile)
    # flatten the repeated profile to compare with the original profile
    profile_clustered_m = repeated_profile.flatten()
    profile_original_m = profile_original[municipality].values

    # rearrange the profiles by the quantiles
    profile_clustered_m = np.sort(profile_clustered_m, axis=0)
    profile_original_m = np.sort(profile_original_m)

    # calculate RMSE
    rmse = np.sqrt(np.mean((profile_clustered_m - profile_original_m)**2))
    Error = np.sum(np.abs(profile_clustered_m - profile_original_m))
    ERROR_percentage = Error/np.sum(profile_original_m)
    Results = pd.concat([Results, pd.DataFrame({'municipality': [municipality], 'RMSE': [rmse], 'Error': [Error],'Error_percentage':[ERROR_percentage]}, index=[0])], ignore_index=True)

# save the RMSEs and Errors
Results.to_csv('municipality_profiles/commercial_profile_error.csv', index=False)
print('Done!')







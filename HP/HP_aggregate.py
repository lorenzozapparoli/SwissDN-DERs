import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import time
import warnings 
warnings.filterwarnings('ignore')

script_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_path, 'Buildings_data')

data_processed = pd.read_csv(os.path.join(data_path, 'buildings_data_with_stations.csv'))

# create folder HP_allocation if it does not exist
if not os.path.exists(os.path.join(script_path, 'HP_allocation')):
    os.makedirs(os.path.join(script_path, 'HP_allocation'))


data_processed.drop(columns=['GKODN', 'GKODE','ISHP',
                             'geometry', 'EGID'], inplace=True)
data_processed['MV_osmid'] = data_processed['MV_osmid'].astype(int)

ts = time.time()
print('--------------Start processing data--------------')
MV_ids = data_processed['MV_grid'].unique()
MV_ids = MV_ids[MV_ids != '-1']
for id in MV_ids:
    print(f'Processing grid {id}...')
    building_data_part = data_processed[data_processed['MV_grid'] == id]
    most_frequent_T_profile = building_data_part.groupby('MV_osmid')['T_PROFILE'].agg(lambda x: x.mode()[0])
    building_data_0 = building_data_part.drop(columns=['MV_grid']).groupby('MV_osmid').sum()
    building_data_0.reset_index(inplace=True)
    building_data_0['T_PROFILE'] = building_data_0['MV_osmid'].map(most_frequent_T_profile)
    building_data_0['PRT'] = building_data_0['PRT']/10**6 # convert PRT to MW
    building_data_0['CBLD'] = building_data_0['CBLD']/10**3 # convert CBLD to MWh/K
    building_data_0['HBLD'] = building_data_0['HBLD']/10**3 # convert HBLD to kW/K
    # rename PRT to PRT_MW
    building_data_0.rename(columns={'PRT':'PRT_MW', 'CBLD':'CBLD_MWh/K',
                                    'HBLD':'HBLD_kW/K', 'GEBF':'GEBF_m2', 'GAREA':'GAREA_m2'}, inplace=True)
    
    if not os.path.exists(os.path.join(script_path, 'HP_allocation_old')):
        os.makedirs(os.path.join(script_path, 'HP_allocation_old'))
    building_data_0.to_csv(os.path.join(script_path, 'HP_allocation_old', 'HP_'+str(id)+'.csv'), index=False)
    
te=time.time()
print(f'--------------Finish processing {len(MV_ids)} grids--------------')
print('Time consumed: ', round(te-ts, 2), 's')

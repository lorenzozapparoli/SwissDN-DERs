import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import geopandas as gpd
import warnings
warnings.filterwarnings("ignore")

buffer_distance = '3000'
Threshold = 3
path = 'PV_allocation_results/'+buffer_distance+'/'
ids = os.listdir(path)
ids.sort()

def mad(arr, min_mad=1e-6):
    median = np.median(arr)
    abs_deviation = np.abs(arr - median)
    mad_value = 1.483 * np.median(abs_deviation)
    return max(mad_value, min_mad)  

def robust_z_scores(arr):
    median = np.median(arr)
    mad_value = mad(arr)
    z_scores = (arr - median) / mad_value
    return z_scores, mad_value, median

def outliers_correction(x, threshold=3):
    z_scores, mad_value, median = robust_z_scores(x['max_demand'])
    x['z_scores'] = z_scores
    x_new = x.copy()
    outliers = x_new[x_new['z_scores'] > threshold]
    normal = x_new[x_new['z_scores'] <= threshold]
    iterations_data = [x.copy()]
    iter = 0
    
    original_mean = x['max_demand'].mean()  
    
    while len(outliers) > 0:
        Lambda = 0.05
        excess_value = 0
    
        for index, row in outliers.iterrows():
            reduction = x.loc[index, 'max_demand'] * Lambda
            x.loc[index, 'max_demand'] -= reduction
            excess_value += reduction
        
        for index, row in normal.iterrows():
            x.loc[index, 'max_demand'] += excess_value / len(normal)

        z_scores, mad_value, median = robust_z_scores(x['max_demand'])
        x['z_scores'] = z_scores
        outliers = x[x['z_scores'] > threshold]
        normal = x[x['z_scores'] <= threshold]
        iterations_data.append(x.copy())
        iter += 1
        mean_value = x['max_demand'].mean()
        if abs(mean_value - original_mean) < 1e-3:
            continue
        else:
            print('Mean value changed')
            break       
    return x, iterations_data

def plot_iteration(iterations_data, id):
    fig = plt.figure(figsize=(12, 6))
    mean = np.mean(iterations_data[0]['max_demand'])
    plt.axhline(mean, color='black', linestyle='--', label='Mean')
    
    outliers = iterations_data[0][iterations_data[0]['z_scores'] > 3]
    plt.scatter(outliers.index, outliers['max_demand'], color='red', marker='*', label='Outliers')
    
    plt.plot(iterations_data[0]['max_demand'], label='Original')
    for i in range(1, len(iterations_data), 3):
        plt.plot(iterations_data[i]['max_demand'], label=f'Iteration {i}')
    
    plt.xlabel('Node')
    plt.xticks(np.arange(0, len(iterations_data[0]['max_demand']), 5))
    plt.ylabel('PV generation (W)')
    plt.grid()
    plt.legend()
    plt.title(f'PV generation {id} with buffer distance {buffer_distance}')
    plt.savefig(f'Pictures/{buffer_distance}_PV_generation_{id}.png', dpi=300,bbox_inches='tight')
    plt.show()

def plot_demand(demand, std, name, id):
    n = 1
    lens_columns = len(demand.columns)-1
    fig = plt.figure(figsize=(12, 6))
    plt.plot(demand.iloc[n,1:].values, label=f"Node {str(n)}")
    plt.fill_between(np.arange(lens_columns), demand.iloc[n,1:]-std.iloc[n,1:], demand.iloc[n,1:]+std.iloc[n,1:], alpha=0.3)

    plt.plot(demand.iloc[n+1,1:].values, label=f"Node {str(n+1)}")
    plt.fill_between(np.arange(lens_columns), demand.iloc[n+1,1:]-std.iloc[n+1,1:], demand.iloc[n+1,1:]+std.iloc[n+1,1:], alpha=0.3)

    plt.plot(demand.iloc[n+2,1:].values, label=f"Node {str(n+2)}")
    plt.fill_between(np.arange(lens_columns), demand.iloc[n+2,1:]-std.iloc[n+2,1:], demand.iloc[n+2,1:]+std.iloc[n+2,1:], alpha=0.3)

    #plt.xticks([])
    plt.xlabel('Hour')
    plt.ylabel('PV generation (W)')
    plt.legend()
    plt.grid()
    plt.title(f'PV generation grid {id} {name}')
    plt.savefig(f'Pictures/{buffer_distance}_+{name}_PV_generation_grid_{id}_node_{str(n)}.png', dpi=300,bbox_inches='tight')
    plt.show()

def HHI_index(demand, id):
    demand_HHI = demand.copy()
    demand_HHI['max_demand'] = demand_HHI.iloc[:,1:].max(axis=1)
    demand_HHI['percentage'] = demand_HHI['max_demand']/demand_HHI['max_demand'].sum()
    HHI = 0
    for i in range(len(demand_HHI)):
        HHI += demand_HHI['percentage'][i]**2
    return HHI

def add_nearest_node(demand, std, id):
    nodes = gpd.read_file('MV/'+id+'_nodes')
    nodes['osmid'] = nodes['osmid'].astype(int)
    max_demand = demand.iloc[:,1:].max(axis=1).idxmax()
    osmid_max_demand = demand.loc[max_demand, 'MV_osmid']
    geo_max_demand = nodes.loc[nodes['osmid']==osmid_max_demand, 'geometry'].values[0]
    nodes['distance_to_reference'] = nodes['geometry'].distance(geo_max_demand)
    gdf_non_reference = nodes[~nodes['osmid'].isin(demand['MV_osmid'])]  
    closest_node = gdf_non_reference.loc[gdf_non_reference['distance_to_reference'].idxmin()]
    demand_means = demand.iloc[:, 1:].mean()*1e-8
    demand_new_row = pd.DataFrame([[closest_node['osmid']] + demand_means.tolist()], columns=demand.columns)
    demand = pd.concat([demand, demand_new_row], ignore_index=True)
    std_means = std.iloc[:, 1:].mean()*1e-8
    std_new_row = pd.DataFrame([[closest_node['osmid']] + std_means.tolist()], columns=std.columns)
    std = pd.concat([std, std_new_row], ignore_index=True)
    return demand, std

def read_demand_std(id):
    demand = pd.read_pickle(path+id+'/'+id+'_demand.pkl')
    std = pd.read_pickle(path+id+'/'+id+'_std.pkl')
    return demand, std

def update_demand_std(new_x, x, demand, std):
    demand_upd = demand.copy()
    std_upd = std.copy()
    correction_factor = new_x['max_demand'] / x['max_demand']
    demand_upd.iloc[:,1:] = demand_upd.iloc[:,1:].multiply(correction_factor, axis=0)
    std_upd.iloc[:,1:] = std_upd.iloc[:,1:].multiply(correction_factor, axis=0)
    return demand_upd, std_upd

#HHI_list = pd.DataFrame(columns=['index', 'id', 'HHI_after'])
for i in range(419,420):
    print(f"processing {ids[i]}")
    demand, std = read_demand_std(ids[i])
    plot_demand(demand, std, 'before', ids[i])
    x = demand.iloc[:,1:].max(axis=1)
    x = pd.DataFrame(x, columns=['max_demand'])
    new_x, iterations_data = outliers_correction(x.copy())  
    demand_upd, std_upd = update_demand_std(new_x, x, demand, std)
    HHInew = HHI_index(demand_upd, ids[i])
    demand_node =demand.copy()
    std_node = std.copy()
    while HHInew > 0.25:
        demand_node, std_node = add_nearest_node(demand_node, std_node, ids[i])
        print("Number of nodes:", len(demand_node))
        x = demand_node.iloc[:, 1:].max(axis=1)
        x = pd.DataFrame(x, columns=['max_demand'])
        new_x, iterations_data = outliers_correction(x.copy())
        demand_upd, std_upd = update_demand_std(new_x, x, demand_node, std_node)
        HHInew = HHI_index(demand_upd, ids[i])
    plot_iteration(iterations_data, ids[i])
    print(f"Final HHI index: {HHInew}")
    print(f"Finish processing {ids[i]}")
    plot_demand(demand, std, 'after', ids[i])
    demand_upd.to_pickle(path+ids[i]+'/'+ids[i]+'_demand_shrinkage.pkl')
    std_upd.to_pickle(path+ids[i]+'/'+ids[i]+'_std_shrinkage.pkl')
    

#HHI_list.to_csv('PV_data/HHI_list.csv', index=False)
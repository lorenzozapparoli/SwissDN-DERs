import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
import geopandas as gpd
from shapely.ops import unary_union
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon
from geovoronoi import voronoi_regions_from_coords, points_to_coords
import tqdm
warnings.filterwarnings('ignore')

'''
This class is used to allocate the percentage of commercial and residential load to each node in the grid
save_id() function is used to save the grid ids in a json file
import_demand() function is used to import the demand data from the geojson file
concat_all_grids() function is used to concatenate all the nodes and edges of the grid
voronoi() function is used to do the voronoi partitioning and assign the residential and commercial percentages to the nodes
save_allocation() function is used to save the allocation for each grid
profile_generation() function is used to generate the profiles for each building in the municipality
Input:
x_el_dmd.geojson: the demand data of the grid
x_nodes.geojson: the nodes of the grid

Output:
x_nodes.geojson: the nodes of the grid with the residential and commercial percentages
x_voronoi.png: the voronoi diagram of the grid
'''
plt.rcParams.update({
    'font.size': 10,            # Base font size for the plot
    'font.family': 'Times New Roman',  # Font style (IEEE recommends Times New Roman)
    'axes.labelsize': 10,       # Font size for axis labels
    'axes.titlesize': 10,       # Font size for the title
    'legend.fontsize': 12,      # Font size for the legend
    'xtick.labelsize': 10,      # Font size for x-axis tick labels
    'ytick.labelsize': 10,      # Font size for y-axis tick labels
    'lines.linewidth': 1.0,    # Line width for plot lines
    'lines.markersize': 4,     # Marker size
    'figure.figsize': [3.5, 2.5], # Size of the figure (width x height) in inches
    'savefig.dpi': 300,        # Resolution of the output figure
    'legend.loc': 'best',      # Location of the legend
    'legend.frameon': False,   # Remove the box frame around legends
    'pdf.fonttype': 42,        # Embedding fonts in PDF for compatibility
    'ps.fonttype': 42
})

class load_allocation:
    def __init__(self):
        self.grids_name = str
        self.grid_ids = list
        self.grid_dict = dict
        self.buildings = gpd.GeoDataFrame()
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.input_path = os.path.join(self.path, 'LV_basicload_input')
        self.output_path = os.path.join(self.path, 'LV_basicload_output')
                
    def create_dict(self,save_dict=False):
        path = self.input_path+'/Square_zones_dmd/'
        list_folder = os.listdir(path)
        dict_folder = {}
        for folder in list_folder:
            path_folder = path+folder+'/'
            list_files = os.listdir(path_folder)
            for file in list_files:
                file_name = file.split('_')[0]
                dict_folder[file_name] = folder
                # sort the dictionary by the keys
                dict_folder = dict(sorted(dict_folder.items()))
        if save_dict:
            with open(self.output_path+'/municipality_profiles/dict_folder.json', 'w') as f:
                json.dump(dict_folder, f)
        return dict_folder
               
    def save_id(self):
        path = 'LV/'+self.grid_dict[self.grids_name]+'/'
        grid_ids = list(set([str(f.split('.')[0][:-6]) for f in os.listdir(path) if f.startswith(self.grids_name+'-')]))
        self.grid_ids = grid_ids
        return grid_ids
        
    def import_demand(self):
        el_dmd_name = self.input_path+'/Square_zones_dmd/'+self.grid_dict[self.grids_name]+'/'+self.grids_name+'_el_dmd.geojson'
        el_dmd = gpd.read_file(el_dmd_name)
        el_dmd['residential_percentage'] = el_dmd['residential']/(el_dmd['residential']+el_dmd['commercial'])
        el_dmd['commercial_percentage'] = el_dmd['commercial']/(el_dmd['residential']+el_dmd['commercial'])
        el_dmd['residential_percentage'] = el_dmd['residential_percentage'].fillna(0)
        el_dmd['commercial_percentage'] = el_dmd['commercial_percentage'].fillna(0)
        return el_dmd
    
    def concat_all_grids(self):
        grid_ids = self.grid_ids
        node_total = gpd.GeoDataFrame()
        edge_total = gpd.GeoDataFrame()
        # add timebar to show the progress using tqdm
        print("Concatenating all the nodes and edges of the grids...")
        for n in tqdm.tqdm(range(len(grid_ids))):
            i = grid_ids[n]
            node_id = i+"_nodes"
            edge_id = i+"_edges"
            try:
                edge = gpd.read_file('LV/'+self.grid_dict[self.grids_name]+'/'+edge_id)
                edge_total = pd.concat([edge_total, edge], ignore_index=True)
            except:
                print("Error in reading the edge file "+edge_id)
            node = gpd.read_file('LV/'+self.grid_dict[self.grids_name]+'/'+node_id)
            node_total = pd.concat([node_total, node], ignore_index=True)
            
        return node_total, edge_total
    
    def voronoi(self, save_plot=False):
        el_dmd = self.import_demand()
        node_total, edge_total = self.concat_all_grids()
        coords = points_to_coords(el_dmd.geometry)
        coords_node = points_to_coords(node_total.geometry)
        node_total['res_percentage'] = -1
        node_total['com_percentage'] = -1

        # add the boundary nodes as input points to avoid the infinite regions
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        a1 = np.around(0.3*(x_max-x_min), decimals=0)
        a2 = np.around(0.3*(y_max-y_min), decimals=0)
        bbox = x_min-a1, x_max+a1, y_min-a2, y_max+a2
        for i in range(int(bbox[0]), int(bbox[1]), int(0.5*a1)):
            coords = np.append(coords, [[i, bbox[2]], [i, bbox[3]]], axis=0)
        for i in range(int(bbox[2]), int(bbox[3]), int(0.5*a2)):
            coords = np.append(coords, [[bbox[0], i], [bbox[1], i]], axis=0)

        # Do the voronoi partitioning
        vor = Voronoi(coords)
        regions = vor.regions
        vertices = vor.vertices
        point_region = vor.point_region

        fig, ax = plt.subplots(figsize=(6,6))
        edge_total.plot(ax=ax, color='black', linewidth=0.5)
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, point_size=1.5, line_colors='grey', line_width=0.3,
                        line_alpha=0.3,show_points=False)
        
        # assign the residential and commercial percentages to the nodes according to the voronoi diagram
        print("Assigning the residential and commercial percentages to the nodes...")
        for i in tqdm.tqdm(range(len(el_dmd))):
            el = el_dmd.iloc[i]
            cor = coords[i]
            res_percentage = el['residential_percentage']   
            com_percentage = el['commercial_percentage']
            region = regions[point_region[i]]
            color =  sns.color_palette("coolwarm", as_cmap=True)(res_percentage)
            # -1 is the index of the infinite region, remove it
            if -1 in region:
                region.remove(-1)
            region_vertices = vertices[region]
            for j in range(len(node_total)):
                node = node_total.iloc[j]
                if node['geometry'].within(Polygon(region_vertices)):
                    node_total.loc[j, 'res_percentage'] = res_percentage
                    node_total.loc[j, 'com_percentage'] = com_percentage
                    plt.plot(node.geometry.x, node.geometry.y, 'ko', markersize=1, color=color)
            plt.fill(*zip(*region_vertices), alpha=0.1, color=color)
        node_index = node_total[(node_total['res_percentage']==-1) | (node_total['com_percentage']==-1)].index
        for index in node_index:
            node = node_total.iloc[index]
            distance = []
            for k in range(len(el_dmd)):
                el = el_dmd.iloc[k]
                distance.append(node['geometry'].distance(el['geometry']))
            min_index = distance.index(min(distance))
            node_total.loc[index, 'res_percentage'] = el_dmd.iloc[min_index]['residential_percentage']
            node_total.loc[index, 'com_percentage'] = el_dmd.iloc[min_index]['commercial_percentage']
        sm = plt.cm.ScalarMappable(cmap=sns.color_palette("coolwarm", as_cmap=True), norm=plt.Normalize(vmin=0, vmax=1))
        sm._A = []
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
        cbar.set_label('residential percentage', rotation=270)
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_coords(-2.5, 0.5)
        cbar.ax.yaxis.set_tick_params(pad=0)
        cbar.ax.tick_params(labelsize=9)
        plt.axis('off')
        # create the folder if it does not exist "results"
        if not os.path.exists(self.output_path+'/results'):
            os.makedirs(self.output_path+'/results')
        if save_plot:
            plt.savefig(self.output_path+'/results/'+self.grids_name+'_voronoi.png', dpi=500, bbox_inches='tight')
            print("The voronoi diagram is saved in the results folder")
        return node_total
    
    def allocation_for_2nodes(self):
        el_dmd = self.import_demand()
        node_total, edge_total = self.concat_all_grids()
        for j in range(len(node_total)):
            node = node_total.iloc[j]
            distance = []
            for i in range(len(el_dmd)):
                # calculate the distance between the node and the every el_dmd, and assign the residential and commercial percentage of the nearest el_dmd to the node
                el = el_dmd.iloc[i]
                distance.append(node['geometry'].distance(el['geometry']))
            min_index = distance.index(min(distance))
            node_total.loc[j, 'res_percentage'] = el_dmd.iloc[min_index]['residential_percentage']
            node_total.loc[j, 'com_percentage'] = el_dmd.iloc[min_index]['commercial_percentage']
        return node_total
    
    def save_allocation(self,save_plot=False):
        if len(self.grid_ids) > 2:
            node_total = self.voronoi(save_plot=save_plot)
        else:
            node_total = self.allocation_for_2nodes()
        grid_ids = self.grid_ids
        for iter, i in enumerate(grid_ids):
            n = len(grid_ids)
            node_id = i+"_nodes"
            node = gpd.read_file('LV/'+self.grid_dict[self.grids_name]+'/'+node_id)
            node['res_percentage'] = -1
            node['com_percentage'] = -1
            for j in range(len(node)):
                # find the index of the node in the node_total and assign the residential and commercial percentages to the node
                index_node = node_total[node_total['x']==node.loc[j,'x']][node_total['y']==node.loc[j,'y']].index
                node.loc[j,'res_percentage'] = node_total.loc[index_node,'res_percentage'].values[0]
                node.loc[j,'com_percentage'] = node_total.loc[index_node,'com_percentage'].values[0]
            node.to_file('LV/'+self.grid_dict[self.grids_name]+'/'+node_id, driver='GeoJSON')
            print("Successfully saved the allocation for grid "+i+" ("+str(iter+1)+"/"+str(n)+")")
            

    def transform_from_pkl(self):
        path = 'Synthetic_networks/Demand_calculator/municipality_normalized_profiles.pkl'
        profiles = pd.read_pickle(path)
        keys = profiles.keys()
        commercial = pd.DataFrame()
        residential = pd.DataFrame()
        error=[]
        # columns is keys
        for key in keys:
            if 'commercial' not in profiles[key].keys():
                profiles[key]['commercial'] = np.zeros(8760)
            elif 'residential' not in profiles[key].keys():
                profiles[key]['residential'] = np.zeros(8760)
            try:
                commercial[key] = profiles[key]['commercial']
                residential[key] = profiles[key]['residential']
            except:
                error.append(key)
                print(key)
                continue
        commercial.to_csv('Synthetic_networks/Demand_calculator/commercial_profiles.csv', index=False)
        residential.to_csv('Synthetic_networks/Demand_calculator/residential_profiles.csv', index=False)

if __name__ == "__main__":
    la = load_allocation()
    with open(la.output_path+'/municipality_profiles/dict_folder.json') as f:
        dict_folder = json.load(f)
    la.grid_dict = dict_folder
    keys = list(dict_folder.keys())
    for key in keys[0:1]:    
        len_dict = len(dict_folder)
        print("Processing grid "+key+" ("+str(list(dict_folder.keys()).index(key)+1)+"/"+str(len_dict)+")")
        la.grids_name = key
        la.save_id()
        save_plot = True
        la.save_allocation(save_plot=save_plot)
    

        

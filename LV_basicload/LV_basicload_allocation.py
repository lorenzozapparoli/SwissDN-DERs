"""
Author: Lorenzo Zapparoli
Institution: ETH Zurich
Date: 15/03/2025

Introduction:
This script, `LV_basicload_allocation.py`, is designed to allocate the percentage of residential and commercial electricity demand to each node in a Low Voltage (LV) grid. The allocation is performed using Voronoi partitioning or proximity-based methods for grids with fewer than three nodes. The script processes multiple grids in parallel using MPI, ensuring efficient handling of large datasets.

The script reads demand data, calculates residential and commercial percentages, and assigns these percentages to grid nodes. The results are saved in GeoJSON format, and Voronoi diagrams can be optionally saved as visualizations.

Usage:
1. Ensure the required input files (demand data and grid node files) are available in the specified directories.
2. Run the script using MPI to process the grids in parallel.
3. The output files will be saved in the `Grids/LV` directory.

Dependencies:
- pandas
- numpy
- geopandas
- shapely
- scipy.spatial.Voronoi
- geovoronoi
- mpi4py
- matplotlib
- seaborn
- tqdm
- json
- os
- warnings
"""

import pandas as pd
from mpi4py import MPI
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
    'figure.figsize': [3.5, 2.5],  # Size of the figure (width x height) in inches
    'savefig.dpi': 300,        # Resolution of the output figure
    'legend.loc': 'best',      # Location of the legend
    'legend.frameon': False,   # Remove the box frame around legends
    'pdf.fonttype': 42,        # Embedding fonts in PDF for compatibility
    'ps.fonttype': 42
})


class load_allocation:
    def __init__(self):
        """
        Initializes the `load_allocation` class.

        Description:
        - Sets up paths for input and output directories.
        - Initializes variables for grid names, IDs, and building data.
        """
        self.grids_name = str
        self.grid_ids = list
        self.grid_dict = dict
        self.buildings = gpd.GeoDataFrame()
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.base_path = os.path.dirname(self.path)
        self.lv_grids_path = os.path.join(self.base_path, 'Grids', 'LV')
        self.dict_path = os.path.join(self.base_path, 'Grids', 'Additional_files')
        self.input_path = os.path.join(self.path, 'LV_basicload_input')
        self.output_path = os.path.join(self.path, 'LV_basicload_output')
                
    def create_dict(self,save_dict=False):
        """
        Creates a dictionary mapping grid files to their corresponding folders.

        Args:
            save_dict (bool): Whether to save the dictionary as a JSON file.

        Returns:
            dict: Dictionary mapping grid file names to folder names.

        Description:
        - Iterates through the input directory to map grid files to folders.
        - Optionally saves the dictionary to a JSON file.
        """

        path = self.input_path+'\\Square_zones_dmd\\'
        list_folder = os.listdir(path)
        dict_folder = {}
        for folder in list_folder:
            path_folder = path+folder+'\\'
            list_files = os.listdir(path_folder)
            for file in list_files:
                file_name = file.split('_')[0]
                dict_folder[file_name] = folder
                # sort the dictionary by the keys
                dict_folder = dict(sorted(dict_folder.items()))
        if save_dict:
            with open(os.path.join(self.dict_path, 'dict_folder.json'), 'w') as f:
                json.dump(dict_folder, f)
        return dict_folder

    def save_id(self):
        """
        Saves the grid IDs for the current grid.

        Returns:
            list: List of unique grid IDs.

        Description:
        - Extracts grid IDs from the file names in the grid directory.
        """

        path = os.path.join(self.lv_grids_path, self.grid_dict[self.grids_name])
        grid_ids = list(set([str(f.split('.')[0][:-6]) for f in os.listdir(path) if f.startswith(self.grids_name+'-')]))
        self.grid_ids = grid_ids
        return grid_ids

    def import_demand(self):
        """
        Imports and preprocesses electricity demand data.

        Returns:
            GeoDataFrame: GeoDataFrame containing demand data with residential and commercial percentages.

        Description:
        - Reads demand data from a GeoJSON file.
        - Calculates residential and commercial percentages for each demand point.
        """

        el_dmd_name = self.input_path+'\\Square_zones_dmd\\'+self.grid_dict[self.grids_name]+'\\'+self.grids_name+'_el_dmd.geojson'
        el_dmd = gpd.read_file(el_dmd_name)
        el_dmd['residential_percentage'] = el_dmd['residential']/(el_dmd['residential']+el_dmd['commercial'])
        el_dmd['commercial_percentage'] = el_dmd['commercial']/(el_dmd['residential']+el_dmd['commercial'])
        el_dmd['residential_percentage'] = el_dmd['residential_percentage'].fillna(0)
        el_dmd['commercial_percentage'] = el_dmd['commercial_percentage'].fillna(0)
        return el_dmd

    def concat_all_grids(self):
        """
        Concatenates all nodes and edges of the grids.

        Returns:
            Tuple: GeoDataFrames for concatenated nodes and edges.

        Description:
        - Reads and combines node and edge data for all grids.
        """

        grid_ids = self.grid_ids
        node_total = gpd.GeoDataFrame()
        edge_total = gpd.GeoDataFrame()
        # add timebar to show the progress using tqdm
        # print("Concatenating all the nodes and edges of the grids...")
        for n in range(len(grid_ids)):
            i = grid_ids[n]
            node_id = i+"_nodes"
            edge_id = i+"_edges"
            try:
                edge = gpd.read_file(os.path.join(self.lv_grids_path, self.grid_dict[self.grids_name], edge_id))
                edge_total = pd.concat([edge_total, edge], ignore_index=True)
            except:
                print("Error in reading the edge file "+edge_id)
            node = gpd.read_file(os.path.join(self.lv_grids_path, self.grid_dict[self.grids_name], node_id))
            node_total = pd.concat([node_total, node], ignore_index=True)

        return node_total, edge_total

    def voronoi(self, save_plot=False):
        """
        Performs Voronoi partitioning and assigns demand percentages to grid nodes.

        Args:
            save_plot (bool): Whether to save the Voronoi diagram as a PNG file.

        Returns:
            GeoDataFrame: GeoDataFrame of nodes with assigned residential and commercial percentages.

        Description:
        - Creates a Voronoi diagram based on demand points.
        - Assigns residential and commercial percentages to nodes based on Voronoi regions.
        - Optionally saves the Voronoi diagram as a visualization.
        """

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
        # print("Assigning the residential and commercial percentages to the nodes...")
        for i in range(len(el_dmd)):
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
        if not os.path.exists(self.output_path+'\\results'):
            os.makedirs(self.output_path+'\\results')
        if save_plot:
            plt.savefig(self.output_path+'\\results\\'+self.grids_name+'_voronoi.png', dpi=500, bbox_inches='tight')
            # print("The voronoi diagram is saved in the results folder")
        return node_total

    def allocation_for_2nodes(self):
        """
        Allocates demand percentages to nodes for grids with two or fewer nodes.

        Returns:
            GeoDataFrame: GeoDataFrame of nodes with assigned residential and commercial percentages.

        Description:
        - Assigns demand percentages to nodes based on proximity to demand points.
        """

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
        """
        Saves the allocation results for each grid.

        Args:
            save_plot (bool): Whether to save the Voronoi diagram as a PNG file.

        Description:
        - Performs Voronoi partitioning or proximity-based allocation.
        - Saves the updated node data to GeoJSON files.
        """

        if len(self.grid_ids) > 2:
            node_total = self.voronoi(save_plot=save_plot)
        else:
            node_total = self.allocation_for_2nodes()
        grid_ids = self.grid_ids
        for iter, i in enumerate(grid_ids):
            n = len(grid_ids)
            node_id = i+"_nodes"
            node = gpd.read_file(os.path.join(self.lv_grids_path, self.grid_dict[self.grids_name], node_id))
            node['res_percentage'] = -1
            node['com_percentage'] = -1
            for j in range(len(node)):
                # find the index of the node in the node_total and assign the residential and commercial percentages to the node
                index_node = node_total[node_total['x']==node.loc[j,'x']][node_total['y']==node.loc[j,'y']].index
                node.loc[j,'res_percentage'] = node_total.loc[index_node,'res_percentage'].values[0]
                node.loc[j,'com_percentage'] = node_total.loc[index_node,'com_percentage'].values[0]
            node.to_file(os.path.join(self.lv_grids_path, self.grid_dict[self.grids_name], node_id), driver='GeoJSON')


if __name__ == "__main__":
    """
    Main execution block.

    Description:
    - Initializes MPI and distributes grids among processes.
    - Processes each grid to allocate demand percentages to nodes.
    - Saves the results in GeoJSON format.
    """

    la = load_allocation()
    with open(os.path.join(la.dict_path, 'dict_folder.json')) as f:
        dict_folder = json.load(f)
    la.grid_dict = dict_folder
    keys = list(dict_folder.keys())

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Distribute keys among processes
    keys_per_process = len(keys) // size
    start = rank * keys_per_process
    end = (rank + 1) * keys_per_process if rank != size - 1 else len(keys)

    for key in keys[start:end]:
        len_dict = len(dict_folder)
        print(f"Processing grid {key} ({keys.index(key)+1}\\{len_dict}) on rank {rank}")
        la.grids_name = key
        la.save_id()
        save_plot = False
        la.save_allocation(save_plot=save_plot)

    # Finalize MPI
    MPI.Finalize()

import os
import json
import pandas as pd
import geopandas as gpd
from datetime import datetime
import calendar
import zipfile
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import contextily as ctx
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.widgets import RectangleSelector

class GridLoader:
    def __init__(self, _grid_type, _grid_name, _start_date, _end_date, _data_year):
        """
        Initialize the GridLoader with the type of grid (MV or LV), grid name, time window, and data year.

        :param _grid_type: Type of the grid ('MV' or 'LV')
        :param _grid_name: Name of the grid
        :param _start_date: Start date-time for loading data (format: 'MM-DD HH:00:00')
        :param _end_date: End date-time for loading data (format: 'MM-DD HH:00:00')
        :param _data_year: Data year (2030, 2040, or 2050)
        """
        self.grid_type = _grid_type
        self.grid_name = _grid_name
        self.start_date = datetime.strptime(_start_date, '%m-%d %H:%M:%S')
        self.end_date = datetime.strptime(_end_date, '%m-%d %H:%M:%S')
        self.data_year = _data_year
        self.script_path = os.path.join(os.path.dirname(os.getcwd()), 'SwissPDG_DERs')
        self.edges_file = None
        self.nodes_file = None
        self.load_profiles = None
        self.mv_load_profiles = None
        self.lv_load_profiles = None
        self.bess_allocation = None
        self.ev_energy_profiles = None
        self.ev_flexible_energy_profiles = None
        self.ev_power_profiles = None
        self.ev_allocation = None
        self.temperature_profiles = None
        self.hp_allocation = None
        self.pv_std = None
        self.pv_p_installed = None
        self.pv_generation = None
        self.nodes = None
        self.edges = None
        self.validate_data_year()
        self.color_palette = ["#ffd45b", "#f6b44d", "#db9671", "#ba7a8d", "#8d5eaa", "#3940bb", "#0033b0", "#B4E5A2"]

    def validate_data_year(self):
        """
        Validate the data year.
        """
        if self.data_year not in [2030, 2040, 2050]:
            raise ValueError("Invalid data year. Please specify 2030, 2040, or 2050.")

    def load_grid(self):
        """
        Load the grid data based on the grid type.
        """
        if self.grid_type == 'LV':
            """
            Load the LV grid data based on the grid name.
            """
            lv_folder_path = os.path.join(self.script_path, '06_Grids')
            dict_file_path = os.path.join(lv_folder_path, 'dict_folder.json')

            # Load the dictionary mapping municipality codes to folder names
            with open(dict_file_path, 'r') as f:
                folder_dict = json.load(f)

            # Extract the municipality code from the grid name
            municipality_code = self.grid_name.split('-')[0]

            # Find the corresponding folder name
            folder_name = folder_dict.get(municipality_code)
            if not folder_name:
                raise FileNotFoundError(f"Folder for municipality code {municipality_code} not found.")

            # Construct the path to the zipped folder
            zip_file_path = os.path.join(lv_folder_path, f"LV.zip")

            # Load the edges and nodes files from the zip
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                with zip_ref.open(f"LV/{folder_name}/{self.grid_name}_edges") as edges_file:
                    self.edges = gpd.read_file(edges_file)
                with zip_ref.open(f"LV/{folder_name}/{self.grid_name}_nodes") as nodes_file:
                    self.nodes = gpd.read_file(nodes_file)

        elif self.grid_type == 'MV':
            """
            Load the MV grid data based on the grid name.
            """
            mv_folder_path = os.path.join(self.script_path, '06_Grids')

            folder_name = 'MV'
            # Construct the path to the zipped folder
            zip_file_path = os.path.join(mv_folder_path, f"{folder_name}.zip")

            # Load the edges and nodes files from the zip
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                with zip_ref.open(f"{folder_name}/{self.grid_name}_edges") as edges_file:
                    self.edges = gpd.read_file(edges_file)
                with zip_ref.open(f"{folder_name}/{self.grid_name}_nodes") as nodes_file:
                    self.nodes = gpd.read_file(nodes_file)
        else:
            raise ValueError("Invalid grid type. Please specify 'MV' or 'LV'.")

    def load_pv_data(self):
        """
        Load and process the PV data based on the grid type and time window.
        """
        base_path = os.path.join(self.script_path, '01_PV', str(self.data_year))
        if self.grid_type == 'LV':
            files = ['LV_generation.csv', 'LV_P_installed.csv', 'LV_std.csv']
        else:
            files = ['MV_generation.csv', 'MV_P_installed.csv', 'MV_std.csv']

        dataframes = {}
        for file in files:
            file_path = os.path.join(base_path, file)
            if not os.path.exists(file_path):
                print(f"File {file} not found in {base_path}.")
                dataframes[file] = None
                continue

            df = pd.read_csv(file_path)
            df_filtered = df[df['LV_grid'] == self.grid_name] if self.grid_type == 'LV' else df[
                df['MV_grid'] == self.grid_name]
            if df_filtered.empty:
                print(f"No PV allocation found for grid {self.grid_name} in {file}.")
                dataframes[file] = None
            else:
                if file != 'LV_P_installed.csv' and file != 'MV_P_installed.csv':
                    dataframes[file] = self.filter_time_interval_pv(df_filtered)
                else:
                    dataframes[file] = df_filtered

        self.pv_generation = dataframes['LV_generation.csv'] if self.grid_type == 'LV' else dataframes[
            'MV_generation.csv']
        self.pv_p_installed = dataframes['LV_P_installed.csv'] if self.grid_type == 'LV' else dataframes[
            'MV_P_installed.csv']
        self.pv_std = dataframes['LV_std.csv'] if self.grid_type == 'LV' else dataframes['MV_std.csv']

    def filter_time_interval_pv(self, df):
        """
        Filter the dataframe to create a yearly profile with 8760 entries and keep only the rows within the specified time interval.

        :param df: DataFrame to filter
        :return: Filtered DataFrame
        """
        time_columns = df.columns[2:]
        expanded_data = {}

        for col in time_columns:
            month_day = col.split()[0]
            hour = col.split()[1]
            month = int(month_day.split('-')[0])
            int(month_day.split('-')[1])
            days_in_month = calendar.monthrange(self.start_date.year, month)[1]
            for day_of_month in range(1, days_in_month + 1):
                new_col = f"{month:02d}-{day_of_month:02d} {hour}"
                expanded_data[new_col] = df[col].values

        expanded_df = pd.DataFrame(expanded_data)
        if self.grid_type == 'LV':
            expanded_df.insert(0, 'LV_grid', df['LV_grid'].values)
            expanded_df.insert(1, 'LV_osmid', df['LV_osmid'].values)
        else:
            expanded_df.insert(0, 'MV_grid', df['MV_grid'].values)
            expanded_df.insert(1, 'MV_osmid', df['MV_osmid'].values)

        # Filter the expanded dataframe based on the specified time interval
        filtered_columns = [col for col in expanded_df.columns[2:] if
                            self.start_date <= datetime.strptime(col, '%m-%d %H:%M:%S') <= self.end_date]
        # Sort the filtered columns by datetime values
        sorted_filtered_columns = sorted(filtered_columns, key=lambda x: datetime.strptime(x, '%m-%d %H:%M:%S'))

        return expanded_df[['LV_grid', 'LV_osmid'] + sorted_filtered_columns] if self.grid_type == 'LV' else \
            expanded_df[['MV_grid', 'MV_osmid'] + sorted_filtered_columns]

    def load_hp_data(self):
        """
        Load and process the heat pumps data based on the grid type and time window.
        """
        base_path = os.path.join(self.script_path, '03_HP', str(self.data_year))
        allocation_file = 'LV_heat_pump_allocation.csv' if self.grid_type == 'LV' else 'MV_heat_pump_allocation.csv'
        temperature_file = 'Temperature_profiles.csv'

        allocation_path = os.path.join(base_path, allocation_file)
        temperature_path = os.path.join(base_path, temperature_file)

        if not os.path.exists(allocation_path) or not os.path.exists(temperature_path):
            raise FileNotFoundError(f"Required files not found in {base_path}.")

        # Load the allocation and temperature profiles files
        allocation_df = pd.read_csv(allocation_path)
        temperature_df = pd.read_csv(temperature_path)

        # Filter the allocation DataFrame
        grid_column = 'LV_grid' if self.grid_type == 'LV' else 'MV_grid'
        filtered_allocation_df = allocation_df[allocation_df[grid_column] == self.grid_name]

        if filtered_allocation_df.empty:
            print(f"No heat pump allocation found for grid {self.grid_name}.")
            self.hp_allocation = None
            self.temperature_profiles = None
            return 0

        # Filter the temperature profiles DataFrame
        profile_names = filtered_allocation_df['Temperature_profile_name'].unique()
        filtered_temperature_df = temperature_df[temperature_df['Temperature_profile_name'].isin(profile_names)]

        # Filter the columns based on the specified time interval
        date_columns = [col for col in filtered_temperature_df.columns if col != 'Temperature_profile_name']
        filtered_columns = [col for col in date_columns if
                            self.start_date <= datetime.strptime(col, '%m-%d %H:%M:%S') <= self.end_date]

        # Create the temperature profiles DataFrame
        temperature_profiles = filtered_temperature_df[['Temperature_profile_name'] + filtered_columns]

        self.hp_allocation = filtered_allocation_df
        self.temperature_profiles = temperature_profiles

    def load_ev_data(self):
        """
        Load and process the electric vehicles data for LV grids based on the time window.
        """
        if self.grid_type != 'LV':
            print("EV data can only be loaded for LV grids.")
            self.ev_allocation = None
            self.ev_energy_profiles = None
            self.ev_flexible_energy_profiles = None
            self.ev_power_profiles = None
            return

        base_path = os.path.join(self.script_path, '04_EV', str(self.data_year))
        files = {
            'allocation': 'EV_allocation_LV.csv',
            'flexible_energy_profiles': 'EV_flexible_energy_profiles_LV.csv',
            'power_profiles': 'EV_power_profiles_LV.csv'
        }

        dataframes = {}
        for key, file in files.items():
            file_path = os.path.join(base_path, file)
            if not os.path.exists(file_path):
                print(f"File {file} not found in {base_path}.")
                dataframes[key] = None
                continue
            dataframes[key] = pd.read_csv(file_path)

        # Filter the allocation DataFrame
        allocation_df = dataframes['allocation']
        if allocation_df is not None:
            filtered_allocation_df = allocation_df[allocation_df['LV_grid'] == self.grid_name]
            if filtered_allocation_df.empty:
                print(f"No EV allocation found for grid {self.grid_name}.")
                self.ev_allocation = None
                self.ev_energy_profiles = None
                self.ev_flexible_energy_profiles = None
                self.ev_power_profiles = None
                return
            self.ev_allocation = filtered_allocation_df
        else:
            self.ev_allocation = None

        # Extract BFS municipality code
        bfs_code = self.grid_name.split('-')[0]

        # Filter the other DataFrames based on BFS municipality code
        for key in ['flexible_energy_profiles', 'power_profiles']:
            df = dataframes[key]
            if df is not None:
                filtered_df = df[df['BFS_municipality_code'] == int(bfs_code)]
                if filtered_df.empty:
                    print(f"No data found for BFS municipality code {bfs_code} in {key}.")
                    dataframes[key] = None
                else:
                    dataframes[key] = filtered_df
            else:
                dataframes[key] = None

        self.ev_flexible_energy_profiles = dataframes['flexible_energy_profiles']
        self.ev_power_profiles = dataframes['power_profiles']

        # Filter the time range for EV_power_profiles_LV
        if self.ev_power_profiles is not None:
            date_columns = [col for col in self.ev_power_profiles.columns if
                            (col != 'BFS_municipality_code' and col != 'Profile_type')]
            filtered_columns = [col for col in date_columns if
                                self.start_date <= datetime.strptime(col, '%m-%d %H:%M:%S') <= self.end_date]
            self.ev_power_profiles = self.ev_power_profiles[
                ['BFS_municipality_code', 'Profile_type'] + filtered_columns]

        # Filter the time range for EV_flexible_energy_profiles_LV
        if self.ev_flexible_energy_profiles is not None:
            date_columns = [col for col in self.ev_flexible_energy_profiles.columns if
                            col != 'BFS_municipality_code']
            filtered_columns = [col for col in date_columns if
                                self.start_date <= datetime.strptime(col, '%m-%d') <= self.end_date]
            self.ev_flexible_energy_profiles = self.ev_flexible_energy_profiles[
                ['BFS_municipality_code'] + filtered_columns]

    def load_bess_data(self):
        """
        Load and process the BESS data based on the grid type.
        """
        base_path = os.path.join(self.script_path, '02_BESS', str(self.data_year))
        allocation_file = 'BESS_allocation_LV.csv' if self.grid_type == 'LV' else 'BESS_allocation_MV.csv'
        allocation_path = os.path.join(base_path, allocation_file)

        if not os.path.exists(allocation_path):
            print(f"File {allocation_file} not found in {base_path}.")
            self.bess_allocation = None
            return

        # Load the allocation file
        allocation_df = pd.read_csv(allocation_path)

        # Filter the allocation DataFrame
        grid_column = 'LV_grid' if self.grid_type == 'LV' else 'MV_grid'
        filtered_allocation_df = allocation_df[allocation_df[grid_column] == self.grid_name]

        if filtered_allocation_df.empty:
            print(f"No BESS allocation found for grid {self.grid_name}.")
            self.bess_allocation = None
        else:
            self.bess_allocation = filtered_allocation_df

    def load_conventional_load_profiles(self):
        """
        Load and process the conventional load profiles based on the grid type and time window.
        """
        if self.grid_type == 'LV':
            base_path = os.path.join(self.script_path, '05_Demand', str(self.data_year))
            files = {
                'commercial_profiles': 'Commercial_profiles.csv',
                'basicload_shares': 'LV_basicload_shares.csv',
                'residential_profiles': 'Residential_profiles.csv'
            }

            dataframes = {}
            for key, file in files.items():
                file_path = os.path.join(base_path, file)
                if not os.path.exists(file_path):
                    print(f"File {file} not found in {base_path}.")
                    dataframes[key] = None
                    continue
                dataframes[key] = pd.read_csv(file_path)

            # Filter the basicload shares DataFrame
            basicload_shares_df = dataframes['basicload_shares']
            if basicload_shares_df is not None:
                filtered_shares_df = basicload_shares_df[basicload_shares_df['LV_grid'] == self.grid_name]
                if filtered_shares_df.empty:
                    print(f"No basic load shares found for grid {self.grid_name}.")
                    self.lv_load_profiles = None
                    return
            else:
                self.lv_load_profiles = None
                return

            # Extract BFS municipality code
            bfs_code = self.grid_name.split('-')[0]

            # Filter the commercial and residential profiles based on BFS municipality code
            commercial_profiles_df = dataframes['commercial_profiles']
            residential_profiles_df = dataframes['residential_profiles']
            if commercial_profiles_df is not None and residential_profiles_df is not None:
                commercial_profile = commercial_profiles_df[
                    commercial_profiles_df['BFS_municipality_code'] == int(bfs_code)]
                residential_profile = residential_profiles_df[
                    residential_profiles_df['BFS_municipality_code'] == int(bfs_code)]
                if commercial_profile.empty or residential_profile.empty:
                    print(f"No profiles found for BFS municipality code {bfs_code}.")
                    self.lv_load_profiles = None
                    return
            else:
                self.lv_load_profiles = None
                return

            # Create the new DataFrame with weighted average profiles
            load_profiles_data = {'LV_grid': filtered_shares_df['LV_grid'].values,
                                  'LV_osmid': filtered_shares_df['LV_osmid'].values}
            for col in commercial_profile.columns[1:]:
                commercial_load = commercial_profile[col].values
                residential_load = residential_profile[col].values
                commercial_share = filtered_shares_df['Commercial_demand_share'].values
                residential_share = filtered_shares_df['Residential_demand_share'].values
                load_profiles_data[col] = commercial_load * commercial_share + residential_load * residential_share

            load_profiles_df = pd.DataFrame(load_profiles_data)

            # Filter the time steps within the time interval of interest
            date_columns = [col for col in load_profiles_df.columns[2:] if
                            self.start_date <= datetime.strptime(col, '%m-%d %H:%M:%S') <= self.end_date]
            load_profiles_df = load_profiles_df[['LV_grid', 'LV_osmid'] + date_columns]

            self.load_profiles = load_profiles_df

        elif self.grid_type == 'MV':
            base_path = os.path.join(self.script_path, '05_Demand', str(self.data_year))
            file_path = os.path.join(base_path, 'MV_load_profile.csv')
            if not os.path.exists(file_path):
                print(f"File MV_load_profile.csv not found in {base_path}.")
                self.mv_load_profiles = None
                return

            # Load the MV load profile
            with open(file_path, 'r') as f:
                lines = f.readlines()
                timestamps = lines[0].strip().split(',')
                values = lines[1].strip().split(',')

            # Convert timestamps to datetime objects
            timestamps = [datetime.strptime(ts, '%m-%d %H:%M:%S') for ts in timestamps]

            # Create a DataFrame
            mv_load_profile_df = pd.DataFrame({
                'Timestamp': timestamps,
                'Value': values
            })

            # Load the MV nodes file to get the osmid values
            if self.nodes is None:
                self.load_grid()
            nodes_df = self.nodes.copy()
            nodes_df = nodes_df[nodes_df['lv_grid'] == '-1']

            # Create the new DataFrame with the load profile for each node
            load_profiles_data = {'MV_grid': [self.grid_name] * len(nodes_df), 'MV_osmid': nodes_df['osmid'].values}
            for i, ts in enumerate(timestamps):
                load_profiles_data[ts.strftime('%m-%d %H:%M:%S')] = mv_load_profile_df['Value'].iloc[i]
            load_profiles_df = pd.DataFrame(load_profiles_data)

            # Filter the time steps within the time interval of interest
            filtered_columns = [col for col in load_profiles_df.columns[2:] if
                                self.start_date <= datetime.strptime(col, '%m-%d %H:%M:%S') <= self.end_date]
            load_profiles_df = load_profiles_df[['MV_grid', 'MV_osmid'] + filtered_columns]
            self.load_profiles = load_profiles_df

    def add_fields_to_nodes(self):
        # Initialize new fields with default values
        self.nodes['PV_capacity'] = 0
        self.nodes['HP_capacity'] = 0
        self.nodes['BESS_capacity'] = 0
        self.nodes['EV_peak_power'] = 0

        # Ensure osmid field is of type int
        self.nodes['osmid'] = self.nodes['osmid'].astype(int)

        # Add PV_capacity
        if self.pv_p_installed is not None and 'LV_osmid' in self.pv_p_installed.columns:
            self.pv_p_installed['LV_osmid'] = self.pv_p_installed['LV_osmid'].astype(int)
            pv_dict = self.pv_p_installed.set_index('LV_osmid')['P_installed_kW'].to_dict()
            self.nodes['PV_capacity'] = self.nodes['osmid'].map(pv_dict).fillna(0)

        # Add HP_capacity
        if self.hp_allocation is not None and 'LV_osmid' in self.hp_allocation.columns:
            self.hp_allocation['LV_osmid'] = self.hp_allocation['LV_osmid'].astype(int)
            hp_dict = self.hp_allocation.set_index('LV_osmid')['Nominal_power_kW'].to_dict()
            self.nodes['HP_capacity'] = self.nodes['osmid'].map(hp_dict).fillna(0)

        # Add BESS_capacity
        if self.bess_allocation is not None and 'LV_osmid' in self.bess_allocation.columns:
            self.bess_allocation['LV_osmid'] = self.bess_allocation['LV_osmid'].astype(int)
            bess_dict = self.bess_allocation.set_index('LV_osmid')['Nominal_power_kW'].to_dict()
            self.nodes['BESS_capacity'] = self.nodes['osmid'].map(bess_dict).fillna(0)

        # Add EV_peak_power
        if self.ev_allocation is not None and 'LV_osmid' in self.ev_allocation.columns and self.ev_power_profiles is not None:
            self.ev_allocation['LV_osmid'] = self.ev_allocation['LV_osmid'].astype(int)
            max_ev_power = self.ev_power_profiles.iloc[:, 2:].max().max()
            ev_dict = self.ev_allocation.set_index('LV_osmid')['EV_share'].to_dict()
            self.nodes['EV_peak_power'] = self.nodes['osmid'].map(ev_dict).fillna(0) * max_ev_power

        # Save the updated nodes file
        output_path = os.path.join(os.getcwd(), 'Grid_data', f'Updated_{self.grid_name}_nodes.geojson')
        self.nodes.to_file(output_path, driver='GeoJSON')

    def plot_edges(self, ax):
        for _, edge in self.edges.iterrows():
            x_coords, y_coords = zip(*edge.geometry.coords)
            ax.plot(x_coords, y_coords, 'k-', linewidth=2)

    def plot_pie_chart(self, ax, sizes, x, y, total_power):
        if all(size == 0 for size in sizes):
            ax.plot(x, y, 'ko', markersize=5)
        else:
            size_factor = total_power
            pie_size = np.sqrt(size_factor) * 0.05
            pie_ax = inset_axes(ax, width=pie_size, height=pie_size, loc='center',
                       bbox_to_anchor=(x, y), bbox_transform=ax.transData,
                       borderpad=0)
            # pie_ax = ax.figure.add_axes([x_norm, y_norm, pie_size, pie_size])
            pie_ax.pie(sizes, startangle=90, colors=[self.color_palette[0], self.color_palette[-1], self.color_palette[2],
                                                     self.color_palette[4], self.color_palette[6]], wedgeprops={'edgecolor': 'black', 'linewidth': 0.5})
            pie_ax.axis('equal')

    def plot_grid(self, zoom_coords=None):
        self.nodes = gpd.read_file(os.path.join('Grid_data', f'Updated_{self.grid_name}_nodes.geojson'))
        categories = ['PV_capacity', 'HP_capacity', 'EV_peak_power', 'el_dmd', 'BESS_capacity']
        self.nodes['total_power'] = self.nodes[categories].sum(axis=1)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(self.nodes['x'].min() - 50, self.nodes['x'].max() + 50)
        ax.set_ylim(self.nodes['y'].min() - 50, self.nodes['y'].max() + 50)
        ax.set_aspect('equal', 'box')  # Set aspect ratio to be equal
        self.plot_edges(ax)
        for _, node in self.nodes.iterrows():
            sizes = [
                node['PV_capacity'],
                node['BESS_capacity'],
                node['HP_capacity'],
                node['EV_peak_power'],
                node['el_dmd'] * 1000
            ]
            total_power = node['total_power']
            self.plot_pie_chart(ax, sizes, node['x'], node['y'], total_power)
        ctx.add_basemap(ax, crs=self.nodes.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik, zoom=17)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_frame_on(False)

        # Add legend above the graph
        labels = ['PV', 'BESS', 'HP', 'EV', 'Demand']
        handles = [mpatches.Patch(color=color) for color in
                   [self.color_palette[0], self.color_palette[-1], self.color_palette[2], self.color_palette[4], self.color_palette[6]]]
        ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=5, frameon=False)

        # Add pie size legend outside the plot area
        legend_ax = fig.add_axes([0.80, 0.33, 0.3, 0.3])  # Adjust the position and size as needed
        legend_ax.set_xlim(0, 1)
        legend_ax.set_ylim(0, 1)
        legend_ax.axis('off')
        sizes = [100, 10, 1]
        max_power = max(self.nodes['total_power'])
        for i, size in enumerate(sizes):
            circle_radius = np.sqrt(size / max_power) * 0.1  # Adjust the scaling factor as needed
            circle = plt.Circle((0.5, 0.8 - i * 0.2), circle_radius, color='grey', fill=True)
            legend_ax.add_patch(circle)
            legend_ax.text(0.63, 0.8 - i * 0.2, f'{size} kW', verticalalignment='center')

        plt.savefig(os.path.join(os.getcwd(), 'Grid_figures', f'Pies_grid_{self.grid_name}.svg'), format='svg',
                    bbox_inches='tight')
        plt.show()

        if zoom_coords:
            x1, y1, x2, y2 = zoom_coords
            x1 = self.nodes['x'].min() + x1 * (self.nodes['x'].max() - self.nodes['x'].min())
            y1 = self.nodes['y'].min() + y1 * (self.nodes['y'].max() - self.nodes['y'].min())
            x2 = self.nodes['x'].min() + x2 * (self.nodes['x'].max() - self.nodes['x'].min())
            y2 = self.nodes['y'].min() + y2 * (self.nodes['y'].max() - self.nodes['y'].min())
            fig_zoom, ax_zoom = plt.subplots(figsize=(10, 10))
            ax_zoom.set_xlim(x1, x2)
            ax_zoom.set_ylim(y1, y2)
            ax_zoom.set_aspect('equal', 'box')
            self.plot_edges(ax_zoom)
            for _, node in self.nodes.iterrows():
                if x1 <= node['x'] <= x2 and y1 <= node['y'] <= y2:
                    sizes = [
                        node['PV_capacity'],
                        node['HP_capacity'],
                        node['EV_peak_power'],
                        node['el_dmd'] * 1000,
                        node['BESS_capacity']
                    ]
                    total_power = node['total_power']
                    self.plot_pie_chart(ax_zoom, sizes, node['x'], node['y'], total_power)
            ctx.add_basemap(ax_zoom, crs=self.nodes.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik, zoom=19)
            ax_zoom.set_xticks([])
            ax_zoom.set_yticks([])
            ax_zoom.set_xlabel('')
            ax_zoom.set_ylabel('')
            ax_zoom.set_frame_on(False)
            # Add legend above the graph
            labels = ['PV', 'HP', 'EV', 'Demand', 'BESS']
            handles = [mpatches.Patch(color=color) for color in
                       [self.color_palette[0], self.color_palette[2], self.color_palette[4], 'grey',
                        self.color_palette[6]]]
            ax_zoom.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=5, frameon=False)

            # Add pie size legend outside the plot area
            legend_ax = fig.add_axes([0.80, 0.33, 0.3, 0.3])  # Adjust the position and size as needed
            legend_ax.set_xlim(0, 1)
            legend_ax.set_ylim(0, 1)
            legend_ax.axis('off')
            sizes = [100, 10, 1]
            max_power = max(self.nodes['total_power'])
            for i, size in enumerate(sizes):
                circle_radius = np.sqrt(size / max_power) * 0.1  # Adjust the scaling factor as needed
                circle = plt.Circle((0.5, 0.8 - i * 0.2), circle_radius, color='grey', fill=True)
                legend_ax.add_patch(circle)
                legend_ax.text(0.63, 0.8 - i * 0.2, f'{size} kW', verticalalignment='center')

            plt.savefig(os.path.join(os.getcwd(), 'Grid_figures', f'Pies_grid_{self.grid_name}_zoomed.svg'),
                        format='svg',
                        bbox_inches='tight')
            plt.show()

    def extract_and_save_profiles(self):
        """
        Extract and save the total PV production, total non-dispatchable load profile,
        total EV profile in the base case, and heat pump consumption neglecting thermal capacitance.
        """
        output_path = 'Hourly_DERs_profiles'
        os.makedirs(output_path, exist_ok=True)

        # Total PV production
        if self.pv_generation is not None:
            total_pv_production = self.pv_generation.iloc[:, 2:].sum(axis=0)
            total_pv_production.to_csv(os.path.join(output_path, f'grid_pv_production_{self.data_year}.csv'), index=False)

        # Total non-dispatchable load profile
        if self.load_profiles is not None:
            total_non_dispatchable_load = self.load_profiles.iloc[:, 2:].sum(axis=0)
            total_non_dispatchable_load.to_csv(os.path.join(output_path, f'grid_non_dispatchable_load_{self.data_year}.csv'),
                                               index=False)

        # Total EV profile in the base case
        if self.ev_power_profiles is not None:
            total_ev_profile = self.ev_power_profiles.iloc[0, 2:] * self.ev_allocation['EV_share'].sum()
            total_ev_profile.to_csv(os.path.join(output_path, f'grid_ev_consumption_{self.data_year}.csv'), index=False)

            # Heat pump consumption considering COP and thermal conductivity
        if self.hp_allocation is not None and self.temperature_profiles is not None:
            hp_consumption = pd.Series(0, index=self.temperature_profiles.columns[1:])
            for _, row in self.hp_allocation.iterrows():
                profile_name = row['Temperature_profile_name']
                nominal_power = row['Nominal_power_kW']
                cop = row['COP']
                thermal_conductivity = row['Thermal_conductivity_kW/K']
                temperature_profile = self.temperature_profiles[
                                          self.temperature_profiles['Temperature_profile_name'] == profile_name].iloc[0,
                                      1:]

                # Calculate power based on ambient temperature
                power = thermal_conductivity * (22 - temperature_profile)

                # Adjust power using COP
                power /= cop

                # Cap power between 0 and nominal power
                power = power.clip(lower=0, upper=nominal_power)

                hp_consumption += power

            hp_consumption.to_csv(os.path.join(output_path, f'grid_hp_consumption_{self.data_year}.csv'), index=False)


if __name__ == "__main__":
    # Example usage
    grid_type = 'LV'  # or 'MV'
    grid_name = '6602-2_0_5'
    start_date = '01-01 00:00:00'
    end_date = '12-31 23:00:00'
    data_year = 2050

    loader = GridLoader(grid_type, grid_name, start_date, end_date, data_year)
    loader.load_grid()
    loader.load_pv_data()
    loader.load_hp_data()
    loader.load_ev_data()
    loader.load_bess_data()
    loader.load_conventional_load_profiles()
    loader.extract_and_save_profiles()
    loader.add_fields_to_nodes()
    loader.plot_grid(zoom_coords=[0.0, 0.0, 0.5, 0.5])
    # print(loader.nodes)
    # # print(loader.temperature_profiles)
    # # print(loader.ev_flexible_energy_profiles)
    # print(loader.load_profiles)

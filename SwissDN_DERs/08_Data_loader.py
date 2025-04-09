import os
import json
import pandas as pd
import geopandas as gpd
from datetime import datetime
import calendar
import zipfile


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
        self.script_path = os.getcwd()
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
            date_columns = [col for col in self.ev_power_profiles.columns if (col != 'BFS_municipality_code' and col != 'Profile_type')]
            filtered_columns = [col for col in date_columns if
                                self.start_date <= datetime.strptime(col, '%m-%d %H:%M:%S') <= self.end_date]
            self.ev_power_profiles = self.ev_power_profiles[['BFS_municipality_code', 'Profile_type'] + filtered_columns]

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
    print('Data loaded successfully.')

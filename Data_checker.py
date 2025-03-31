import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import geopandas as gpd
import matplotlib.colors as mcolors
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from matplotlib import rcParams, font_manager


class DataChecker:
    def __init__(self):
        self.lv_basicload_shares_df = None
        self.lv_heat_pump_allocation_df = None
        self.lv_generation_df = None
        self.lv_p_installed_df = None
        self.mv_generation_df = None
        self.mv_p_installed_df = None
        self.map_of_switzerland = None
        self.lv_std_df = None
        self.mv_std_df = None
        self.lv_hp_df = None
        self.mv_hp_df = None
        self.temperature_profiles_df = None
        self.simulation_year = None
        self.lv_bess_df = None
        self.mv_bess_df = None
        self.ev_allocation_df = None
        self.ev_energy_profiles_df = None
        self.ev_flexible_energy_profiles_df = None
        self.ev_power_profiles_df = None
        self.lv_grid_data = pd.DataFrame()
        self.lv_basicload_shares_df = None
        self.commercial_profiles_df = None
        self.residential_profiles_df = None
        self.mv_load_profile_df = None
        self.mv_grid_data = pd.DataFrame()
        self.simulation_years = [2030, 2040, 2050]
        self.map_color_min = 1e2
        self.map_color_max = 1e6
        self.mv_to_BSF = None

    def load_data(self):
        self.mv_to_BSF_path = 'Plotting_data/mv_grid_canton.csv'
        self.lv_basicload_shares_path = f'LV_basicload/LV_basicload_output/{self.simulation_year}/LV_basicload_shares.csv'
        self.lv_commercial_profiles_path = f'LV_basicload/LV_basicload_output/{self.simulation_year}/Commercial_profiles.csv'
        self.lv_residential_profiles_path = f'LV_basicload/LV_basicload_output/{self.simulation_year}/Residential_profiles.csv'
        # self.mv_basicload_profiles_path = f'MV_basicload/MV_basicload_output/{self.simulation_year}/MV_load_profile.csv'
        self.lv_basicload_shares_path = f'LV_basicload/LV_basicload_output/{self.simulation_year}/LV_basicload_shares.csv'
        self.mv_load_profile_path = f'MV_basicload/MV_basicload_output/{self.simulation_year}/MV_load_profile.csv'
        self.lv_generation_path = f'PV/PV_output/{self.simulation_year}/LV_generation.csv'
        self.lv_p_installed_path = f'PV/PV_output/{self.simulation_year}/LV_P_installed.csv'
        self.mv_generation_path = f'PV/PV_output/{self.simulation_year}/MV_generation.csv'
        self.mv_p_installed_path = f'PV/PV_output/{self.simulation_year}/MV_P_installed.csv'
        self.map_of_switzerland_path = f'Plotting_data/nine_zones.geojson'
        self.lv_std_path = f'PV/PV_output/{self.simulation_year}/LV_std.csv'
        self.mv_std_path = f'PV/PV_output/{self.simulation_year}/MV_std.csv'
        self.lv_hp_path = f'HP/HP_output/{self.simulation_year}/LV_heat_pump_allocation.csv'
        self.mv_hp_path = f'HP/HP_output/{self.simulation_year}/MV_heat_pump_allocation.csv'
        self.lv_bess_path = f'BESS/Output/{self.simulation_year}/BESS_allocation_LV.csv'
        self.mv_bess_path = f'BESS/Output/{self.simulation_year}/BESS_allocation_MV.csv'
        self.temperature_profiles_path = f'HP/HP_output/{self.simulation_year}/temperature_profiles.csv'
        self.ev_allocation_path = f'EV/EV_output/{self.simulation_year}/EV_allocation_LV.csv'
        self.ev_energy_profiles_path = f'EV/EV_output/{self.simulation_year}/EV_energy_profiles_LV.csv'
        self.ev_flexible_energy_profiles_path = f'EV/EV_output/{self.simulation_year}/EV_flexible_energy_profiles_LV.csv'
        self.ev_power_profiles_path = f'EV/EV_output/{self.simulation_year}/EV_power_profiles_LV.csv'
        # self.lv_basicload_shares_df = pd.read_csv(self.lv_basicload_shares_path)
        self.mv_to_BSF = pd.read_csv(self.mv_to_BSF_path)
        # self.lv_heat_pump_allocation_df = pd.read_csv(self.lv_hp_path)
        # self.lv_generation_df = pd.read_csv(self.lv_generation_path)
        # self.lv_p_installed_df = pd.read_csv(self.lv_p_installed_path)
        # self.mv_generation_df = pd.read_csv(self.mv_generation_path)
        # self.mv_p_installed_df = pd.read_csv(self.mv_p_installed_path)
        # self.map_of_switzerland = gpd.read_file(self.map_of_switzerland_path)
        # self.lv_std_df = pd.read_csv(self.lv_std_path)
        # self.mv_std_df = pd.read_csv(self.mv_std_path)
        # self.lv_hp_df = pd.read_csv(self.lv_hp_path)
        # self.mv_hp_df = pd.read_csv(self.mv_hp_path)
        # self.temperature_profiles_df = pd.read_csv(self.temperature_profiles_path)
        # self.lv_bess_df = pd.read_csv(self.lv_bess_path)
        # self.mv_bess_df = pd.read_csv(self.mv_bess_path)
        # self.ev_allocation_df = pd.read_csv(self.ev_allocation_path)
        # self.ev_energy_profiles_df = pd.read_csv(self.ev_energy_profiles_path)
        # self.ev_flexible_energy_profiles_df = pd.read_csv(self.ev_flexible_energy_profiles_path)
        # self.ev_power_profiles_df = pd.read_csv(self.ev_power_profiles_path)
        # self.load_lv_grids('PV/LV')
        # self.lv_basicload_shares_df = pd.read_csv(self.lv_basicload_shares_path)
        # self.commercial_profiles_df = pd.read_csv(self.lv_commercial_profiles_path)
        # self.residential_profiles_df = pd.read_csv(self.lv_residential_profiles_path)
        # self.mv_load_profile_df = pd.read_csv(self.mv_load_profile_path)
        # self.load_mv_grids('PV/MV')

    def check_lv_grid_consistency(self):
        lv_basicload_grids = set(self.lv_basicload_shares_df['LV_grid'].unique())
        lv_heat_pump_grids = set(self.lv_heat_pump_allocation_df['LV_grid'].unique())
        lv_generation_grids = set(self.lv_generation_df['LV_grid'].unique())

        missing_in_basicload = (lv_heat_pump_grids | lv_generation_grids) - lv_basicload_grids
        missing_in_heat_pump = (lv_basicload_grids | lv_generation_grids) - lv_heat_pump_grids
        missing_in_generation = (lv_basicload_grids | lv_heat_pump_grids) - lv_generation_grids

        if missing_in_basicload:
            print("Grids missing in LV_basicload_shares.csv:", missing_in_basicload)
        if missing_in_heat_pump:
            print("Grids missing in LV_heat_pump_allocation.csv:", missing_in_heat_pump)
        if missing_in_generation:
            print("Grids missing in LV_generation.csv:", missing_in_generation)

    def compute_pv_power_and_generation(self):
        # Compute total installed power
        total_lv_p_installed = self.lv_p_installed_df['P_installed (kWp)'].sum() / 1000  # Convert to MW
        total_mv_p_installed = self.mv_p_installed_df['P_installed (kWp)'].sum() / 1000  # Convert to MW

        # Compute total annual generation
        def compute_annual_generation(df):
            hourly_generation = df.iloc[:, 2:].sum(axis=0)
            days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            total_generation = sum(hourly_generation[i * 24:(i + 1) * 24].sum() * days_in_month[i] for i in range(12))
            return total_generation / 1000000  # Convert to MWh

        total_lv_generation = compute_annual_generation(self.lv_generation_df)
        total_mv_generation = compute_annual_generation(self.mv_generation_df)

        # Compute equivalent hours
        lv_equivalent_hours = total_lv_generation / total_lv_p_installed
        mv_equivalent_hours = total_mv_generation / total_mv_p_installed

        print(f"Total LV installed power: {total_lv_p_installed} MW")
        print(f"Total MV installed power: {total_mv_p_installed} MW")
        print(f"Total LV annual generation: {total_lv_generation} MWh")
        print(f"Total MV annual generation: {total_mv_generation} MWh")
        print(f"LV equivalent hours: {lv_equivalent_hours} hours")
        print(f"MV equivalent hours: {mv_equivalent_hours} hours")

        # Compute hourly generation for LV and MV
        lv_hourly_generation = self.lv_generation_df.iloc[:, 2:].sum(axis=0).values
        mv_hourly_generation = self.mv_generation_df.iloc[:, 2:].sum(axis=0).values

        # Repeat the monthly representative days to cover the entire year
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        lv_hourly_generation_year = []
        mv_hourly_generation_year = []
        for i, days in enumerate(days_in_month):
            daily_lv_profile = lv_hourly_generation[i * 24:(i + 1) * 24]
            daily_mv_profile = mv_hourly_generation[i * 24:(i + 1) * 24]
            lv_hourly_generation_year.extend(daily_lv_profile.tolist() * days)
            mv_hourly_generation_year.extend(daily_mv_profile.tolist() * days)

        # Sum LV and MV hourly generation
        total_hourly_generation = [(lv + mv) / 1000000 for lv, mv in zip(lv_hourly_generation_year, mv_hourly_generation_year)]

        # Save the hourly generation to a CSV file
        output_dir = os.path.join('Plotting_data', 'Hourly_DERs_profiles')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'total_pv_hourly_generation_{self.simulation_year}.csv')
        hourly_generation_df = pd.DataFrame(total_hourly_generation, columns=['Hourly Generation (MWh)'])
        hourly_generation_df.to_csv(output_file, index=False)

        print(f"Hourly PV generation saved to {output_file}")

        # Compute monthly generation for LV and MV
        def compute_monthly_generation(df):
            hourly_generation = df.iloc[:, 2:].sum(axis=0)
            monthly_generation = [hourly_generation[i * 24:(i + 1) * 24].sum() * days_in_month[i] for i in range(12)]
            return monthly_generation

        lv_monthly_generation = compute_monthly_generation(self.lv_generation_df)
        mv_monthly_generation = compute_monthly_generation(self.mv_generation_df)

        # Sum LV and MV monthly generation and convert to TWh
        total_monthly_generation_twh = [(lv + mv) / 1000000000000 for lv, mv in
                                        zip(lv_monthly_generation, mv_monthly_generation)]

        # Save the monthly generation to a CSV file
        output_dir = os.path.join('Plotting_data', 'Monthly_DERs_heatmap')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'total_pv_generation_{self.simulation_year}.csv')
        monthly_generation_df = pd.DataFrame([total_monthly_generation_twh], columns=[
            'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
            'November', 'December'
        ])
        monthly_generation_df.to_csv(output_file, index=False)

        print(f"Monthly PV generation saved to {output_file}")

    def plot_histograms(self):
        # Plot histogram of installed capacity per node for LV
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        lv_installed_capacity = self.lv_p_installed_df['P_installed (kWp)']
        plt.hist(lv_installed_capacity, bins=30, alpha=0.7, density=True)
        plt.xlabel('Installed Capacity (kWp)')
        plt.ylabel('Relative Frequency')
        plt.title('Histogram of LV Installed Capacity per Node')

        # Plot histogram of installed capacity per node for MV
        plt.subplot(1, 2, 2)
        mv_installed_capacity = self.mv_p_installed_df['P_installed (kWp)']
        plt.hist(mv_installed_capacity, bins=30, alpha=0.7, density=True)
        plt.xlabel('Installed Capacity (kWp)')
        plt.ylabel('Relative Frequency')
        plt.title('Histogram of MV Installed Capacity per Node')

        plt.tight_layout()
        plt.show()

        # Plot histogram of yearly production per node for LV
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        lv_yearly_production = self.lv_generation_df.iloc[:, 2:].sum(axis=1) / 1000000 * 30.42  # Convert to MWh and yearly
        plt.hist(lv_yearly_production, bins=30, alpha=0.7, density=True)
        plt.xlabel('Yearly Production (MWh)')
        plt.ylabel('Relative Frequency')
        plt.title('Histogram of LV Yearly Production per Node')

        # Plot histogram of yearly production per node for MV
        plt.subplot(1, 2, 2)
        mv_yearly_production = self.mv_generation_df.iloc[:, 2:].sum(axis=1) / 1000000 * 30.42  # Convert to MWh and yearly
        plt.hist(mv_yearly_production, bins=30, alpha=0.7, density=True)
        plt.xlabel('Yearly Production (MWh)')
        plt.ylabel('Relative Frequency')
        plt.title('Histogram of MV Yearly Production per Node')

        plt.tight_layout()
        plt.show()

    def check_installed_power_vs_peak_production(self):
        def check_peak(df_installed, df_generation, grid_col, osmid_col):
            installed_power = df_installed.set_index([grid_col, osmid_col])['P_installed (kWp)']
            peak_production = df_generation.set_index([grid_col, osmid_col]).iloc[:, 2:].max(axis=1)
            comparison = installed_power > (peak_production / 1000)  # Convert Wh to kWp
            return comparison

        lv_comparison = check_peak(self.lv_p_installed_df, self.lv_generation_df, 'LV_grid', 'LV_osmid')
        mv_comparison = check_peak(self.mv_p_installed_df, self.mv_generation_df, 'MV_grid', 'MV_osmid')

        if not lv_comparison.all():
            print("Some LV nodes have peak production higher than installed power.")
        else:
            print("All LV nodes have installed power higher than peak production.")
        if not mv_comparison.all():
            print("Some MV nodes have peak production higher than installed power.")
        else:
            print("All MV nodes have installed power higher than peak production.")

    def plot_installed_capacity_per_municipality(self):
        # Extract municipality codes and sum installed capacity
        self.lv_p_installed_df['municipality'] = self.lv_p_installed_df['LV_grid'].str.split('-').str[0]

        # Extract municipality part from MV_grid using mv_to_BSF dataframe
        self.mv_p_installed_df = self.mv_p_installed_df.merge(self.mv_to_BSF[['grid', 'BFS']], left_on='MV_grid', right_on='grid',
                                            how='left')
        self.mv_p_installed_df['municipality'] = self.mv_p_installed_df['BFS'].astype(int)

        lv_capacity_per_municipality = self.lv_p_installed_df.groupby('municipality')['P_installed (kWp)'].sum()
        mv_capacity_per_municipality = self.mv_p_installed_df.groupby('municipality')['P_installed (kWp)'].sum()

        total_capacity_per_municipality = lv_capacity_per_municipality.add(mv_capacity_per_municipality,
                                                                           fill_value=0).reset_index()
        total_capacity_per_municipality['municipality'] = total_capacity_per_municipality['municipality'].astype(int)
        total_capacity_per_municipality['P_installed (MW)'] = total_capacity_per_municipality[
                                                                  'P_installed (kWp)']  # Convert to MW

        output_file = os.path.join('Plotting_data', 'Switzerland_maps', 'total_PV_capacity_per_municipality.csv')
        total_capacity_per_municipality.to_csv(output_file, index=False)

        # Merge with map of Switzerland
        self.map_of_switzerland['BFS_NUMMER'] = self.map_of_switzerland['BFS_NUMMER'].astype(int)
        merged_map = self.map_of_switzerland.merge(total_capacity_per_municipality, left_on='BFS_NUMMER',
                                                   right_on='municipality', how='left')
        merged_map['P_installed (MWp)'] = merged_map.groupby('BFS_NUMMER')['P_installed (MW)'].transform('sum').fillna(0)

        # Plot the map
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))

        # Logarithmic normalization for color scale
        norm = mcolors.LogNorm(vmin=self.map_color_min,
                               vmax=self.map_color_max)
        # norm = mcolors.LogNorm(vmin=merged_map['P_installed (MWp)'].min() + 1,
        #                        vmax=merged_map['P_installed (MWp)'].max())
        cmap = plt.get_cmap('cividis')

        # Plot the map without automatic legend
        merged_map.plot(column='P_installed (MWp)', cmap=cmap, linewidth=0.2, ax=ax, edgecolor='0.5', legend=False,
                        norm=norm)

        # Create a colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Dummy array for colorbar
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5)  # Adjust shrink to make colorbar narrower

        # # Normalize color scale
        # norm = mcolors.Normalize(vmin=0, vmax=merged_map['P_installed (MWp)'].max())
        # cmap = plt.get_cmap('cividis')
        #
        # # Plot the map without automatic legend
        # merged_map.plot(column='P_installed (MWp)', cmap=cmap, linewidth=0.2, ax=ax, edgecolor='0.5', legend=False,
        #                 norm=norm)
        #
        # # Create a colorbar
        # sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])  # Dummy array for colorbar
        # cbar = fig.colorbar(sm, ax=ax, shrink=0.5)  # Adjust shrink to make colorbar narrower

        # Add colorbar label
        cbar.set_label("Installed capacity (MWp)")

        # Set plot title
        ax.set_title('Installed PV Capacity per Municipality in Switzerland')
        ax.set_axis_off()
        plt.savefig(os.path.join(os.getcwd(), 'Plotting_data', 'Switzerland_maps', f'PV_map.png'), format='png',
                    bbox_inches='tight')
        plt.show()

    def plot_cov_histogram(self):
        # Load additional data
        rooftop_pv_path = 'PV/PV_input/PV_data/rooftop_PV_CH_EPV_W_by_building.csv'
        rooftop_pv_std_path = 'PV/PV_input/PV_data/rooftop_PV_CH_EPV_W_std_by_building.csv'
        rooftop_pv_df = pd.read_csv(rooftop_pv_path)
        rooftop_pv_std_df = pd.read_csv(rooftop_pv_std_path)

        # Compute CoV for LV
        lv_std_values = self.lv_std_df.iloc[:, 2:]
        lv_generation_values = self.lv_generation_df.iloc[:, 2:]
        lv_cov = lv_std_values.div(lv_generation_values.replace(0, np.nan)).fillna(0).values.flatten()

        # Compute CoV for MV
        mv_std_values = self.mv_std_df.iloc[:, 2:]
        mv_generation_values = self.mv_generation_df.iloc[:, 2:]
        mv_cov = mv_std_values.div(mv_generation_values.replace(0, np.nan)).fillna(0).values.flatten()

        # Compute CoV for rooftop PV
        rooftop_std_values = rooftop_pv_std_df.iloc[:, 1:]
        rooftop_generation_values = rooftop_pv_df.iloc[:, 1:]
        rooftop_cov = rooftop_std_values.div(rooftop_generation_values.replace(0, np.nan)).fillna(0).values.flatten()

        # Combine LV and MV CoV values
        lv_mv_cov = np.concatenate([lv_cov, mv_cov])

        # Filter CoVs to be between 0 and 1
        filtered_lv_mv_cov = lv_mv_cov[(lv_mv_cov > 0) & (lv_mv_cov < 1)]
        filtered_rooftop_cov = rooftop_cov[(rooftop_cov > 0) & (rooftop_cov < 1)]

        # Count CoVs larger than 1
        count_lv_mv_cov_larger_than_one = np.sum(lv_mv_cov > 1)
        count_rooftop_cov_larger_than_one = np.sum(rooftop_cov > 1)
        print(f"Number of LV/MV CoVs larger than 1: {count_lv_mv_cov_larger_than_one}")
        print(f"Number of Rooftop PV CoVs larger than 1: {count_rooftop_cov_larger_than_one}")

        # Plot histograms of filtered CoV
        plt.figure(figsize=(12, 12))

        plt.subplot(2, 1, 1)
        plt.hist(filtered_lv_mv_cov, bins=30, alpha=0.7, density=True)
        plt.xlabel('Coefficient of Variation (CoV)')
        plt.ylabel('Relative Frequency')
        plt.title('Histogram of LV/MV Coefficient of Variation (CoV) (0 < CoV < 1)')

        plt.subplot(2, 1, 2)
        plt.hist(filtered_rooftop_cov, bins=30, alpha=0.7, density=True)
        plt.xlabel('Coefficient of Variation (CoV)')
        plt.ylabel('Relative Frequency')
        plt.title('Histogram of Rooftop PV Coefficient of Variation (CoV) (0 < CoV < 1)')

        plt.tight_layout()
        plt.show()

    def plot_yearly_generation_per_municipality(self):
        # Extract municipality codes
        lv_yearly_production = pd.DataFrame()
        mv_yearly_production = pd.DataFrame()
        lv_yearly_production['generation (GWh)'] = self.lv_generation_df.iloc[:, 2:].sum(axis=1) / 1000000000 * 30.42
        mv_yearly_production['generation (GWh)'] = self.mv_generation_df.iloc[:, 2:].sum(axis=1) / 1000000000 * 30.42
        lv_yearly_production['municipality'] = self.lv_generation_df['LV_grid'].str.split('-').str[0]
        mv_yearly_production['municipality'] = self.mv_generation_df['MV_grid'].str.split('_').str[0]

        # # Compute yearly generation for LV and MV
        # def compute_annual_generation(df):
        #     hourly_generation = df.iloc[:, 2:].sum(axis=1)
        #     days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        #     total_generation = sum(hourly_generation[i * 24:(i + 1) * 24].sum() * days_in_month[i] for i in range(12))
        #     return total_generation / 1000000  # Convert to GWh

        # # Ensure the production values are floats
        # lv_yearly_production = lv_yearly_production.astype(float)
        # mv_yearly_production = mv_yearly_production.astype(float)

        lv_generation_per_municipality = lv_yearly_production.groupby('municipality').sum()
        mv_generation_per_municipality = mv_yearly_production.groupby('municipality').sum()

        total_generation_per_municipality = lv_generation_per_municipality.add(mv_generation_per_municipality,
                                                                               fill_value=0).reset_index()
        total_generation_per_municipality['municipality'] = total_generation_per_municipality['municipality'].astype(
            int)
        total_generation_per_municipality.columns = ['municipality', 'generation (GWh)']

        # Merge with map of Switzerland
        self.map_of_switzerland['BFS_NUMMER'] = self.map_of_switzerland['BFS_NUMMER'].astype(int)
        merged_map = self.map_of_switzerland.merge(total_generation_per_municipality, left_on='BFS_NUMMER',
                                                   right_on='municipality', how='left')
        merged_map['generation (GWh)'] = merged_map.groupby('BFS_NUMMER')['generation (GWh)'].transform('sum').fillna(0)

        # Plot the map
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))

        # Logarithmic normalization for color scale
        norm = mcolors.LogNorm(vmin=merged_map['generation (GWh)'].min() + 1,
                               vmax=merged_map['generation (GWh)'].max())
        cmap = plt.get_cmap('cividis')

        # Plot the map without automatic legend
        merged_map.plot(column='generation (GWh)', cmap=cmap, linewidth=0.2, ax=ax, edgecolor='0.5', legend=False,
                        norm=norm)

        # Create a colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Dummy array for colorbar
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5)  # Adjust shrink to make colorbar narrower

        # Add colorbar label
        cbar.set_label("Yearly generation (GWh)")

        # Set plot title
        ax.set_title('Yearly PV Generation per Municipality in Switzerland')
        ax.set_axis_off()

        plt.show()

    def plot_average_equivalent_hours_per_municipality(self):
        # Extract municipality codes and add as the first column
        self.lv_generation_df.insert(0, 'municipality', self.lv_generation_df['LV_grid'].str.split('-').str[0])
        self.mv_generation_df.insert(0, 'municipality', self.mv_generation_df['MV_grid'].str.split('_').str[0])
        self.lv_p_installed_df.insert(0, 'municipality', self.lv_p_installed_df['LV_grid'].str.split('-').str[0])
        self.mv_p_installed_df.insert(0, 'municipality', self.mv_p_installed_df['MV_grid'].str.split('_').str[0])

        # Compute yearly production for LV and MV starting from the third column
        lv_yearly_production = self.lv_generation_df.iloc[:, 3:].sum(axis=1) / 1000000000 * 30.42  # Convert to GWh
        mv_yearly_production = self.mv_generation_df.iloc[:, 3:].sum(axis=1) / 1000000000 * 30.42  # Convert to GWh

        # Sum the production per municipality
        lv_generation_per_municipality = lv_yearly_production.groupby(self.lv_generation_df['municipality']).sum()
        mv_generation_per_municipality = mv_yearly_production.groupby(self.mv_generation_df['municipality']).sum()

        total_generation_per_municipality = lv_generation_per_municipality.add(mv_generation_per_municipality,
                                                                               fill_value=0).reset_index()
        total_generation_per_municipality['municipality'] = total_generation_per_municipality['municipality'].astype(
            int)
        total_generation_per_municipality.columns = ['municipality', 'generation (GWh)']

        # Sum the installed power per municipality
        lv_capacity_per_municipality = self.lv_p_installed_df.groupby('municipality')[
                                           'P_installed (kWp)'].sum() / 1000  # Convert to MW
        mv_capacity_per_municipality = self.mv_p_installed_df.groupby('municipality')[
                                           'P_installed (kWp)'].sum() / 1000  # Convert to MW

        total_capacity_per_municipality = lv_capacity_per_municipality.add(mv_capacity_per_municipality,
                                                                           fill_value=0).reset_index()
        total_capacity_per_municipality['municipality'] = total_capacity_per_municipality['municipality'].astype(int)
        total_capacity_per_municipality.columns = ['municipality', 'P_installed (MW)']

        # Merge generation and capacity data
        merged_data = total_generation_per_municipality.merge(total_capacity_per_municipality, on='municipality')
        merged_data['equivalent_hours'] = merged_data['generation (GWh)'] * 1000 / merged_data[
            'P_installed (MW)']  # Convert GWh to MWh

        # Merge with map of Switzerland
        self.map_of_switzerland['BFS_NUMMER'] = self.map_of_switzerland['BFS_NUMMER'].astype(int)
        merged_map = self.map_of_switzerland.merge(merged_data, left_on='BFS_NUMMER', right_on='municipality',
                                                   how='left')
        merged_map['equivalent_hours'] = merged_map.groupby('BFS_NUMMER')['equivalent_hours'].transform('sum').fillna(0)

        # Plot the map
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))

        # Linear normalization for color scale
        norm = mcolors.Normalize(vmin=800, vmax=merged_map['equivalent_hours'].max())
        cmap = plt.get_cmap('cividis')

        # Plot the map without automatic legend
        merged_map.plot(column='equivalent_hours', cmap=cmap, linewidth=0.2, ax=ax, edgecolor='0.5', legend=False,
                        norm=norm)

        # Create a colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Dummy array for colorbar
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5)  # Adjust shrink to make colorbar narrower

        # Add colorbar label
        cbar.set_label("Average Equivalent Hours")

        # Set plot title
        ax.set_title('Average Equivalent Hours per Municipality in Switzerland')
        ax.set_axis_off()

        plt.show()

    def plot_hp_histograms(self):
        # Plot histograms for LV heat pumps
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 3, 1)
        plt.hist(self.lv_hp_df['PRT_kW'], bins=30, alpha=0.7)
        plt.xlabel('Nominal Power (kW)')
        plt.title('LV Heat Pumps Nominal Power')

        plt.subplot(2, 3, 2)
        plt.hist(self.lv_hp_df['CBLD_KWh/K'], bins=30, alpha=0.7)
        plt.xlabel('Thermal Capacitance (kWh/K)')
        plt.title('LV Heat Pumps Thermal Capacitance')

        plt.subplot(2, 3, 3)
        plt.hist(self.lv_hp_df['HBLD_kW/K'], bins=30, alpha=0.7)
        plt.xlabel('Thermal Conductivity (kW/K)')
        plt.title('LV Heat Pumps Thermal Conductivity')

        # Plot histograms for MV heat pumps
        plt.subplot(2, 3, 4)
        plt.hist(self.mv_hp_df['PRT_kW'], bins=30, alpha=0.7)
        plt.xlabel('Nominal Power (kW)')
        plt.title('MV Heat Pumps Nominal Power')

        plt.subplot(2, 3, 5)
        plt.hist(self.mv_hp_df['CBLD_KWh/K'], bins=30, alpha=0.7)
        plt.xlabel('Thermal Capacitance (kWh/K)')
        plt.title('MV Heat Pumps Thermal Capacitance')

        plt.subplot(2, 3, 6)
        plt.hist(self.mv_hp_df['HBLD_kW/K'], bins=30, alpha=0.7)
        plt.xlabel('Thermal Conductivity (kW/K)')
        plt.title('MV Heat Pumps Thermal Conductivity')

        plt.tight_layout()
        plt.show()

        # Compute total installed power
        total_lv_power = self.lv_hp_df['PRT_kW'].sum() / 1000  # Convert to MW
        total_mv_power = self.mv_hp_df['PRT_kW'].sum() / 1000  # Convert to MW
        print(f"Total installed power in LV networks: {total_lv_power:.2f} MW")
        print(f"Total installed power in MV networks: {total_mv_power:.2f} MW")

    def compute_hp_consumption(self, installation_share=1, efficiency_factor=1):
        target_temp = 18  # Target internal temperature in Celsius

        # Randomly sample the dataframes based on installation_share
        self.lv_hp_df = self.lv_hp_df.sample(frac=installation_share, random_state=1)
        self.mv_hp_df = self.mv_hp_df.sample(frac=installation_share, random_state=1)

        # Apply efficiency factor
        self.lv_hp_df['HBLD_kW/K'] *= efficiency_factor
        self.lv_hp_df['PRT_kW'] *= efficiency_factor
        self.mv_hp_df['HBLD_kW/K'] *= efficiency_factor
        self.mv_hp_df['PRT_kW'] *= efficiency_factor

        def compute_group_consumption(group, temp_profile):
            consumption = []
            total_conductance = group['HBLD_kW/K'].sum()
            total_nominal_power = group['PRT_kW'].sum()
            cop = group['COP'].mean()
            for temp in temp_profile:
                if temp < target_temp:
                    power = (target_temp - temp) * total_conductance
                    power = min(power, total_nominal_power)  # Cap to total nominal power
                    consumption.append(power / cop)  # Divide by COP
                else:
                    consumption.append(0)
            return consumption

        lv_consumption = []
        for t_profile, group in tqdm(self.lv_hp_df.groupby('T_PROFILE'), desc="Processing LV Heat Pumps"):
            temp_profile = self.temperature_profiles_df[t_profile].values
            lv_consumption.append(compute_group_consumption(group, temp_profile))

        mv_consumption = []
        for t_profile, group in tqdm(self.mv_hp_df.groupby('T_PROFILE'), desc="Processing MV Heat Pumps"):
            temp_profile = self.temperature_profiles_df[t_profile].values
            mv_consumption.append(compute_group_consumption(group, temp_profile))

        lv_consumption = np.sum(lv_consumption, axis=0)
        mv_consumption = np.sum(mv_consumption, axis=0)

        # Sum LV and MV yearly consumption
        total_yearly_consumption = (lv_consumption + mv_consumption) / 1000

        # Save the yearly consumption to a CSV file
        output_dir = os.path.join('Plotting_data', 'Hourly_DERs_profiles')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'total_hp_yearly_consumption_{self.simulation_year}.csv')
        yearly_consumption_df = pd.DataFrame(total_yearly_consumption, columns=['Hourly Consumption (MWh)'])
        yearly_consumption_df.to_csv(output_file, index=False)

        print(f"Yearly HP consumption saved to {output_file}")

        # Compute monthly consumption for LV and MV
        def compute_monthly_consumption(consumption):
            days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            hours_in_month = [days * 24 for days in days_in_month]
            monthly_consumption = []
            start = 0
            for hours in hours_in_month:
                monthly_consumption.append(sum(consumption[start:start + hours]))
                start += hours
            return monthly_consumption

        lv_monthly_consumption = compute_monthly_consumption(lv_consumption)
        mv_monthly_consumption = compute_monthly_consumption(mv_consumption)

        # Sum LV and MV monthly consumption and convert to TWh
        total_monthly_consumption_twh = [(lv + mv) / 1000000000 for lv, mv in
                                         zip(lv_monthly_consumption, mv_monthly_consumption)]

        # Save the monthly consumption to a CSV file
        output_dir = os.path.join('Plotting_data', 'Monthly_DERs_heatmap')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'total_hp_consumption_{self.simulation_year}.csv')
        monthly_consumption_df = pd.DataFrame([total_monthly_consumption_twh], columns=[
            'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
            'November', 'December'
        ])
        monthly_consumption_df.to_csv(output_file, index=False)

        print(f"Monthly HP consumption saved to {output_file}")

        # Plot the profile
        plt.figure(figsize=(12, 6))
        plt.plot(lv_consumption / 1e6, label='LV Heat Pumps')
        plt.plot(mv_consumption / 1e6, label='MV Heat Pumps')
        plt.plot((mv_consumption + lv_consumption) / 1e6, label='Total')
        plt.xlabel('Time Step')
        plt.ylabel('Power Consumption (GW)')
        plt.title('Heat Pump Power Consumption Profile')
        plt.legend()
        plt.show()

        # Compute total yearly energy consumption
        total_lv_energy = np.sum(lv_consumption) / 1000000  # Convert to GWh
        total_mv_energy = np.sum(mv_consumption) / 1000000  # Convert to GWh
        print(f"Total yearly energy consumption in LV networks: {total_lv_energy:.2f} GWh")
        print(f"Total yearly energy consumption in MV networks: {total_mv_energy:.2f} GWh")
        print(f"Total yearly energy consumption: {total_lv_energy + total_mv_energy:.2f} GWh")

        # Extract municipality part from grid_name
        self.lv_hp_df['municipality'] = self.lv_hp_df['LV_grid'].str.split('-').str[0].astype(int)

        # Extract municipality part from MV_grid using mv_to_BSF dataframe
        self.mv_hp_df = self.mv_hp_df.merge(self.mv_to_BSF[['grid', 'BFS']], left_on='MV_grid', right_on='grid',
                                                how='left')
        self.mv_hp_df['municipality'] = self.mv_hp_df['BFS'].astype(int)

        # Group by municipality and check the sum of percentages for LV and MV
        lv_hp_allocation = self.lv_hp_df.groupby('municipality')['PRT_kW'].sum()
        mv_hp_allocation = self.mv_hp_df.groupby('municipality')['PRT_kW'].sum()

        # Combine LV and MV HP allocations
        hp_allocation = lv_hp_allocation.add(mv_hp_allocation, fill_value=0)

        output_file = os.path.join('Plotting_data', 'Switzerland_maps', 'total_HP_capacity_per_municipality.csv')
        hp_allocation.to_csv(output_file, index=True)

        # Plot map of Switzerland with yearly energy consumption of EVs in each municipality
        # bess_allocation = self.lv_bess_df.loc['municipality', 'Nominal_Power_kW']

        # yearly_ev_consumption.name = 'yearly_ev_consumption'  # Assign a name to the Series
        gdf = gpd.read_file(self.map_of_switzerland_path)
        gdf['BFS_NUMMER'] = gdf['BFS_NUMMER'].astype(int)
        gdf = gdf.merge(hp_allocation, left_on='BFS_NUMMER', right_on='municipality')
        gdf['hp_allocation'] = gdf.groupby('BFS_NUMMER')['PRT_kW'].transform('sum').fillna(0)

        # Plot the map
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))

        # Logarithmic normalization for color scale
        # norm = mcolors.LogNorm(vmin=gdf['hp_allocation'].min() + 1,
        #                        vmax=gdf['hp_allocation'].max())
        norm = mcolors.LogNorm(vmin=self.map_color_min,
                               vmax=self.map_color_max)
        cmap = plt.get_cmap('cividis')

        # Plot the map without automatic legend
        gdf.plot(column='hp_allocation', cmap=cmap, linewidth=0.2, ax=ax, edgecolor='0.5', legend=False,
                 norm=norm)

        # Create a colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Dummy array for colorbar
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5)  # Adjust shrink to make colorbar narrower

        # Add colorbar label
        cbar.set_label("HP capacity (kW)")

        # Set plot title
        ax.set_title('HP Power capacity in each Municipality')
        ax.set_axis_off()
        plt.savefig(os.path.join(os.getcwd(), 'Plotting_data', 'Switzerland_maps', f'HP_map.png'), format='png',
                    bbox_inches='tight')
        plt.show()

    def compute_bess_statistics_and_plot(self):
        # Compute total installed power and capacity for LV
        total_lv_power = self.lv_bess_df['Nominal_Power_kW'].sum() / 1000  # Convert to MW
        total_lv_capacity = self.lv_bess_df['Battery_Capacity_kWh'].sum() / 1000  # Convert to MWh

        # Compute total installed power and capacity for MV
        total_mv_power = self.mv_bess_df['Nominal_Power_kW'].sum() / 1000  # Convert to MW
        total_mv_capacity = self.mv_bess_df['Battery_Capacity_kWh'].sum() / 1000  # Convert to MWh

        print(f"Total installed power in LV networks: {total_lv_power:.2f} MW")
        print(f"Total installed capacity in LV networks: {total_lv_capacity:.2f} MWh")
        print(f"Total installed power in MV networks: {total_mv_power:.2f} MW")
        print(f"Total installed capacity in MV networks: {total_mv_capacity:.2f} MWh")

        # Plot histograms for LV BESS
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(self.lv_bess_df['Nominal_Power_kW'], bins=30, alpha=0.7)
        plt.xlabel('Nominal Power (kW)')
        plt.title('LV BESS Nominal Power')

        plt.subplot(1, 2, 2)
        plt.hist(self.lv_bess_df['Battery_Capacity_kWh'], bins=30, alpha=0.7)
        plt.xlabel('Battery Capacity (kWh)')
        plt.title('LV BESS Battery Capacity')

        plt.tight_layout()
        plt.show()

        # Plot histograms for MV BESS
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(self.mv_bess_df['Nominal_Power_kW'], bins=30, alpha=0.7)
        plt.xlabel('Nominal Power (kW)')
        plt.title('MV BESS Nominal Power')

        plt.subplot(1, 2, 2)
        plt.hist(self.mv_bess_df['Battery_Capacity_kWh'], bins=30, alpha=0.7)
        plt.xlabel('Battery Capacity (kWh)')
        plt.title('MV BESS Battery Capacity')

        plt.tight_layout()
        plt.show()

        # Extract municipality part from grid_name
        self.lv_bess_df['municipality'] = self.lv_bess_df['LV_grid'].str.split('-').str[0].astype(int)

        # Extract municipality part from MV_grid using mv_to_BSF dataframe
        self.mv_bess_df = self.mv_bess_df.merge(self.mv_to_BSF[['grid', 'BFS']], left_on='MV_grid', right_on='grid',
                                                how='left')
        self.mv_bess_df['municipality'] = self.mv_bess_df['BFS'].astype(int)

        # Group by municipality and check the sum of percentages for LV and MV
        lv_bess_allocation = self.lv_bess_df.groupby('municipality')['Nominal_Power_kW'].sum()
        mv_bess_allocation = self.mv_bess_df.groupby('municipality')['Nominal_Power_kW'].sum()

        # Combine LV and MV BESS allocations
        bess_allocation = lv_bess_allocation.add(mv_bess_allocation, fill_value=0)

        output_file = os.path.join('Plotting_data', 'Switzerland_maps', 'total_BESS_capacity_per_municipality.csv')
        bess_allocation.to_csv(output_file, index=True)

        # Plot map of Switzerland with yearly energy consumption of EVs in each municipality
        # bess_allocation = self.lv_bess_df.loc['municipality', 'Nominal_Power_kW']

        # yearly_ev_consumption.name = 'yearly_ev_consumption'  # Assign a name to the Series
        gdf = gpd.read_file(self.map_of_switzerland_path)
        gdf['BFS_NUMMER'] = gdf['BFS_NUMMER'].astype(int)
        gdf = gdf.merge(bess_allocation, left_on='BFS_NUMMER', right_on='municipality')
        gdf['bess_allocation'] = gdf.groupby('BFS_NUMMER')['Nominal_Power_kW'].transform('sum').fillna(0)

        # Plot the map
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))

        # Logarithmic normalization for color scale
        # norm = mcolors.LogNorm(vmin=gdf['bess_allocation'].min() + 1,
        #                        vmax=gdf['bess_allocation'].max())
        norm = mcolors.LogNorm(vmin=self.map_color_min,
                               vmax=self.map_color_max)
        cmap = plt.get_cmap('cividis')

        # Plot the map without automatic legend
        gdf.plot(column='bess_allocation', cmap=cmap, linewidth=0.2, ax=ax, edgecolor='0.5', legend=False,
                 norm=norm)

        # Create a colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Dummy array for colorbar
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5)  # Adjust shrink to make colorbar narrower

        # Add colorbar label
        cbar.set_label("Bess capacity (kW)")

        # Set plot title
        ax.set_title('BESS Power capaicty in Each Municipality')
        ax.set_axis_off()
        plt.savefig(os.path.join(os.getcwd(), 'Plotting_data', 'Switzerland_maps', f'BESS_map.png'), format='png',
                    bbox_inches='tight')
        plt.show()

    def analyze_ev_data(self):
        # Plot base profile and upper/lower bounds
        base_profile = self.ev_power_profiles_df[self.ev_power_profiles_df['Type'] == 'Base'].iloc[:, 2:].sum()
        upper_bound = self.ev_power_profiles_df[self.ev_power_profiles_df['Type'] == 'Upper'].iloc[:, 2:].sum()
        lower_bound = self.ev_power_profiles_df[self.ev_power_profiles_df['Type'] == 'Lower'].iloc[:, 2:].sum()

        plt.figure(figsize=(12, 6))
        plt.plot(base_profile, label='Base Profile')
        plt.plot(upper_bound, label='Upper Bound', linestyle='--')
        plt.plot(lower_bound, label='Lower Bound', linestyle='--')
        plt.xlabel('Hour of the Year')
        plt.ylabel('Power (kW)')
        plt.title('EV Power Profiles')
        plt.legend()
        plt.show()

        # Compute total yearly EV consumption on the base power profile
        total_yearly_ev_consumption = base_profile.sum() / 1000  # Convert to MWh
        print(f"Total yearly EV consumption on the base power profile: {total_yearly_ev_consumption:.2f} MWh")

        # Save the yearly consumption to a CSV file
        output_dir = os.path.join('Plotting_data', 'Hourly_DERs_profiles')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'total_ev_yearly_consumption_{self.simulation_year}.csv')
        yearly_consumption_df = pd.DataFrame(base_profile / 1000, columns=['Yearly Consumption (MW)'])
        yearly_consumption_df.to_csv(output_file, index=False)

        print(f"Yearly EV consumption saved to {output_file}")

        # Compute monthly consumption for EVs
        def compute_monthly_consumption(consumption):
            days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            hours_in_month = [days * 24 for days in days_in_month]
            monthly_consumption = []
            start = 0
            for hours in hours_in_month:
                monthly_consumption.append(sum(consumption[start:start + hours]))
                start += hours
            return monthly_consumption

        ev_monthly_consumption = compute_monthly_consumption(base_profile)

        # Convert to TWh
        ev_monthly_consumption_twh = [consumption / 1000000000 for consumption in ev_monthly_consumption]

        # Save the monthly consumption to a CSV file
        output_dir = os.path.join('Plotting_data', 'Monthly_DERs_heatmap')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'total_ev_consumption_{self.simulation_year}.csv')
        monthly_consumption_df = pd.DataFrame([ev_monthly_consumption_twh], columns=[
            'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
            'November', 'December'
        ])
        monthly_consumption_df.to_csv(output_file, index=False)

        print(f"Monthly EV consumption saved to {output_file}")

        # Check that the base power is always larger or equal than the lower bound and lower or equal than the upper bound
        for municipality in self.ev_power_profiles_df['BFS'].unique():
            base = self.ev_power_profiles_df[(self.ev_power_profiles_df['BFS'] == municipality) & (self.ev_power_profiles_df['Type'] == 'Base')].iloc[:, 2:].values.flatten()
            upper = self.ev_power_profiles_df[(self.ev_power_profiles_df['BFS'] == municipality) & (self.ev_power_profiles_df['Type'] == 'Upper')].iloc[:, 2:].values.flatten()
            lower = self.ev_power_profiles_df[(self.ev_power_profiles_df['BFS'] == municipality) & (self.ev_power_profiles_df['Type'] == 'Lower')].iloc[:, 2:].values.flatten()
            assert np.all(base >= lower), f"Base profile is lower than the lower bound for municipality {municipality}"
            assert np.all(base <= upper), f"Base profile is higher than the upper bound for municipality {municipality}"

        # Extract municipality part from grid_name
        self.ev_allocation_df['municipality'] = self.ev_allocation_df['grid_name'].str.split('-').str[0]

        # Group by municipality and check the sum of percentages
        municipality_shares = self.ev_allocation_df.groupby('municipality')['percentage'].sum()
        assert np.allclose(municipality_shares, 1), "Shares for each municipality do not sum to one"

        # Plot map of Switzerland with yearly energy consumption of EVs in each municipality
        yearly_ev_peak_consumption = pd.DataFrame()
        temp = self.ev_power_profiles_df.loc[self.ev_power_profiles_df['Type'] == 'Upper']
        yearly_ev_peak_consumption['BFS'] = temp['BFS']
        yearly_ev_peak_consumption['yearly_ev_peak_consumption'] = temp.iloc[:, 2:].max(axis=1)

        output_file = os.path.join('Plotting_data', 'Switzerland_maps', 'total_EV_capacity_per_municipality.csv')
        yearly_ev_peak_consumption.to_csv(output_file, index=False)

        # yearly_ev_consumption.name = 'yearly_ev_consumption'  # Assign a name to the Series
        gdf = gpd.read_file(self.map_of_switzerland_path)
        gdf = gdf.merge(yearly_ev_peak_consumption, left_on='BFS_NUMMER', right_on='BFS')
        gdf['yearly_ev_peak_consumption'] = gdf.groupby('BFS_NUMMER')['yearly_ev_peak_consumption'].transform('sum').fillna(0)

        # Plot the map
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))

        # Logarithmic normalization for color scale
        norm = mcolors.LogNorm(vmin=self.map_color_min,
                               vmax=self.map_color_max)
        # norm = mcolors.LogNorm(vmin=gdf['yearly_ev_peak_consumption'].min() + 1,
        #                        vmax=gdf['yearly_ev_peak_consumption'].max())
        cmap = plt.get_cmap('cividis')

        # Plot the map without automatic legend
        gdf.plot(column='yearly_ev_peak_consumption', cmap=cmap, linewidth=0.2, ax=ax, edgecolor='0.5', legend=False,
                 norm=norm)

        # Create a colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Dummy array for colorbar
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5)  # Adjust shrink to make colorbar narrower

        # Add colorbar label
        cbar.set_label("Yearly EV Peak Consumption (kW)")

        # Set plot title
        ax.set_title('Yearly Peak Consumption EVs in Each Municipality')
        ax.set_axis_off()
        plt.savefig(os.path.join(os.getcwd(), 'Plotting_data', 'Switzerland_maps', f'EV_map.png'), format='png',
                    bbox_inches='tight')
        plt.show()

        # Plot map of Switzerland with available flexible energy for all municipalities
        total_flexible_energy = pd.DataFrame()
        total_flexible_energy['BFS'] = self.ev_flexible_energy_profiles_df['BFS']
        total_flexible_energy['total_flexible_energy'] = self.ev_flexible_energy_profiles_df.iloc[:, 1:].sum(axis=1)
        # total_flexible_energy = self.ev_flexible_energy_profiles_df.iloc[:, 1:].sum(axis=1)
        # total_flexible_energy.name = 'total_flexible_energy'  # Assign a name to the Series
        gdf = gdf.merge(total_flexible_energy, left_on='BFS_NUMMER', right_on='BFS')
        gdf['total_flexible_energy'] = gdf.groupby('BFS_NUMMER')['total_flexible_energy'].transform('sum').fillna(0)

        # Plot the map
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))

        # Logarithmic normalization for color scale
        norm = mcolors.LogNorm(vmin=gdf['total_flexible_energy'].min() + 1,
                               vmax=gdf['total_flexible_energy'].max())
        cmap = plt.get_cmap('cividis')

        # Plot the map without automatic legend
        gdf.plot(column='total_flexible_energy', cmap=cmap, linewidth=0.2, ax=ax, edgecolor='0.5', legend=False,
                 norm=norm)

        # Create a colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Dummy array for colorbar
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5)  # Adjust shrink to make colorbar narrower

        # Add colorbar label
        cbar.set_label("Available Flexible Energy (MWh)")

        # Set plot title
        ax.set_title('Available Flexible Energy for EVs in Each Municipality')
        ax.set_axis_off()

        plt.show()

    def load_lv_grids(self, base_path):
        file_path = 'Plotting_data/lv_grid_data.csv'

        # Check if the file already exists
        if os.path.exists(file_path):
            self.lv_grid_data = pd.read_csv(file_path)
            print(f"Loaded LV grid data from {file_path}")
            return

        grid_data = []
        for root, dirs, files in os.walk(base_path):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                print(f"Loading LV grids directory: {dir_path}")
                for file in os.listdir(dir_path):
                    if file.endswith("_nodes"):
                        grid_name = file.split("_nodes")[0]
                        file_path = os.path.join(dir_path, file)
                        gdf = gpd.read_file(file_path)
                        for _, row in gdf.iterrows():
                            grid_data.append({
                                'LV_grid': grid_name,
                                'LV_osmid': row['osmid'],
                                'Peak_load': row['el_dmd']
                            })

        self.lv_grid_data = pd.DataFrame(grid_data)

        # Save the DataFrame to a file
        self.lv_grid_data.to_csv('Plotting_data/lv_grid_data.csv', index=False)

    def load_mv_grids(self, base_path):
        grid_data = []
        for file in os.listdir(base_path):
            if file.endswith("_nodes"):
                municipality, mv_grid = file.split("_")[0], file.split("_")[1]
                file_path = os.path.join(base_path, file)
                gdf = gpd.read_file(file_path)
                for _, row in gdf.iterrows():
                    if str(row['lv_grid']) != str(-1):
                        continue
                    grid_data.append({
                        'municipality': int(municipality),
                        'MV_grid': int(mv_grid),
                        'osmid': row['osmid'],
                        'Peak_load': row['el_dmd']
                    })
        self.mv_grid_data = pd.DataFrame(grid_data)

    def compute_lv_load_profiles(self, chunk_size=100000):
        # Merge lv_grid_data with lv_basicload_shares_df to get com_percentage and res_percentage
        merged_df = self.lv_grid_data.merge(self.lv_basicload_shares_df, on=['LV_grid', 'LV_osmid'])

        # Extract municipality from LV_grid
        merged_df['municipality'] = merged_df['LV_grid'].str.split('-').str[0]

        # Ensure municipality values are integers to match the profile DataFrames
        merged_df['municipality'] = merged_df['municipality'].astype(int)

        # Transpose the profile DataFrames to get municipalities as indices
        commercial_profiles_df_transposed = self.commercial_profiles_df.T
        residential_profiles_df_transposed = self.residential_profiles_df.T

        # Convert the indices of the transposed DataFrames to integers
        commercial_profiles_df_transposed.index = commercial_profiles_df_transposed.index.astype(int)
        residential_profiles_df_transposed.index = residential_profiles_df_transposed.index.astype(int)

        # Check if all municipalities in merged_df are present in the profile DataFrames
        missing_municipalities = set(merged_df['municipality']) - set(commercial_profiles_df_transposed.index)
        if missing_municipalities:
            raise KeyError(f"Municipalities {missing_municipalities} are missing in the commercial profiles DataFrame")

        missing_municipalities = set(merged_df['municipality']) - set(residential_profiles_df_transposed.index)
        if missing_municipalities:
            raise KeyError(f"Municipalities {missing_municipalities} are missing in the residential profiles DataFrame")

        total_load_profile = np.zeros(8760)
        total_peak_power = 0

        # Process data in chunks
        for start in range(0, len(merged_df), chunk_size):
            print(f"Processing chunk {start // chunk_size + 1} of {len(merged_df) // chunk_size + 1}")
            end = min(start + chunk_size, len(merged_df))
            chunk = merged_df.iloc[start:end]

            # Get commercial and residential profiles for each municipality
            commercial_profiles = commercial_profiles_df_transposed.loc[chunk['municipality']].values
            residential_profiles = residential_profiles_df_transposed.loc[chunk['municipality']].values

            # Compute load profiles using vectorized operations
            load_profiles = chunk['Peak_load'].values[:, np.newaxis] * (
                    chunk['com_percentage'].values[:, np.newaxis] * commercial_profiles +
                    chunk['res_percentage'].values[:, np.newaxis] * residential_profiles
            )

            total_load_profile += load_profiles.sum(axis=0)
            total_peak_power += chunk['Peak_load'].sum()

        # Save the yearly consumption to a CSV file
        output_dir = os.path.join('Plotting_data', 'Hourly_DERs_profiles')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'total_LV_cl_yearly_consumption_{self.simulation_year}.csv')
        yearly_consumption_df = pd.DataFrame(total_load_profile, columns=['Yearly Consumption (MW)'])
        yearly_consumption_df.to_csv(output_file, index=False)

        print(f"Yearly LV Load consumption saved to {output_file}")

        # Compute monthly consumption for EVs
        def compute_monthly_consumption(consumption):
            days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            hours_in_month = [days * 24 for days in days_in_month]
            monthly_consumption = []
            start = 0
            for hours in hours_in_month:
                monthly_consumption.append(sum(consumption[start:start + hours]))
                start += hours
            return monthly_consumption

        lv_monthly_consumption = compute_monthly_consumption(total_load_profile)

        # Convert to TWh
        lv_monthly_consumption_twh = [consumption / 1000000 for consumption in lv_monthly_consumption]

        # Save the monthly consumption to a CSV file
        output_dir = os.path.join('Plotting_data', 'Monthly_DERs_heatmap')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'total_LV_cl_consumption_{self.simulation_year}.csv')
        monthly_consumption_df = pd.DataFrame([lv_monthly_consumption_twh], columns=[
            'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
            'November', 'December'
        ])
        monthly_consumption_df.to_csv(output_file, index=False)

        print(f"Monthly LV consumption saved to {output_file}")

        total_energy_consumption = total_load_profile.sum() / 1000  # Convert to GWh
        total_peak_power_mw = total_peak_power  # Convert to MW

        print(f"Total LV energy consumption: {total_energy_consumption:.2f} GWh")
        print(f"Total peak power: {total_peak_power_mw:.2f} MW")

        plt.figure(figsize=(12, 6))
        plt.plot(total_load_profile, label='Total LV Power Profile')
        plt.xlabel('Hour of the Year')
        plt.ylabel('Power (MW)')
        plt.title('Total LV Power Profile')
        plt.legend()
        plt.show()

    def compute_mv_load_profiles(self):
        # Ensure the MV load profile is loaded
        if self.mv_load_profile_df is None or self.mv_grid_data.empty:
            raise ValueError("MV load profile or MV grid data is not loaded")

        # Extract the power profile
        power_profile = self.mv_load_profile_df['Power_pu'].values

        total_load_profile = np.zeros(len(power_profile))
        total_peak_power = 0

        # Compute load profiles for each MV grid
        for _, row in self.mv_grid_data.iterrows():
            peak_load = row['Peak_load']
            load_profile = peak_load * power_profile
            total_load_profile += load_profile
            total_peak_power += peak_load

        total_energy_consumption = total_load_profile.sum() / 1000  # Convert to GWh
        total_peak_power_mw = total_peak_power  # Convert to MW

        print(f"Total MV energy consumption: {total_energy_consumption:.2f} GWh")
        print(f"Total peak power: {total_peak_power_mw:.2f} MW")

        plt.figure(figsize=(12, 6))
        plt.plot(total_load_profile, label='Total MV Power Profile')
        plt.xlabel('Hour of the Year')
        plt.ylabel('Power (MW)')
        plt.title('Total MV Power Profile')
        plt.legend()
        plt.show()

        # Save the yearly consumption to a CSV file
        output_dir = os.path.join('Plotting_data', 'Hourly_DERs_profiles')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'total_MV_cl_yearly_consumption_{self.simulation_year}.csv')
        yearly_consumption_df = pd.DataFrame(total_load_profile, columns=['Yearly Consumption (MW)'])
        yearly_consumption_df.to_csv(output_file, index=False)

        print(f"Yearly MV Load consumption saved to {output_file}")

        # Compute monthly consumption for EVs
        def compute_monthly_consumption(consumption):
            days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            hours_in_month = [days * 24 for days in days_in_month]
            monthly_consumption = []
            start = 0
            for hours in hours_in_month:
                monthly_consumption.append(sum(consumption[start:start + hours]))
                start += hours
            return monthly_consumption

        mv_monthly_consumption = compute_monthly_consumption(total_load_profile)

        # Convert to TWh
        mv_monthly_consumption_twh = [consumption / 1000000 for consumption in mv_monthly_consumption]

        # Save the monthly consumption to a CSV file
        output_dir = os.path.join('Plotting_data', 'Monthly_DERs_heatmap')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'total_MV_cl_consumption_{self.simulation_year}.csv')
        monthly_consumption_df = pd.DataFrame([mv_monthly_consumption_twh], columns=[
            'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
            'November', 'December'
        ])
        monthly_consumption_df.to_csv(output_file, index=False)

        print(f"Monthly MV consumption saved to {output_file}")

    def load_and_modify_pv_data(self, year):
        base_path = f'PV/PV_output/{year}'
        output_path = f'SwissPDG_DERs/01_PV/{year}'
        os.makedirs(output_path, exist_ok=True)

        files = ['LV_P_installed.csv', 'LV_generation.csv', 'LV_std.csv',
                 'MV_generation.csv', 'MV_P_installed.csv', 'MV_std.csv']

        for file in files:
            df = pd.read_csv(os.path.join(base_path, file))
            if 'P_installed' in file:
                df.rename(columns={'P_installed (kWp)': 'P_installed_kW'}, inplace=True)
            else:
                # Get the column names to be modified
                columns_to_modify = df.columns[2:]
                # Remove the year from the datetime columns
                new_columns = [col[5:] if '-' in col else col for col in columns_to_modify]
                df.columns = list(df.columns[:2]) + new_columns
                df[new_columns] = df[new_columns].astype('float64') / 1000.0  # Divide profiles by 1000
            if 'LV_osmid' in df.columns:
                df['LV_osmid'] = df['LV_osmid'].astype(int)
            if 'MV_osmid' in df.columns:
                df['MV_osmid'] = df['MV_osmid'].astype(int)
            df.to_csv(os.path.join(output_path, file), index=False)

        print(f"Loaded and modified PV data for {year}")

    def load_and_modify_hp_data(self, year):
        base_path = f'HP/HP_output/{year}'
        output_path = f'SwissPDG_DERs/03_HP/{year}'
        os.makedirs(output_path, exist_ok=True)

        files = ['LV_heat_pump_allocation.csv', 'MV_heat_pump_allocation.csv', 'Temperature_profiles.csv']

        for file in files:
            df = pd.read_csv(os.path.join(base_path, file))
            if 'heat_pump_allocation' in file:
                df.rename(columns={'PRT_kW': 'Nominal_power_kW', 'CBLD_KWh/K': 'Thermal_capacitance_KWh/K',
                                   'HBLD_kW/K': 'Thermal_conductivity_kW/K', 'T_PROFILE': 'Temperature_profile_name'},
                          inplace=True)
                if 'LV_osmid' in df.columns:
                    df['LV_osmid'] = df['LV_osmid'].astype(int)
                if 'MV_osmid' in df.columns:
                    df['MV_osmid'] = df['MV_osmid'].astype(int)
                df.to_csv(os.path.join(output_path, file), index=False)
            elif 'Temperature_profiles' in file:
                df = df.transpose()
                df.columns = df.iloc[0]
                df = df[1:]
                df.columns = pd.date_range(start=f'{year}-01-01', periods=8760, freq='h').strftime('%m-%d %H:%M:%S')
                df.rename_axis('Temperature_profile_name', inplace=True)
                df.to_csv(os.path.join(output_path, file), index=True)

        print(f"Loaded and modified HP data for {year}")

    def load_and_modify_bess_data(self, year):
        base_path = f'BESS/Output/{year}'
        output_path = f'SwissPDG_DERs/02_BESS/{year}'
        os.makedirs(output_path, exist_ok=True)

        files = ['BESS_allocation_LV.csv', 'BESS_allocation_MV.csv']

        for file in files:
            df = pd.read_csv(os.path.join(base_path, file))
            df.rename(columns={'Battery_Capacity_kWh': 'Battery_capacity_kWh', 'Nominal_Power_kW': 'Nominal_power_kW',
                               'Charging_Efficiency': 'Charging_efficiency', 'Discharging_Efficiency': 'Discharging_efficiency'}, inplace=True)
            if 'LV_osmid' in df.columns:
                df['LV_osmid'] = df['LV_osmid'].astype(int)
            if 'MV_osmid' in df.columns:
                df['MV_osmid'] = df['MV_osmid'].astype(int)
            df.to_csv(os.path.join(output_path, file), index=False)

        print(f"Loaded and modified BESS data for {year}")

    def load_and_modify_ev_data(self, year):
        base_path = f'EV/EV_output/{year}'
        output_path = f'SwissPDG_DERs/04_EV/{year}'
        os.makedirs(output_path, exist_ok=True)

        files = ['EV_allocation_LV.csv', 'EV_flexible_energy_profiles_LV.csv',
                 'EV_power_profiles_LV.csv']

        for file in files:
            df = pd.read_csv(os.path.join(base_path, file))
            if 'EV_allocation_LV' in file:
                df.rename(columns={'grid_name': 'LV_grid', 'node_name': 'LV_osmid', 'percentage': 'EV_share'},
                          inplace=True)
            elif 'EV_energy_profiles_LV' in file or 'EV_flexible_energy_profiles_LV' in file:
                df.rename(columns={'BFS': 'BFS_municipality_code'}, inplace=True)
            elif 'EV_power_profiles_LV' in file:
                df.rename(columns={'BFS': 'BFS_municipality_code', 'Type': 'Profile_type'}, inplace=True)

            if 'EV_flexible_energy_profiles_LV' in file:
                df.columns = [df.columns[0]] + pd.date_range(start=f'{year}-01-01', periods=365, freq='d').strftime('%m-%d').tolist()
            elif 'EV_power_profiles_LV' in file:
                df.columns = [df.columns[0], df.columns[1]] + pd.date_range(start=f'{year}-01-01', periods=8760, freq='h').strftime('%m-%d %H:%M:%S').tolist()

            if 'LV_osmid' in df.columns:
                df['LV_osmid'] = df['LV_osmid'].astype(int)
            if 'MV_osmid' in df.columns:
                df['MV_osmid'] = df['MV_osmid'].astype(int)

            df.to_csv(os.path.join(output_path, file), index=False)

        print(f"Loaded and modified EV data for {year}")

    def load_and_modify_lv_load_data(self, year):
        base_path = f'LV_basicload/LV_basicload_output/{year}'
        output_path = f'SwissPDG_DERs/05_Demand/{year}'
        os.makedirs(output_path, exist_ok=True)

        files = ['LV_basicload_shares.csv', 'Commercial_profiles.csv', 'Residential_profiles.csv']

        for file in files:
            df = pd.read_csv(os.path.join(base_path, file))
            if 'LV_basicload_shares' in file:
                df.rename(
                    columns={'com_percentage': 'Commercial_demand_share', 'res_percentage': 'Residential_demand_share'},
                    inplace=True)
                if 'LV_osmid' in df.columns:
                    df['LV_osmid'] = df['LV_osmid'].astype(int)
                if 'MV_osmid' in df.columns:
                    df['MV_osmid'] = df['MV_osmid'].astype(int)
                df.to_csv(os.path.join(output_path, file), index=False)
            else:
                df = df.transpose()
                df.columns = pd.date_range(start=f'{year}-01-01', periods=8760, freq='h').strftime('%m-%d %H:%M:%S').tolist()
                df.rename_axis('BSF_municipality_code', inplace=True)
                if 'LV_osmid' in df.columns:
                    df['LV_osmid'] = df['LV_osmid'].astype(int)
                if 'MV_osmid' in df.columns:
                    df['MV_osmid'] = df['MV_osmid'].astype(int)
                df.to_csv(os.path.join(output_path, file), index=True)

        print(f"Loaded and modified LV load data for {year}")

    def load_and_modify_mv_load_data(self, year):
        base_path = f'MV_basicload/MV_basicload_output/{year}'
        output_path = f'SwissPDG_DERs/05_Demand/{year}'
        os.makedirs(output_path, exist_ok=True)

        file = 'MV_load_profile.csv'
        df = pd.read_csv(os.path.join(base_path, file))

        # Remove the Power_pu header
        df.columns = df.iloc[0]
        # df = df[1:]

        # Set the index row to dates from 1 to 8760 in "month-day hour" format
        df.index = pd.date_range(start=f'{year}-01-01', periods=8760, freq='h').strftime('%m-%d %H:%M:%S')

        # Transpose the DataFrame
        df = df.transpose()

        # Save the modified DataFrame back to the CSV file
        df.to_csv(os.path.join(output_path, file), index=False, header=True)

        print(f"Loaded and modified MV load data for {year}")

    def create_output_directory(self):
        for year in self.simulation_years:
            self.load_and_modify_pv_data(year)
            # self.load_and_modify_hp_data(year)
            # self.load_and_modify_bess_data(year)
            # self.load_and_modify_ev_data(year)
            # self.load_and_modify_lv_load_data(year)
            # self.load_and_modify_mv_load_data(year)

    def calculate_correlation_coefficients(self):
        # Load the geojson file
        nine_zones = gpd.read_file('Plotting_data/nine_zones.geojson')
        nine_zones['BFS_NUMMER'] = nine_zones['BFS_NUMMER'].astype(int)
        nine_zones['POP'] = nine_zones['POP'].fillna(0).astype(int)

        # Load the CSV files
        pv_capacity = pd.read_csv('Plotting_data/Switzerland_maps/total_PV_capacity_per_municipality.csv')
        bess_capacity = pd.read_csv('Plotting_data/Switzerland_maps/total_BESS_capacity_per_municipality.csv')
        ev_capacity = pd.read_csv('Plotting_data/Switzerland_maps/total_EV_capacity_per_municipality.csv')
        hp_capacity = pd.read_csv('Plotting_data/Switzerland_maps/total_HP_capacity_per_municipality.csv')

        # Rename columns for consistency
        pv_capacity.columns = ['municipality', 'PV_capacity', 'None']
        bess_capacity.columns = ['municipality', 'BESS_capacity']
        ev_capacity.columns = ['municipality', 'EV_capacity']
        hp_capacity.columns = ['municipality', 'HP_capacity']

        # Merge dataframes with nine_zones
        merged_df = nine_zones.merge(pv_capacity, left_on='BFS_NUMMER', right_on='municipality', how='left')
        merged_df = merged_df.merge(bess_capacity, on='municipality', how='left')
        merged_df = merged_df.merge(ev_capacity, on='municipality', how='left')
        merged_df = merged_df.merge(hp_capacity, on='municipality', how='left')

        # Drop municipalities with no installations
        merged_df.dropna(subset=['PV_capacity', 'BESS_capacity', 'EV_capacity', 'HP_capacity'], how='all',
                         inplace=True)

        # Calculate Pearson and Spearman correlation coefficients
        correlations = {
            'Pearson': {
                'PV': pearsonr(merged_df['POP'], merged_df['PV_capacity'].fillna(0))[0],
                'BESS': pearsonr(merged_df['POP'], merged_df['BESS_capacity'].fillna(0))[0],
                'EV': pearsonr(merged_df['POP'], merged_df['EV_capacity'].fillna(0))[0],
                'HP': pearsonr(merged_df['POP'], merged_df['HP_capacity'].fillna(0))[0]
            },
            'Spearman': {
                'PV': spearmanr(merged_df['POP'], merged_df['PV_capacity'].fillna(0))[0],
                'BESS': spearmanr(merged_df['POP'], merged_df['BESS_capacity'].fillna(0))[0],
                'EV': spearmanr(merged_df['POP'], merged_df['EV_capacity'].fillna(0))[0],
                'HP': spearmanr(merged_df['POP'], merged_df['HP_capacity'].fillna(0))[0]
            }
        }

        # Save correlation coefficients to CSV
        corr_df = pd.DataFrame(correlations)
        corr_df.to_csv('Plotting_data/Switzerland_maps/correlation_coefficients.csv')

        # Set the font globally
        font_path = 'C:/Windows/Fonts'
        font_files = font_manager.findSystemFonts(fontpaths=[font_path])
        for font_file in font_files:
            font_manager.fontManager.addfont(font_file)
        rcParams['font.family'] = 'CMU Bright'

        # Create and save individual plots
        plot_params = [
            ('PV_capacity', 'Population vs PV Capacity'),
            ('BESS_capacity', 'Population vs BESS Capacity'),
            ('EV_capacity', 'Population vs EV Capacity'),
            ('HP_capacity', 'Population vs HP Capacity')
        ]

        for i, (y_var, title) in enumerate(plot_params):
            fig, ax = plt.subplots(figsize=(3, 3))
            sns.regplot(x='POP', y=y_var, data=merged_df, ax=ax, scatter_kws={'alpha': 0.5}, fit_reg=False)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(title)
            ax.set_xlabel('Population')
            ax.set_ylabel(y_var.replace('_', ' ').title())
            plt.tight_layout()
            plt.savefig(f'Plotting_data/Switzerland_maps/{y_var}_vs_population.svg', format='svg')
            plt.show()

        # Calculate and save correlation matrices among installations
        installations = merged_df[['PV_capacity', 'BESS_capacity', 'EV_capacity', 'HP_capacity']].fillna(0)
        pearson_corr_matrix = installations.corr(method='pearson')
        spearman_corr_matrix = installations.corr(method='spearman')

        pearson_corr_matrix.to_csv('Plotting_data/Switzerland_maps/pearson_correlation_matrix.csv')
        spearman_corr_matrix.to_csv('Plotting_data/Switzerland_maps/spearman_correlation_matrix.csv')

        # Plot and save heatmaps for correlation matrices
        for corr_matrix, name in [(pearson_corr_matrix, 'pearson'), (spearman_corr_matrix, 'spearman')]:
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax, square=True)
            ax.set_title(f'{name.capitalize()} Correlation Matrix')
            plt.tight_layout()
            plt.savefig(f'Plotting_data/Switzerland_maps/{name}_correlation_matrix.svg', format='svg')
            plt.show()

if __name__ == "__main__":
    checker = DataChecker()
    checker.simulation_year = 2050
    checker.create_output_directory()
    # checker.load_data()
    # checker.calculate_correlation_coefficients()
    # checker.compute_lv_load_profiles()
    # checker.compute_mv_load_profiles()
    # checker.compute_bess_statistics_and_plot()
    # checker.analyze_ev_data()
    # checker.compute_hp_consumption()
    # checker.plot_hp_histograms()
    # checker.compute_pv_power_and_generation()
    # checker.check_installed_power_vs_peak_production()
    # checker.plot_histograms()
    # checker.plot_average_equivalent_hours_per_municipality()
    # checker.plot_installed_capacity_per_municipality()
    # checker.plot_yearly_generation_per_municipality()
    # checker.check_installed_power_vs_peak_production()
    # checker.plot_cov_histogram()
    # checker.compute_pv_power_and_generation()
    # checker.check_lv_grid_consistency()
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LogNorm
from matplotlib import rcParams, font_manager

class Plotter:
    def __init__(self, year, cmap='Greys', figsize=(3, 3), font_path=None):
        self.year = year
        self.cmap = plt.get_cmap(cmap)
        self.data_dir = 'Monthly_DERs_heatmap'
        self.hourly_data_dir = 'Hourly_DERs_profiles'
        self.figsize = figsize
        self.font_path = font_path
        self.profiles = {
            'PV': f'total_pv_generation_{year}.csv',
            'HP': f'total_hp_consumption_{year}.csv',
            'EV': f'total_ev_consumption_{year}.csv',
            'LV_CL': f'total_LV_cl_consumption_{year}.csv',
            'MV_CL': f'total_MV_cl_consumption_{year}.csv'
        }
        self.profiles_hourly = {
            'PV': f'total_pv_hourly_generation_{year}.csv',
            'HP': f'total_hp_yearly_consumption_{year}.csv',
            'EV': f'total_ev_yearly_consumption_{year}.csv',
            'LV_CL': f'total_LV_cl_yearly_consumption_{year}.csv',
            'MV_CL': f'total_MV_cl_yearly_consumption_{year}.csv'
        }
        self.months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        self.data = {}
        self.hourly_data = {}
        self.color_palette = ["#ffd45b", "#f6b44d", "#db9671", "#ba7a8d", "#8d5eaa", "#3940bb", "#0033b0", "#b4e5a2"]

        # Set the font globally if the font path is provided
        if self.font_path:
            font_files = font_manager.findSystemFonts(fontpaths=[self.font_path])
            for font_file in font_files:
                font_manager.fontManager.addfont(font_file)
            rcParams['font.family'] = 'CMU Bright'

    def load_data(self):
        for key, file_name in self.profiles.items():
            file_path = os.path.join(self.data_dir, file_name)
            if os.path.exists(file_path):
                self.data[key] = pd.read_csv(file_path).values.flatten()
            else:
                self.data[key] = [0] * 12  # If file does not exist, fill with zeros

    def load_hourly_data(self):
        for key, file_name in self.profiles_hourly.items():
            file_path = os.path.join(self.hourly_data_dir, file_name)
            if os.path.exists(file_path):
                self.hourly_data[key] = pd.read_csv(file_path, skiprows=1).values.flatten()
            else:
                self.hourly_data[key] = [0] * 8760  # If file does not exist, fill with zeros

    def compute_load(self):
        lv_cl = self.data.get('LV_CL', [0] * 12)
        mv_cl = self.data.get('MV_CL', [0] * 12)
        self.data['Load'] = [lv + mv for lv, mv in zip(lv_cl, mv_cl)]

    def compute_hourly_load(self):
        lv_cl = self.hourly_data.get('LV_CL', [0] * 8760)
        mv_cl = self.hourly_data.get('MV_CL', [0] * 8760)
        self.hourly_data['Load'] = [lv + mv for lv, mv in zip(lv_cl, mv_cl)]

    def plot_heatmap(self):
        fig, ax = plt.subplots(figsize=[6, 3])
        data_matrix = [self.data['PV'], self.data['HP'], self.data['EV'], self.data['Load']]
        cax = ax.imshow(data_matrix, cmap=self.cmap, norm=LogNorm(vmin=0.1, vmax=10))

        cbar = fig.colorbar(cax)
        cbar.set_label('Yearly energy consumption or generation [TWh]')

        ax.set_xticks(range(12))
        ax.set_xticklabels(self.months)
        ax.set_yticks(range(4))
        ax.set_yticklabels(['PV', 'HP', 'EV', 'Load'])

        ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
        ax.set_xticks([x - 0.5 for x in range(1, 12)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, 4)], minor=True)
        ax.tick_params(which='minor', size=0)
        for spine in ax.spines.values():
            spine.set_visible(False)  # Remove the outer black border

        # Add the reference year on the right side of the heatmap vertically
        ax.text(11.8, 1.5, str(self.year), va='center', ha='left', rotation=90, fontsize=12, color='black')

        plt.title(f'Monthly Profiles Heatmap for {self.year}')
        plt.savefig(os.path.join(os.getcwd(), 'Profiles_figure', f'heatmap_profile_{self.year}.svg'), format='svg',
                    bbox_inches='tight')
        plt.show()

    def plot_hourly_profiles_switzerland(self, day):
        start_hour = (day - 1) * 24
        end_hour = start_hour + 24
        hours = range(1, 25)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.step(hours, [x / 1000 for x in self.hourly_data['PV'][start_hour:end_hour]], label='PV Generation',
                color=self.color_palette[0], where='mid', linewidth=2)
        ax.step(hours, [x / 1000 for x in self.hourly_data['HP'][start_hour:end_hour]], label='HP Consumption',
                color=self.color_palette[2], where='mid', linewidth=2)
        ax.step(hours, [x / 1000 for x in self.hourly_data['EV'][start_hour:end_hour]], label='EV Consumption',
                color=self.color_palette[4], where='mid', linewidth=2)
        ax.step(hours, [x / 1000 for x in self.hourly_data['Load'][start_hour:end_hour]], label='Load Consumption',
                color=self.color_palette[6], where='mid', linewidth=2)

        ax.set_ylim(0, 10)
        ax.set_xlim(1, 24)
        ax.set_xticks([1] + list(range(4, 22, 4)) + [24])
        ax.set_xlabel('Hour of the day')
        ax.set_ylabel('Energy (GWh)')

        # Adjust legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False)

        plt.savefig(os.path.join(os.getcwd(), 'Profiles_figure', f'daily_energy_profile_{self.year}.svg'), format='svg',
                    bbox_inches='tight')
        plt.show()

    def plot_monthly_profiles_switzerland(self):
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.step(self.months, self.data['PV'], label='PV Generation', color=self.color_palette[0], where='mid', linewidth=2)
        ax.step(self.months, self.data['HP'], label='HP Consumption', color=self.color_palette[2], where='mid', linewidth=2)
        ax.step(self.months, self.data['EV'], label='EV Consumption', color=self.color_palette[4], where='mid', linewidth=2)
        ax.step(self.months, self.data['Load'], label='Load Consumption', color=self.color_palette[6], where='mid', linewidth=2)

        ax.set_ylim(0, 5)
        ax.set_xlim(0, 11)  # Adjust x-axis limits to start from the beginning and end of the chart
        ax.set_xticks(range(0, 12, 2))
        ax.set_xlabel('Month')
        ax.set_ylabel('Energy (TWh)')

        # Adjust legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False)

        # plt.title(f'Monthly Energy Profiles for {self.year}')
        plt.savefig(os.path.join(os.getcwd(), 'Profiles_figure', f'monthly_energy_profile_{self.year}.svg'), format='svg',
                    bbox_inches='tight')
        plt.show()

    def plot_daily_boxplot_switzerland(self):
        daily_data = {
            'PV': [sum(self.hourly_data['PV'][i:i + 24]) / 1000 for i in range(0, 8760, 24)],
            'HP': [sum(self.hourly_data['HP'][i:i + 24]) / 1000 for i in range(0, 8760, 24)],
            'EV': [sum(self.hourly_data['EV'][i:i + 24]) / 1000 for i in range(0, 8760, 24)],
            'Load': [sum(self.hourly_data['Load'][i:i + 24]) / 1000 for i in range(0, 8760, 24)]
        }

        fig, ax = plt.subplots(figsize=self.figsize)
        box = ax.boxplot([daily_data['PV'], daily_data['HP'], daily_data['EV'], daily_data['Load']],
                         tick_labels=['PV', 'HP', 'EV', 'Load'],
                         patch_artist=True,
                         medianprops=dict(color='black'))

        colors = [self.color_palette[0], self.color_palette[2], self.color_palette[4], self.color_palette[6]]

        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_ylim(0, 150)
        ax.set_ylabel('Daily energy (GWh)')
        # plt.title(f'Daily Energy Profiles for {self.year}')
        plt.savefig(os.path.join(os.getcwd(), 'Profiles_figure', f'daily_energy_boxplot_{self.year}.svg'), format='svg',
                    bbox_inches='tight')
        plt.show()

    def plot_hourly_profiles_single_grid(self, day):
        start_hour = (day - 1) * 24
        end_hour = start_hour + 24
        hours = range(1, 25)

        # Load data from CSV files
        # Load data from CSV files
        pv_profile = pd.read_csv(f'Hourly_DERs_profiles/grid_pv_production_{self.year}.csv',
                                 skiprows=1, header=None).values.flatten()  # Convert W to kW
        hp_profile = pd.read_csv(f'Hourly_DERs_profiles/grid_hp_consumption_{self.year}.csv',
                                 skiprows=1, header=None).values.flatten()
        ev_profile = pd.read_csv(f'Hourly_DERs_profiles/grid_ev_consumption_{self.year}.csv',
                                 skiprows=1, header=None).values.flatten()
        load_profile = pd.read_csv(f'Hourly_DERs_profiles/grid_non_dispatchable_load_{self.year}.csv',
                                   skiprows=1, header=None).values.flatten()

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.step(hours, pv_profile[start_hour:end_hour], label='PV Generation', color=self.color_palette[0], where='mid',
                linewidth=2)
        ax.step(hours, hp_profile[start_hour:end_hour], label='HP Consumption', color=self.color_palette[2],
                where='mid', linewidth=2)
        ax.step(hours, ev_profile[start_hour:end_hour], label='EV Consumption', color=self.color_palette[4],
                where='mid', linewidth=2)
        ax.step(hours, load_profile[start_hour:end_hour], label='Load Consumption', color=self.color_palette[6],
                where='mid', linewidth=2)

        ax.set_ylim(0, 700)
        ax.set_xlim(1, 24)
        ax.set_xticks([1] + list(range(4, 22, 4)) + [24])
        ax.set_xlabel('Hour of the day')
        ax.set_ylabel('Energy (kWh)')

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False)

        plt.savefig(os.path.join(os.getcwd(), 'Profiles_figure', f'daily_energy_profile_single_grid_{self.year}.svg'),
                    format='svg', bbox_inches='tight')
        plt.show()

    def plot_monthly_profiles_single_grid(self):
        # Load data from CSV files
        pv_profile = pd.read_csv(f'Hourly_DERs_profiles/grid_pv_production_{self.year}.csv',
                                 skiprows=1, header=None).values.flatten()  # Convert W to kW
        hp_profile = pd.read_csv(f'Hourly_DERs_profiles/grid_hp_consumption_{self.year}.csv',
                                 skiprows=1, header=None).values.flatten()
        ev_profile = pd.read_csv(f'Hourly_DERs_profiles/grid_ev_consumption_{self.year}.csv',
                                 skiprows=1, header=None).values.flatten()
        load_profile = pd.read_csv(f'Hourly_DERs_profiles/grid_non_dispatchable_load_{self.year}.csv',
                                   skiprows=1, header=None).values.flatten()

        # Compute monthly profiles
        pv_monthly = [sum(pv_profile[i:i + 730]) / 1000 for i in range(0, 8760, 730)]  # Convert kW to MWh
        hp_monthly = [sum(hp_profile[i:i + 730]) / 1000 for i in range(0, 8760, 730)]
        ev_monthly = [sum(ev_profile[i:i + 730]) / 1000 for i in range(0, 8760, 730)]
        load_monthly = [sum(load_profile[i:i + 730]) / 1000 for i in range(0, 8760, 730)]

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.step(self.months, pv_monthly, label='PV Generation', color=self.color_palette[0], where='mid', linewidth=2)
        ax.step(self.months, hp_monthly, label='HP Consumption', color=self.color_palette[2], where='mid', linewidth=2)
        ax.step(self.months, ev_monthly, label='EV Consumption', color=self.color_palette[4], where='mid', linewidth=2)
        ax.step(self.months, load_monthly, label='Load Consumption', color=self.color_palette[6], where='mid',
                linewidth=2)

        ax.set_ylim(0, 200)
        ax.set_xlim(0, 11)
        ax.set_xticks(range(0, 12, 2))
        ax.set_xlabel('Month')
        ax.set_ylabel('Energy (MWh)')

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False)

        plt.savefig(os.path.join(os.getcwd(), 'Profiles_figure', f'monthly_energy_profile_single_grid_{self.year}.svg'),
                    format='svg', bbox_inches='tight')
        plt.show()

    def plot_daily_boxplot_single_grid(self):
        # Load data from CSV files
        pv_profile = pd.read_csv(f'Hourly_DERs_profiles/grid_pv_production_{self.year}.csv',
                                 skiprows=1, header=None).values.flatten()  # Convert W to kW
        hp_profile = pd.read_csv(f'Hourly_DERs_profiles/grid_hp_consumption_{self.year}.csv',
                                 skiprows=1, header=None).values.flatten()
        ev_profile = pd.read_csv(f'Hourly_DERs_profiles/grid_ev_consumption_{self.year}.csv',
                                 skiprows=1, header=None).values.flatten()
        load_profile = pd.read_csv(f'Hourly_DERs_profiles/grid_non_dispatchable_load_{self.year}.csv',
                                   skiprows=1, header=None).values.flatten()

        # Compute daily profiles
        daily_data = {
            'PV': [sum(pv_profile[i:i + 24]) / 1000 for i in range(0, 8760, 24)],  # Convert kW to MWh
            'HP': [sum(hp_profile[i:i + 24]) / 1000 for i in range(0, 8760, 24)],
            'EV': [sum(ev_profile[i:i + 24]) / 1000 for i in range(0, 8760, 24)],
            'Load': [sum(load_profile[i:i + 24]) / 1000 for i in range(0, 8760, 24)]
        }

        fig, ax = plt.subplots(figsize=self.figsize)
        box = ax.boxplot([daily_data['PV'], daily_data['HP'], daily_data['EV'], daily_data['Load']],
                         tick_labels=['PV', 'HP', 'EV', 'Load'],
                         patch_artist=True,
                         medianprops=dict(color='black'),
                         whis=[0, 100])  # Set whiskers to the extremes

        colors = [self.color_palette[0], self.color_palette[2], self.color_palette[4], self.color_palette[6]]

        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_ylim(0, 10)
        ax.set_ylabel('Daily energy (MWh)')

        plt.savefig(os.path.join(os.getcwd(), 'Profiles_figure', f'daily_energy_boxplot_single_grid_{self.year}.svg'),
                    format='svg', bbox_inches='tight')
        plt.show()

    def plot_boxplots(self):
        base_path = os.path.join(os.path.dirname(os.getcwd()), 'SwissPDG_DERs')
        year_path = str(self.year)

        # File paths for DERs
        pv_lv_path = os.path.join(base_path, '01_PV', year_path, 'LV_P_installed.csv')
        pv_mv_path = os.path.join(base_path, '01_PV', year_path, 'MV_P_installed.csv')
        bess_lv_path = os.path.join(base_path, '02_BESS', year_path, 'BESS_allocation_LV.csv')
        bess_mv_path = os.path.join(base_path, '02_BESS', year_path, 'BESS_allocation_MV.csv')
        hp_lv_path = os.path.join(base_path, '03_HP', year_path, 'LV_heat_pump_allocation.csv')
        hp_mv_path = os.path.join(base_path, '03_HP', year_path, 'MV_heat_pump_allocation.csv')
        ev_allocation_lv_path = os.path.join(base_path, '04_EV', year_path, 'EV_allocation_LV.csv')
        ev_power_profiles_lv_path = os.path.join(base_path, '04_EV', year_path, 'EV_power_profiles_LV.csv')

        # File paths for nodal demand
        mv_grid_data_path = 'mv_grid_data.csv'
        lv_grid_data_path = 'lv_grid_data.csv'

        # Load data for DERs
        pv_lv_df = pd.read_csv(pv_lv_path)
        pv_mv_df = pd.read_csv(pv_mv_path)
        bess_lv_df = pd.read_csv(bess_lv_path)
        bess_mv_df = pd.read_csv(bess_mv_path)
        hp_lv_df = pd.read_csv(hp_lv_path)
        hp_mv_df = pd.read_csv(hp_mv_path)
        ev_allocation_lv_df = pd.read_csv(ev_allocation_lv_path)
        ev_power_profiles_lv_df = pd.read_csv(ev_power_profiles_lv_path)

        # Load data for nodal demand
        mv_grid_data_df = pd.read_csv(mv_grid_data_path)
        lv_grid_data_df = pd.read_csv(lv_grid_data_path)

        # EV peak power per municipality
        # ev_power_profiles_lv_df['BFS_municipality_code'] = ev_power_profiles_lv_df['LV_grid'].str.split('-').str[0]
        ev_peak_power = ev_power_profiles_lv_df.groupby('BFS_municipality_code').max().iloc[:, 1:].max(
            axis=1).reset_index()

        # Merge EV allocation with peak power
        ev_allocation_lv_df['BFS_municipality_code'] = ev_allocation_lv_df['LV_grid'].str.split('-').str[0].astype(int)
        ev_allocation_lv_df = ev_allocation_lv_df.merge(ev_peak_power, on='BFS_municipality_code')
        ev_allocation_lv_df['New_EV_share'] = ev_allocation_lv_df['EV_share'] * ev_allocation_lv_df.iloc[:, -1]

        # Prepare data for boxplots, discarding elements equal to zero
        lv_data = [pv_lv_df['P_installed_kW'][pv_lv_df['P_installed_kW'] != 0],
                   bess_lv_df['Nominal_power_kW'][bess_lv_df['Nominal_power_kW'] != 0],
                   hp_lv_df['Nominal_power_kW'][hp_lv_df['Nominal_power_kW'] != 0],
                   # ev_allocation_lv_df['New_EV_share'][ev_allocation_lv_df['New_EV_share'] != 0],
                   # lv_grid_data_df['Peak_load'][lv_grid_data_df['Peak_load'] != 0] * 1000
                   ]
        mv_data = [pv_mv_df['P_installed_kW'][pv_mv_df['P_installed_kW'] != 0],
                   bess_mv_df['Nominal_power_kW'][bess_mv_df['Nominal_power_kW'] != 0],
                   hp_mv_df['Nominal_power_kW'][hp_mv_df['Nominal_power_kW'] != 0],
                   #mv_grid_data_df['Peak_load'][mv_grid_data_df['Peak_load'] != 0] * 1000
                   ]

        # Plot LV boxplot
        self.plot_boxplot(lv_data, ['PV', 'BESS', 'HP',
                                    # 'EV', 'Demand'
                                    ], 'LV Nodal Power Distribution', 'LV')

        # Plot MV boxplot
        self.plot_boxplot(mv_data, ['PV', 'BESS', 'HP'
                                    # , 'Demand'
                                    ], 'MV Nodal Power Distribution', 'MV')

    def plot_boxplot(self, data, labels, title, grid_type='LV'):
        fig, ax = plt.subplots(figsize=self.figsize)
        box = ax.boxplot(data, patch_artist=True, medianprops=dict(color='black'), showfliers=False)

        # Update colors for MV and LV boxplots
        if grid_type == 'LV':
            colors = [self.color_palette[0], self.color_palette[7], self.color_palette[2], self.color_palette[4],
                      self.color_palette[6]]
            ax.set_ylim(0, 30)
        else:
            colors = [self.color_palette[0], self.color_palette[7], self.color_palette[2], self.color_palette[6]]
            ax.set_ylim(0, 600)

        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_xticklabels(labels)
        ax.set_ylabel('Nodal Power (kW)')
        ax.set_title(title)
        plt.savefig(os.path.join(os.getcwd(), 'Deployment_histograms', f'boxplot_{title}.svg'), format='svg',
                    bbox_inches='tight')
        plt.show()

    def generate_plots(self):
        # self.load_data()
        # self.load_hourly_data()
        # self.compute_load()
        # self.compute_hourly_load()
        # self.plot_heatmap()
        # self.plot_hourly_profiles_switzerland(day=70)  # Example: 1st of May is the 121st day of the year
        # self.plot_monthly_profiles_switzerland()
        # self.plot_daily_boxplot_switzerland()
        self.plot_hourly_profiles_single_grid(day=70)  # Example: 1st of May is the 121st day of the year
        self.plot_monthly_profiles_single_grid()
        self.plot_daily_boxplot_single_grid()
        # self.plot_boxplots()


# Generate plots for 2030, 2040, and 2050
for year in [2030, 2040, 2050]:
    plotter = Plotter(year, font_path='C:/Windows/Fonts')  # Update the path to the CMU Bright font
    # plotter.plot_daily_boxplot_single_grid()
    plotter.generate_plots()


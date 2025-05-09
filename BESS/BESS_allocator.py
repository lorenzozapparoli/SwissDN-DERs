"""
Author: Lorenzo Zapparoli
Institution: ETH Zurich
Date: 15/03/2025

Introduction:
This script, `BESS_allocator.py`, is designed to allocate Battery Energy Storage Systems (BESS) to Low Voltage (LV)
and Medium Voltage (MV) grids based on the installed Photovoltaic (PV) capacity. The allocation process considers
future scenarios for the years 2030, 2040, and 2050, with varying shares of PV installations equipped with batteries.

The script reads input data for PV installations, calculates battery parameters (e.g., capacity, nominal power,
charging/discharging efficiency), and generates output files containing the allocated BESS data for each grid.
The allocation ensures continuity between years by considering previously allocated nodes.

This work is part of a broader project at ETH Zurich to model and analyze Distributed Energy Resources (DERs)
in Swiss distribution grids. The input data and methodology are described in the related publication.

Usage:
1. Ensure the required PV data is available in the specified directories.
2. Run the script to generate BESS allocation results for 2030, 2040, and 2050.
3. The output files will be saved in the corresponding `BESS/Output/` directories.

Dependencies:
- pandas
- os

"""

import pandas as pd
import os


# Function to allocate batteries to LV and MV grids based on PV installations
def allocate_batteries(year, pv_share_dict, previous_year_data=None, duration_hours=2.5, round_trip_efficiency=0.85):
    """
    Allocates Battery Energy Storage Systems (BESS) to LV and MV grids based on PV installations.

    Parameters:
        year (int): The year for which the allocation is being performed (e.g., 2030, 2040, 2050).
        pv_share_dict (dict): A dictionary containing the share of PV installations with batteries for each year.
        previous_year_data (dict, optional): Data from the previous year's allocation to ensure continuity.
        duration_hours (float): The duration (in hours) for which the battery can provide power.
        round_trip_efficiency (float): The round-trip efficiency of the battery.

    Returns:
        dict: A dictionary containing the sampled LV and MV data with battery allocations.
    """

    # Define file paths for input and output data
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lv_input_path = os.path.join(base_path, f'PV\\PV_output\\{year}\\LV_P_installed.csv')
    mv_input_path = os.path.join(base_path, f'PV\\PV_output\\{year}\\MV_P_installed.csv')
    lv_output_path = os.path.join(base_path, f'BESS\\Output\\{year}\\BESS_allocation_LV.csv')
    mv_output_path = os.path.join(base_path, f'BESS\\Output\\{year}\\BESS_allocation_MV.csv')

    # Read LV and MV installed PV data
    lv_data = pd.read_csv(lv_input_path)
    mv_data = pd.read_csv(mv_input_path)

    # Get the share of PV installations with a battery for the given year
    pv_share = pv_share_dict[year]

    # Sample the dataframe based on the pv_share
    if previous_year_data is not None:
        lv_previous = previous_year_data['LV']
        mv_previous = previous_year_data['MV']

        # Add new nodes to meet the new share
        lv_sampled = lv_data[lv_data[['LV_grid', 'LV_osmid']].apply(tuple, axis=1).isin(
            lv_previous[['LV_grid', 'LV_osmid']].apply(tuple, axis=1))].copy()
        mv_sampled = mv_data[mv_data[['MV_grid', 'MV_osmid']].apply(tuple, axis=1).isin(
            mv_previous[['MV_grid', 'MV_osmid']].apply(tuple, axis=1))].copy()
        lv_remaining = lv_data[~lv_data[['LV_grid', 'LV_osmid']].apply(tuple, axis=1).isin(
            lv_previous[['LV_grid', 'LV_osmid']].apply(tuple, axis=1))]
        mv_remaining = mv_data[~mv_data[['MV_grid', 'MV_osmid']].apply(tuple, axis=1).isin(
            mv_previous[['MV_grid', 'MV_osmid']].apply(tuple, axis=1))]

        lv_additional = lv_remaining.sample(frac=((pv_share - len(lv_previous) / len(lv_data)) / (1 - len(lv_previous) / len(lv_data))), random_state=1)
        mv_additional = mv_remaining.sample(frac=((pv_share - len(mv_previous) / len(mv_data)) / (1 - len(mv_previous) / len(mv_data))), random_state=1)

        lv_sampled = pd.concat([lv_sampled, lv_additional])
        mv_sampled = pd.concat([mv_sampled, mv_additional])
    else:
        lv_sampled = lv_data.sample(frac=pv_share, random_state=1)
        mv_sampled = mv_data.sample(frac=pv_share, random_state=1)

    # Calculate battery parameters
    def calculate_battery_parameters(df):
        """
        Calculates battery parameters such as capacity, nominal power, and efficiencies.

        Parameters:
            df (DataFrame): The input DataFrame containing PV installation data.

        Returns:
            DataFrame: The updated DataFrame with battery parameters.
        """

        df['Battery_Capacity_kWh'] = df['P_installed (kWp)'] * duration_hours
        df['Nominal_Power_kW'] = df['P_installed (kWp)']
        df['Charging_Efficiency'] = round_trip_efficiency ** 0.5
        df['Discharging_Efficiency'] = round_trip_efficiency ** 0.5
        return df

    # Apply calculations to LV and MV data
    lv_battery_data = calculate_battery_parameters(lv_sampled)
    mv_battery_data = calculate_battery_parameters(mv_sampled)

    # Select relevant columns
    lv_battery_data = lv_battery_data[
        ['LV_grid', 'LV_osmid', 'Battery_Capacity_kWh', 'Nominal_Power_kW', 'Charging_Efficiency',
         'Discharging_Efficiency']]
    mv_battery_data = mv_battery_data[
        ['MV_grid', 'MV_osmid', 'Battery_Capacity_kWh', 'Nominal_Power_kW', 'Charging_Efficiency',
         'Discharging_Efficiency']]

    # Save to CSV files
    lv_battery_data.to_csv(lv_output_path, index=False)
    mv_battery_data.to_csv(mv_output_path, index=False)

    print('Battery allocation completed for year:', year)

    return {'LV': lv_sampled, 'MV': mv_sampled}


# Example usage of the script
if __name__ == "__main__":
    pv_shares_dict = {
        2030: 0.14 + (0.7 - 0.14) * (2030 - 2021) / (2050 - 2021),
        2040: 0.14 + (0.7 - 0.14) * (2040 - 2021) / (2050 - 2021),
        2050: 0.7
    }

    data_2030 = allocate_batteries(2030, pv_shares_dict)
    data_2040 = allocate_batteries(2040, pv_shares_dict, previous_year_data=data_2030)
    data_2050 = allocate_batteries(2050, pv_shares_dict, previous_year_data=data_2040)

    print('Battery allocation process completed for all years.')

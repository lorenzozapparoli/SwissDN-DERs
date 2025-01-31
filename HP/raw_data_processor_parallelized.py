import pandas as pd
import numpy as np
import os
import time
import geopandas as gpd
from mpi4py import MPI  
from shapely.geometry import Point

# Function to determine heat pump presence
def determine_HP_presence(penetration):
    """
    Determine if a building has a heat pump based on a given penetration.

    Args:
        rate (float): The probability of having a heat pump.

    Returns:
        int: 1 if a heat pump is present, 0 otherwise.
    """
    return 1 if np.random.rand() <= penetration else 0

def get_average_rated_power():
    P_rated_params = RLC_parameters['P_rated']
    MFH = RLC_parameters['Multi_Family_House']
    SFH = RLC_parameters['Single_Family_House']

    PU = P_rated_params['PU(W/m2)']
    PL = P_rated_params['PL(W/m2)']

    def compute_p_rated(data):
        results = []
        for i in range(len(data['Starting Year'])):
            start_year = data['Starting Year'][i]
            P_U = (data['Heavy'][i]*PU[0] + data['Medium'][i]*PU[1] + data['Light'][i]*PU[2])/100
            P_L = (data['Heavy'][i]*PL[0] + data['Medium'][i]*PL[1] + data['Light'][i]*PL[2])/100
            p_rated_val = (P_U + P_L) / 2
            results.append({'Year': start_year, 'P_rated': p_rated_val})
        return pd.DataFrame(results)

    P_rated_MFH = compute_p_rated(MFH)
    P_rated_SFH = compute_p_rated(SFH)
    return P_rated_MFH, P_rated_SFH

def determine_HP_configuration(year, is_single, is_residential):
    P_rated_MFH, P_rated_SFH = get_average_rated_power()
    if is_single:
        H_B = RLC_parameters['Building_U_SFH']['Hbldg/Area[W/m2K]']
    else:
        H_B = RLC_parameters['Building_U_MFH']['Hbldg/Area[W/m2K]']
    # Determine the appropriate parameters based on the year
    # Specific termal capacitance in kWh/m2K
    C_B_rate = 0.3 / 3.6
    # Specific thermal conductance based on the year
    years_slot = np.where(RLC_parameters['Building_HandT']['Starting Year'] <= year)[0][-1]
    H_B_rate = H_B[years_slot] # in W/m2K
    # Determine temperature parameters
    T_HK_M8_temp = RLC_parameters['Building_HandT']['T_HK_M8_temp'][years_slot]
    T_HK_15_temp = RLC_parameters['Building_HandT']['T_HK_15_temp'][years_slot]
    
    # Determine rated power
    if is_residential:
        years_slot = np.where(P_rated_SFH['Year'] <= year)[0][-1]
        if is_single:
            P_rate = P_rated_SFH['P_rated'][years_slot]
        else:
            P_rate = P_rated_MFH['P_rated'][years_slot]
    else:
        PL = RLC_parameters['P_rated']['PL(W/m2)'][3]
        PU = RLC_parameters['P_rated']['PU(W/m2)'][3]
        P_rate = (PL + PU) / 2
    
    # Determine heat pump type, specifying the probability to have air wtr geothermal or water.
    HP_type = int(np.random.rand() <= air_HP_share)
    
    return C_B_rate, H_B_rate, T_HK_M8_temp, T_HK_15_temp, P_rate, HP_type

def closest_temperature_measuring_station(X, Y):
    crs_point = "EPSG:2056"
    # Convert the point of interest to the same CRS as the GeoDataFrame
    point_of_interest = Point(X, Y)
    # Calculate distances
    temperature_stations['distance_to_point'] = temperature_stations.distance(point_of_interest)
    # Find the label with the minimum distance
    closest_label = temperature_stations.loc[temperature_stations['distance_to_point'].idxmin()]
    # Get the name of the closest label
    return closest_label['station']  # Assuming 'name' is the column with label names

def buildings_block_solver(sub_buildings):
    results = []
    for _, building in sub_buildings.iterrows():
        # Process each building
        if (building['GKAT'] == 1060) or (building['GKAT'] == 1080):
            is_Rsd = 0
        else:
            is_Rsd = 1
        # Randomly determine HP presence
        is_HP = determine_HP_presence(heat_pumps_penetration)
        C_B_rate, H_B_rate, T_HK_M8_temp, T_HK_15_temp, P_rate, HP_type = \
            determine_HP_configuration(building['GBAUJ'], building['GANZWHG'] <= 1, is_Rsd)
        
        ref_area = building['GEBF']
        T_station = closest_temperature_measuring_station(building['GKODE'], building['GKODN'])
        
        results.append((is_Rsd, is_HP, C_B_rate * ref_area, H_B_rate * ref_area, T_HK_M8_temp,
                        T_HK_15_temp, P_rate * ref_area, ref_area, HP_type, T_station))
    return results

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Set paths
        script_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_path,"HP_input", "Buildings_data", "Buildings data Switzerland")
        config_path = os.path.join(script_path,"HP_input", "configuration_files")
        tempertures_path = os.path.join(script_path,"HP_input", "Temperature_data")
        save_path = os.path.join(script_path,"HP_input", "Buildings_data")

        # Read and preprocess data
        heat_pumps_penetration = 0.3
        air_HP_share = 70.83 / 100

        ts = time.time()
        file_path = os.path.join(data_path, "gebaeude_batiment_edificio.csv")
        buildings = pd.read_csv(file_path, sep="\t", low_memory=False)
        te = time.time()
        print(f"Time to read the building file: {te - ts:.2f} s")

        file_path = os.path.join(config_path, "RLC_parameters.xlsx")
        RLC_parameters = pd.read_excel(file_path, sheet_name=[
            'Multi_Family_House', 'Single_Family_House', 'Building_C',
            'Building_HandT', 'P_rated', 'Building_U_MFH', 'Building_U_SFH'])

        file_path = os.path.join(tempertures_path, "station_locations.csv")
        temperature_stations = pd.read_csv(file_path)
        temperature_stations = gpd.GeoDataFrame(
            temperature_stations,
            geometry=gpd.points_from_xy(temperature_stations.X, temperature_stations.Y),
            crs='EPSG:2052'
        )

        # Filter and preprocess building data
        buildings = buildings[
            (buildings['GSTAT'] == 1004) & (buildings['GAREA'].notna()) & (buildings['GAREA'] > 5) &
            (buildings['GKODN'].notna()) & (buildings['GKODE'].notna())
        ]
        average_construction_year = buildings['GBAUJ'].mean()
        buildings['GBAUJ'] = buildings['GBAUJ'].fillna(average_construction_year)
        buildings['GASTW'] = buildings['GASTW'].fillna(1)
        buildings['GANZWHG'] = buildings['GANZWHG'].fillna(buildings['GASTW'])
        buildings['GEBF'] = buildings['GEBF'].fillna(buildings['GAREA'] * buildings['GASTW'])

        # Create chunks
        total_buildings = len(buildings.index)
        chunk_size = (total_buildings + size - 1) // size
        # devide total_buildings into sub_buildings
        sub_buildings = [buildings.iloc[i:i + chunk_size] for i in range(0, total_buildings, chunk_size)]
            
    else:
        sub_buildings = None
        RLC_parameters = None
        temperature_stations = None
        heat_pumps_penetration = 0.3
        air_HP_share = 70.83 / 100

    # Broadcast data and scatter chunks
    RLC_parameters = comm.bcast(RLC_parameters, root=0)
    temperature_stations = comm.bcast(temperature_stations, root=0)
    chunk_to_process = comm.scatter(sub_buildings, root=0)

    # Process building chunks in parallel
    results = buildings_block_solver(chunk_to_process)

    # Gather results at root
    gathered_results = comm.gather(results, root=0)

    if rank == 0:
        # Flatten and assign results
        final_results = [item for sublist in gathered_results for item in sublist]
        is_Rsds, is_HPs, C_B_rates, H_B_rates,  T_HK_M8_temps,\
        T_HK_15_temps, P_rates, ref_areas, HP_types, T_station = zip(*final_results)

        # Add computed parameters to the dataframe
        buildings['ISRSD'] = pd.Series(data=is_Rsds, index=buildings.index)
        buildings['ISHP'] = pd.Series(data=is_HPs, index=buildings.index)
        buildings['CBLD'] = C_B_rates
        buildings['HBLD'] = H_B_rates
        buildings['THKM8'] = T_HK_M8_temps
        buildings['THK15'] = T_HK_15_temps
        buildings['PRT'] = P_rates
        buildings['HPTYP'] = HP_types
        buildings['T_PROFILE'] = T_station

        # Save to CSV
        buildings.to_csv(
            os.path.join(save_path, 'Buildings_data_new.csv'),
            columns=['EGID', 'GKODN', 'GKODE', 'HBLD', 'CBLD', 'PRT', 'GEBF', 'GAREA', 'ISHP', 'T_PROFILE'],
            index=False
        )
    print("Finished processing")    
    MPI.Finalize()

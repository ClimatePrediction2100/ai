import xarray as xr
import numpy as np
import os

from config import ROOT_DIR

temperature_path = os.path.join(ROOT_DIR, 'data', 'raw', 'globalTemperature', 'Land_and_Ocean_LatLong1.nc')
co2_path = os.path.join(ROOT_DIR, 'data', 'raw', 'globalGhgEmissions', 'CO2_1deg_month_1850-2013.nc')

# ssp_scenario = 'SSP119', 'SSP126', 'SSP245', 'SSP370', 'SSP434', 'SSP460', 'SSP534', 'SSP585'
def DataLoader(ssp_scenario='SSP119'):
    temperature_ds = xr.open_dataset(temperature_path)
    co2_ds = xr.open_dataset(co2_path)

    temperature_ds['time'] = temperature_ds['time'].assign_coords(time=np.arange(len(temperature_ds.time)))
    co2_ds['Times'] = co2_ds['Times'].assign_coords(Times=np.arange(len(co2_ds.Times)))

    ssp_path = os.path.join(ROOT_DIR, 'data', 'raw', 'globalGhgEmissions', f'CO2_{ssp_scenario}_2015_2150.nc')
        
    ssp_ds = xr.open_dataset(ssp_path)
    ssp_ds['time'] = ssp_ds['time'].assign_coords(time=np.arange(len(ssp_ds.time)))

    # Extract temperature data
    temperature_data = temperature_ds['temperature']
    temperature_mean = temperature_data.mean()
    temperature_std = temperature_data.std()
    normalized_temperature = (temperature_data - temperature_mean) / temperature_std

    climatology = temperature_ds['climatology']
    land_mask = temperature_ds['land_mask']

    train_data_dict = {
        'temperature': normalized_temperature,
        'climatology': climatology,
        'land_mask': land_mask,
        'co2': co2_ds['value'],
        'time_length': 1968, # 1850-2013
        'scenario': 'historical',
    }

    test_data_dict = {
        'temperature': normalized_temperature[1980:1980+109],
        'climatology': climatology,
        'land_mask': land_mask,
        'co2': ssp_ds['CO2'],
        'time_length': 108,  # 2015-2023
        'scenario': f'{ssp_scenario}',
    }

    predict_data_dict = {
        'climatology': climatology,
        'land_mask': land_mask,
        'co2': ssp_ds['CO2'],
        'time_length': 1632,  # 2015-2150
        'scenario': f'{ssp_scenario}',
    }

    return train_data_dict, test_data_dict, predict_data_dict
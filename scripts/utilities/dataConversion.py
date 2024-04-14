import os
from abc import *

import xarray as xr
import numpy as np

from torch.utils.data import Dataset

# class TempertureData():
#     def __init__(self, path):

# class GHGData():
#     def __init__(self, path):

# time conversion to each dataset is needed

class TemperatureData():
    # historical January 1850 - December 2013
    # projection January 2014 - December 2150
    def __init__(self, historical_path, future_path):
        self.historical = xr.open_dataset(historical_path)
        self.projection = xr.open_dataset(future_path)

    def __len__(self):
        return self.historical.time.size + self.projection.time.size

    def __getitem__(self, idx):
        if idx['time'] < 0 or idx['time'] >= self.historical.time.size + self.projection.time.size:
            raise IndexError("Time out of range")
        
        #time format conversion is needed
        if idx['time'] < self.historical.time.size:
            deviate_temp = self.historical['temperature'].isel(time=idx['time'], latitude=idx['latitude'], longitude=idx['longitude'])
        else:
            deviate_temp = self.projection['temperature'].isel(time=idx['time'] - self.historical.time.size, latitude=idx['latitude'], longitude=idx['longitude'])
        average_temp = self.historical['climatology'].isel(time=idx['time'] % 12 + 1, latitude=idx['latitude'], longitude=idx['longitude'])
        land_mask = self.historical['land_mask'].isel(latitude=idx['latitude'], longitude=idx['longitude'])
        
        state = dict()
        state['deviate_temp'] = deviate_temp
        state['average_temp'] = average_temp
        state['land_mask'] = land_mask
        
        return state

class GHGData():
    # historical January 1850 - December 2013
    # projection January 2014 - December 2150
    def __init__(self, historical_path, ssp_path):
        self.historical = xr.open_dataset(historical_path)
        self.projection = xr.open_dataset(ssp_path)

    def __getitem__(self, idx):
        if idx['time'] < 0 or idx['time'] >= self.historical.time.size + self.projection.time.size:
            raise IndexError("Time out of range")
        
        #time format conversion is needed
        if idx['time'] < self.historical.time.size:
            ppm = self.historical['value'].isel(Times=idx['time'], LatDim=idx['latitude'], LonDim=idx['longitude'])
        else:
            ppm = self.projection.isel(time=idx['time'] - self.historical.time.size, latitude=idx['latitude'], longitude=idx['longitude'])
        
        state = dict()
        state['co2_ppm'] = ppm
        
        return state

class GlobalClimateData(Dataset):
    def __init__(self, temperature_path, ghg_path):
        self.temperatureData = TemperatureData(temperature_path['historical'], temperature_path['projection'])
        self.ghgData = GHGData(ghg_path['historical'], ghg_path['projection'])
        
    def __len__(self):
        return len(self.temperatureData)
    
    def __getitem__(self, idx):
        temperature = self.temperatureData[idx]
        ghg = self.ghgData[idx]
        return temperature | ghg
    
    
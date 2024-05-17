import torch
from torch.utils.data import Dataset
import random
import numpy as np

def convert_months():
    if not hasattr(convert_months, "cos_months") or not hasattr(convert_months, "sin_months"):
        # Compute only once
        angles = 2 * np.pi * np.arange(12) / 12
        convert_months.cos_months = np.cos(angles)
        convert_months.sin_months = np.sin(angles)

    def computation(months):
        months = np.array(months) % 12
        cos_values = convert_months.cos_months[months]
        sin_values = convert_months.sin_months[months]
        return cos_values, sin_values

    return computation

class TestData(Dataset):
    def __init__(self, data_dict, seq_length):
        """
        Data between Jan 2015 and Dec 2023
        Initializes the dataset with a dictionary of data variables.
        :param data_dict: Dictionary of data variables (e.g., {'temperature': ..., 'land_mask': ...}).
        :param seq_length: Number of historical steps used as input to the model.
        """
        self.data_dict = data_dict
        self.seq_length = seq_length
        # Assumes all data variables share the same latitude and longitude coordinates
        self.latitudes = data_dict['land_mask']['latitude'].values
        self.longitudes = data_dict['land_mask']['longitude'].values
        
        self.time_steps = data_dict['time_length'] - seq_length + 1
        self.month_converter = convert_months()

    def __len__(self):
        return 1000  # Randomly chosen number

    def __getitem__(self, index):
        while True:
            # Randomly select a location
            lat_idx = random.randint(0, 179)
            lon_idx = random.randint(0, 359)
            start_time = random.randint(0, self.time_steps - 1)  # Randomly select a starting time step

            temperature_sequence = self.data_dict['temperature'].isel(latitude=lat_idx, longitude=lon_idx).values
            
            x_temp = temperature_sequence[start_time:start_time + self.seq_length]
            y = temperature_sequence[start_time + self.seq_length]
            
            #check NaN values
            if np.isnan(x_temp).any() or np.isnan(y):
                continue
            
            co2_sequence = self.data_dict['co2'].isel(latitude=lat_idx, longitude=lon_idx).values
            x_co2 = co2_sequence[start_time:start_time + self.seq_length]
            x_combined = list(zip(x_temp, x_co2))
            
            land_mask = self.data_dict['land_mask'].isel(latitude=lat_idx, longitude=lon_idx).values.item()  # Get scalar value
            lat_norm = (lat_idx - 89.5) / 90

            months = np.arange(start_time, start_time + self.seq_length)
            cos_months, sin_months = self.month_converter(months)

            # Prepare input features for each timestep
            x_features = [np.append(np.array([temp[0], temp[1]]), [land_mask, lat_norm, cos_months[i], sin_months[i]]) for i, temp in enumerate(x_combined)]
            x_concat = np.stack(x_features)  # Stack to form a 2D array where each row is a timestep
            
            return torch.tensor(x_concat, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class TrainData(Dataset):
    def __init__(self, data_dict, seq_length):
        """
        Data between Jan 1850 and Dec 2013
        Initializes the dataset with a dictionary of data variables.
        :param data_dict: Dictionary of data variables (e.g., {'temperature': ..., 'land_mask': ...}).
        :param seq_length: Number of historical steps used as input to the model.
        """
        self.data_dict = data_dict
        self.seq_length = seq_length
        # Assumes all data variables share the same latitude and longitude coordinates
        self.latitudes = data_dict['land_mask']['latitude'].values
        self.longitudes = data_dict['land_mask']['longitude'].values
        
        self.time_steps = data_dict['time_length'] - seq_length + 1
        self.month_converter = convert_months()
        

    def __len__(self):
        return 10000  # Randomly chosen number

    def __getitem__(self, index):
        while True:
            # Randomly select a location
            lat_idx = random.randint(0, 179)
            lon_idx = random.randint(0, 359)
            start_time = random.randint(1300, self.time_steps - 1)  # Randomly select a starting time step

            temperature_sequence = self.data_dict['temperature'].isel(latitude=lat_idx, longitude=lon_idx).values
            
            x_temp = temperature_sequence[start_time:start_time + self.seq_length]
            y = temperature_sequence[start_time + self.seq_length]
            
            #check NaN values
            if np.isnan(x_temp).any() or np.isnan(y):
                continue
            
            co2_sequence = self.data_dict['co2'].isel(LatDim=179-lat_idx, LonDim=359-lon_idx).values
            x_co2 = co2_sequence[start_time:start_time + self.seq_length]
            x_combined = list(zip(x_temp, x_co2))
            
            land_mask = self.data_dict['land_mask'].isel(latitude=lat_idx, longitude=lon_idx).values.item()  # Get scalar value
            lat_norm = (lat_idx - 89.5) / 90
            
            months = np.arange(start_time, start_time + self.seq_length)
            cos_months, sin_months = self.month_converter(months)

            # Prepare input features for each timestep
            x_features = [np.append(np.array([temp[0], temp[1]]), [land_mask, lat_norm, cos_months[i], sin_months[i]]) for i, temp in enumerate(x_combined)]
            x_concat = np.stack(x_features)  # Stack to form a 2D array where each row is a timestep
            
            return torch.tensor(x_concat, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

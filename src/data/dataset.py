import torch
from torch.utils.data import Dataset
import random
import numpy as np

# Precompute cos and sin values
angles = 2 * np.pi * np.arange(12) / 12
cos_months = np.cos(angles)
sin_months = np.sin(angles)

def convert_months(months):
    months = np.array(months) % 12
    cos_values = cos_months[months]
    sin_values = sin_months[months]
    return cos_values, sin_values

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
        # self.month_converter = convert_months()

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
            cos_months, sin_months = convert_months(months)

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
        # self.month_converter = convert_months()
        

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
            cos_months, sin_months = convert_months(months)

            # Prepare input features for each timestep
            x_features = [np.append(np.array([temp[0], temp[1]]), [land_mask, lat_norm, cos_months[i], sin_months[i]]) for i, temp in enumerate(x_combined)]
            x_concat = np.stack(x_features)  # Stack to form a 2D array where each row is a timestep
            
            return torch.tensor(x_concat, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class PredictData(Dataset):
    def __init__(self, data_dict, seq_length, nc_file):
        self.data_dict = data_dict
        self.seq_length = seq_length
        self.latitudes = data_dict['land_mask']['latitude'].values
        self.longitudes = data_dict['land_mask']['longitude'].values
        self.nc_file = nc_file

        # Adjust the time steps to prevent negative indexing
        self.time_steps = data_dict['time_length'] - seq_length + 1

    def __len__(self):
        return len(self.latitudes) * len(self.longitudes) * self.time_steps

    def __getitem__(self, index):
        total_locations = len(self.latitudes) * len(self.longitudes)
        time_idx = index // total_locations
        loc_idx = index % total_locations
        lat_idx = loc_idx // len(self.longitudes)
        lon_idx = loc_idx % len(self.longitudes)

        # Accessing CO2 and land_mask data
        co2_sequence = self.data_dict['co2'].isel(latitude=lat_idx, longitude=lon_idx).values
        land_mask = self.data_dict['land_mask'].isel(latitude=lat_idx, longitude=lon_idx).values.item()
        lat_norm = (lat_idx - 89.5) / 90

        # Safely computing start and end indices for sequences
        start_time = time_idx
        end_time = start_time + self.seq_length

        # Getting temperature and CO2 sequences
        x_temp = self.nc_file.variables['temperature'][start_time:end_time, lat_idx, lon_idx]
        x_co2 = co2_sequence[start_time:end_time]
        x_combined = np.column_stack((x_temp, x_co2))
        
        months = np.arange(time_idx, time_idx + self.seq_length)
        cos_months, sin_months = convert_months(months)
        
        # Combine all features
        x_features = np.hstack((x_combined, np.full((self.seq_length, 1), land_mask), 
                                np.full((self.seq_length, 1), lat_norm), cos_months[:, None], sin_months[:, None]))
        
        return torch.tensor(x_features, dtype=torch.float32), torch.tensor(x_temp[-1], dtype=torch.float32)
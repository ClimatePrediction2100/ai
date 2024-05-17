import torch
from torch.utils.data import Dataset
import random
import numpy as np

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

    def __len__(self):
        return 1000  # Randomly chosen number

    def __getitem__(self, index):
        while True:
            # Randomly select a location
            lat = random.choice(self.latitudes)
            lon = random.choice(self.longitudes)
            start_time = random.randint(0, self.time_steps - 1)  # Randomly select a starting time step

            temperature_sequence = self.data_dict['temperature'].sel(latitude=lat, longitude=lon, method="nearest").values
            
            #check NaN values
            if np.isnan(temperature_sequence).any():
                continue
            
            co2_sequence = self.data_dict['co2'].sel(latitude=lat, longitude=lon, method="nearest").values
            
            land_mask = self.data_dict['land_mask'].sel(latitude=lat, longitude=lon, method="nearest").values.item()  # Get scalar value
            lat_norm = lat / 90

            months = [(start_time + i) % 12 for i in range(self.seq_length + 1)]  # Calculate month for each timestep
            cos_months = [np.cos(2 * np.pi * month / 12) for month in months]
            sin_months = [np.sin(2 * np.pi * month / 12) for month in months]

            x_temp = temperature_sequence[start_time:start_time + self.seq_length]
            x_co2 = co2_sequence[start_time:start_time + self.seq_length]
            x_combined = list(zip(x_temp, x_co2))
            
            y = temperature_sequence[start_time + self.seq_length]

            # if not np.isnan(x_temp).any() and not np.isnan(y):
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

    def __len__(self):
        return 10000  # Randomly chosen number

    def __getitem__(self, index):
        while True:
            # Randomly select a location
            lat = random.choice(self.latitudes)
            lon = random.choice(self.longitudes)
            start_time = random.randint(0, self.time_steps - 1)  # Randomly select a starting time step

            temperature_sequence = self.data_dict['temperature'].sel(latitude=lat, longitude=lon, method="nearest").values
            
            #check NaN values
            if np.isnan(temperature_sequence).any():
                continue
            
            co2_sequence = self.data_dict['co2'].sel(LatDim=int(89.5 - lat), LonDim=int(179.5 - lon)).values
            
            land_mask = self.data_dict['land_mask'].sel(latitude=lat, longitude=lon, method="nearest").values.item()  # Get scalar value
            lat_norm = lat / 90

            months = [(start_time + i) % 12 for i in range(self.seq_length + 1)]  # Calculate month for each timestep
            cos_months = [np.cos(2 * np.pi * month / 12) for month in months]
            sin_months = [np.sin(2 * np.pi * month / 12) for month in months]

            x_temp = temperature_sequence[start_time:start_time + self.seq_length]
            x_co2 = co2_sequence[start_time:start_time + self.seq_length]
            x_combined = list(zip(x_temp, x_co2))
            
            y = temperature_sequence[start_time + self.seq_length]

            # if not np.isnan(x_temp).any() and not np.isnan(y):
            # Prepare input features for each timestep
            x_features = [np.append(np.array([temp[0], temp[1]]), [land_mask, lat_norm, cos_months[i], sin_months[i]]) for i, temp in enumerate(x_combined)]
            x_concat = np.stack(x_features)  # Stack to form a 2D array where each row is a timestep
            
            return torch.tensor(x_concat, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

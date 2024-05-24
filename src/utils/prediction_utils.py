import torch
import netCDF4 as nc
import numpy as np
from tqdm import tqdm

import config
from torch.utils.data import DataLoader
from src.model import *
from src.data.dataset import *

# Precompute cos and sin values
angles = 2 * np.pi * np.arange(12) / 12
cos_months = np.cos(angles)
sin_months = np.sin(angles)

def convert_months(months):
    months = np.array(months) % 12
    cos_values = cos_months[months]
    sin_values = sin_months[months]
    return cos_values, sin_values


def initialize_netcdf_with_historical_data(source_file_path, new_file_path):
    # Open source netCDF with historical temperature data
    source_nc = nc.Dataset(source_file_path, 'r')

    # Create a new netCDF file
    new_nc = nc.Dataset(new_file_path, 'w', format='NETCDF4_CLASSIC')
    new_nc.createDimension('time', None)  # Unlimited time dimension
    new_nc.createDimension('latitude', 180)
    new_nc.createDimension('longitude', 360)
    
    times = new_nc.createVariable('time', 'i4', ('time',))
    latitudes = new_nc.createVariable('latitude', 'f4', ('latitude',))
    longitudes = new_nc.createVariable('longitude', 'f4', ('longitude',))
    # After creating the temperature variable
    temperatures = new_nc.createVariable('temperature', 'f4', ('time', 'latitude', 'longitude'))
    temperatures.units = 'degree Celsius'
    temperatures.valid_min = np.float32(-100.0)  # Example minimum, adjust as necessary
    temperatures.valid_max = np.float32(100.0)   # Example maximum, adjust as necessary
    
    # Ensure dimensions match, especially for slicing operations
    assert source_nc.variables['latitude'][:].shape[0] == 180
    assert source_nc.variables['longitude'][:].shape[0] == 360
    
    latitudes[:] = source_nc.variables['latitude'][:]
    longitudes[:] = source_nc.variables['longitude'][:]

    # Assuming the original file starts from 2015 and each month is sequentially stored
    num_months = 12 * 9  # 9 years
    times[:] = np.arange(0, num_months)  # Time index from 0 to 107
    
    # Copy temperatures in chunks to avoid memory overload
    for i in range(num_months):
        temperatures[i, :, :] = source_nc.variables['temperature'][1980 + i, :, :]

    temperatures.units = 'degree Celsius'
    latitudes.units = 'degree north'
    longitudes.units = 'degree east'
    times.units = 'months since 2015-01'

    new_nc.sync()  # Make sure the data is written to the file
    source_nc.close()
    new_nc.close()
    
    return new_file_path  # Return the path for the new file

def get_input_data(year, month, lat_idx, lon_idx, data_dict, nc_file, seq_length=12):

    co2_sequence = data_dict['co2'].isel(latitude=lat_idx, longitude=lon_idx).values
    land_mask = data_dict['land_mask'].isel(latitude=lat_idx, longitude=lon_idx).values.item()
    lat_norm = (lat_idx - 89.5) / 90

    target_time_idx = (year - 2015) * 12 + month
    start_time_idx = target_time_idx - seq_length
    if start_time_idx < 0:
        raise IndexError
    
    months = np.arange(start_time_idx, start_time_idx + seq_length)
    cos_months, sin_months = convert_months(months)

    x_temp = nc_file.variables['temperature'][start_time_idx:start_time_idx + seq_length, lat_idx, lon_idx]
    x_co2 = co2_sequence[start_time_idx:start_time_idx + seq_length]
    x_combined = list(zip(x_temp, x_co2))

    # Prepare input features for each timestep
    x_features = [np.append(np.array([temp[0], temp[1]]), [land_mask, lat_norm, cos_months[i], sin_months[i]]) for i, temp in enumerate(x_combined)]
    x_concat = np.stack(x_features)  # Stack to form a 2D array where each row is a timestep

    return x_concat
    
def predict_and_update_nc_monthly(model, nc_dataset_path, device, predict_data_dict, seq_length, start_year, end_year):
    # Open the netCDF dataset
    nc_file = nc.Dataset(nc_dataset_path, 'r+')

    model.eval()
    time_idx = 108  # Initialize time index

    for year in range(start_year, end_year + 1):
        for month in range(12):
            monthly_predictions = np.zeros((180, 360))
            # Use tqdm in the outer loop for latitude
            for lat in tqdm(range(180), desc=f"Processing Year {year}, Month {month}"):
                # Nested tqdm for longitude, you can also remove this if it's too verbose
                for lon in tqdm(range(360), desc="Longitude", leave=False):
                    input_data = get_input_data(year, month, lat, lon, predict_data_dict, nc_file, seq_length)
                    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        predicted_temp = model(input_tensor).item()
                    monthly_predictions[lat, lon] = predicted_temp

            # Update the netCDF file with the predictions for this month
            nc_file.variables['temperature'][time_idx, :, :] = monthly_predictions
            nc_file.variables['time'][time_idx] = 9 * 12 + month  # Update time index based on year and month
            time_idx += 1
            nc_file.sync()  # Ensure data is written to disk
    
    nc_file.close()
    

def denormalize_temperature(nparray):
    mean = 0.06711918
    std = 1.4353496
    nparray *= std
    nparray += mean
    
    return nparray


def initialize_model(weight_path, feature_dim):
    # Parse model type and parameters from the weight filename
    weight_name = weight_path.split("/")[-1]
    model_name, num_layers, hidden_dim, _, _, _, seq_length = weight_name.split("_")
    num_layers = int(num_layers)
    hidden_dim = int(hidden_dim)
    seq_length = int(seq_length.split(".")[0])
    
    # Select the appropriate model
    if model_name == "lstm":
        model = LSTMModel(input_dim=feature_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    elif model_name == "gru":
        model = GRUModel(input_dim=feature_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    elif model_name == "rnn":
        model = RNNModel(input_dim=feature_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    elif model_name == "mlp":
        model = MLPModel(input_dim=feature_dim*seq_length, hidden_dim=hidden_dim, num_layers=num_layers)
    elif model_name == "attn":
        model = AttentionModel(input_dim=feature_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    else:
        raise ValueError("Unsupported model type")
    
    # Load model weights
    model.load_state_dict(torch.load(weight_path, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()
    return model

def predict(predict_data, args):
    new_file_path = args.save_path
    weight_path = args.weight_path
    start_year = args.start_year
    end_year = args.end_year
    
    # Initialize netCDF with historical data
    source_file_path = "data/raw/globalTemperature/Land_and_Ocean_LatLong1.nc"
    initialize_netcdf_with_historical_data(source_file_path=source_file_path, new_file_path=new_file_path)
    seq_length =  int(weight_path.split("_")[-1].split(".")[0])
    
    # Load the dataset
    nc_file = nc.Dataset(new_file_path, 'r+')
    predict_dataset = PredictData(predict_data, seq_length, nc_file)
    predict_loader = DataLoader(predict_dataset, batch_size=360, shuffle=False, num_workers=config.NUM_WORKERS)
    
    # Prepare the model
    # feature_dim = predict_dataset[0][0].shape[1]
    feature_dim = 6
    model = initialize_model(weight_path, feature_dim)
    print(f"Model: {model.__class__.__name__}, Feature Dimension: {feature_dim}, Hidden Dimension: {model.hidden_dim}, Number of Layers: {model.num_layers}")
    
    # Perform prediction and update nc file
    predict_and_update_nc_monthly(model, new_file_path, config.DEVICE, predict_data, seq_length, start_year, end_year)

import torch
import netCDF4 as nc
import numpy as np
from tqdm import tqdm
import config
from torch.utils.data import Dataset, DataLoader


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

    months = [(month + i) % 12 for i in range(seq_length + 1)]  # Calculate month for each timestep
    cos_months = [np.cos(2 * np.pi * month / 12) for month in months]
    sin_months = [np.sin(2 * np.pi * month / 12) for month in months]

    
    target_time_index = (year - 2015) * 12 + month
    start_time_index = target_time_index - seq_length
    if start_time_index < 0:
        raise IndexError

    x_temp = nc_file.variables['temperature'][start_time_index:start_time_index + seq_length, lat_idx, lon_idx]
    x_co2 = co2_sequence[start_time_index:start_time_index + seq_length]
    x_combined = list(zip(x_temp, x_co2))

    # Prepare input features for each timestep
    x_features = [np.append(np.array([temp[0], temp[1]]), [land_mask, lat_norm, cos_months[i], sin_months[i]]) for i, temp in enumerate(x_combined)]
    x_concat = np.stack(x_features)  # Stack to form a 2D array where each row is a timestep

    return x_concat
    
# def predict_and_update_nc_monthly(model, nc_dataset_path, device, predict_data_dict, seq_length, start_year=2024, end_year=2150):
#     # Open the netCDF dataset
#     nc_file = nc.Dataset(nc_dataset_path, 'r+')

#     model.eval()
#     time_idx = 108  # Initialize time index

#     for year in range(start_year, end_year + 1):
#         for month in range(12):
#             monthly_predictions = np.zeros((180, 360))
#             # Use tqdm in the outer loop for latitude
#             for lat in tqdm(range(180), desc=f"Processing Year {year}, Month {month}"):
#                 # Nested tqdm for longitude, you can also remove this if it's too verbose
#                 for lon in tqdm(range(360), desc="Longitude", leave=False):
#                     input_data = get_input_data(year, month, lat, lon, predict_data_dict, nc_file, seq_length)
#                     input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
                    
#                     with torch.no_grad():
#                         predicted_temp = model(input_tensor).item()
#                     monthly_predictions[lat, lon] = predicted_temp

#             # Update the netCDF file with the predictions for this month
#             nc_file.variables['temperature'][time_idx, :, :] = monthly_predictions
#             nc_file.variables['time'][time_idx] = 8 * 12 + month  # Update time index based on year and month
#             time_idx += 1
#             nc_file.sync()  # Ensure data is written to disk
    
#     nc_file.close()
    
# def predict_and_update_nc_monthly(model, nc_dataset_path, device, predict_data_dict, seq_length, start_year=2024, end_year=2150):
#     # Open the netCDF dataset
#     nc_file = nc.Dataset(nc_dataset_path, 'r+')

#     model.eval()
#     time_idx = 108  # Initialize time index based on the expected start

#     # Prepare to display progress over all months and years
#     total_months = (end_year - start_year + 1) * 12
#     progress_bar = tqdm(total=total_months, desc='Overall Progress')

#     for year in range(start_year, end_year + 1):
#         for month in range(12):
#             # Prepare input data using ThreadPoolExecutor
#             input_data = []
#             with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust number of workers to your environment
#                 # Create future tasks for data preparation
#                 futures = [executor.submit(get_input_data, year, month, lat, lon, predict_data_dict, nc_file, seq_length)
#                            for lat in range(180) for lon in range(360)]
#                 for future in as_completed(futures):
#                     input_data.append(future.result())

#             # Convert list of numpy arrays to a single numpy array before creating a tensor
#             input_data_np = np.stack(input_data)  # This combines the list into one array
#             input_tensor = torch.tensor(input_data_np, dtype=torch.float32).to(device)
            
#             with torch.no_grad():
#                 output = model(input_tensor)
#                 if output.shape != (180*360,):
#                     raise ValueError(f"Model output shape mismatch, expected (64800,), got {output.shape}")
#                 output = output.view(180, 360)  # Reshape output to match the geographic grid

#             # Update the netCDF file with the predictions for this month
#             nc_file.variables['temperature'][time_idx, :, :] = output.cpu().numpy()
#             nc_file.variables['time'][time_idx] = (year - 2015) * 12 + month
#             time_idx += 1
#             nc_file.sync()  # Ensure data is written to disk

#             # Update the progress bar after each month
#             progress_bar.update(1)

#     progress_bar.close()
#     nc_file.close()
    

def denormalize_temperature(nparray):
    mean = 0.06711918
    std = 1.4353496
    nparray *= std
    nparray += mean
    
    return nparray

def predict_and_update_nc_monthly(model, predict_loader, device, nc_file, start_year=2024, end_year=2150):
    model.eval()  # Set the model to evaluation mode
    time_idx = 108  # Initialize time index based on the expected start
    
    # Total months is determined from the DataLoader's dataset length, assuming each batch corresponds to one month.
    progress_bar = tqdm(total=len(predict_loader), desc='Overall Progress')

    # Iterate through each batch provided by DataLoader
    for data, lat_idx, lon_idx, target_time_index in predict_loader:
        input_tensor = data.to(device)  # Move the data to the appropriate device (GPU or CPU)

        with torch.no_grad():
            output = model(input_tensor)
            if output.shape != (180, 360):
                raise ValueError(f"Model output shape mismatch, expected (180, 360), got {output.shape}")
            output = output.view(180, 360)  # Ensure the output is correctly reshaped

        # Update the netCDF file with the predictions for this batch
        nc_file.variables['temperature'][time_idx, :, :] = denormalize_temperature(output.cpu().numpy())
        nc_file.variables['time'][time_idx] = target_time_index  # Use the target time index directly from the batch
        time_idx += 1
        nc_file.sync()  # Ensure data is written to disk

        # Update the progress bar after each batch
        progress_bar.update(1)

    progress_bar.close()
    nc_file.close()
    
    

def initialize_model(weight_path, feature_dim):
    # Parse model type and parameters from the weight filename
    model_name, num_layers, hidden_dim, _, _, _, seq_length = weight_path.split("_")
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

def predict(predict_data, new_file_path, weight_path, start_year=2024, end_year=2150):
    # Initialize netCDF with historical data
    source_file_path = "data/raw/globalTemperature/Land_and_Ocean_LatLong1.nc"
    initialize_netcdf_with_historical_data(source_file_path=source_file_path, new_file_path=new_file_path)
    seq_length =  int(weight_path.split("_")[5])
    
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
    predict_and_update_nc_monthly(model, predict_loader, config.DEVICE, seq_length, start_year, end_year)

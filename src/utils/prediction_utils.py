import torch
import netCDF4 as nc
import numpy as np
from tqdm import tqdm

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
    num_months = 12 * 8  # 8 years
    times[:] = np.arange(0, num_months)  # Time index from 0 to 95
    
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

def get_input_data(year, month, lat_index, lon_index, data_dict, nc_file, seq_length=48):
    lat = data_dict['land_mask']['latitude'].values[lat_index]
    lon = data_dict['land_mask']['longitude'].values[lon_index]
    
    co2_sequence = data_dict['co2'].sel(latitude=lat, longitude=lon, method="nearest").values

    land_mask = data_dict['land_mask'].sel(latitude=lat, longitude=lon, method="nearest").values.item()  # Get scalar value
    lat_norm = lat / 90

    months = [(month + i) % 12 for i in range(seq_length + 1)]  # Calculate month for each timestep
    cos_months = [np.cos(2 * np.pi * month / 12) for month in months]
    sin_months = [np.sin(2 * np.pi * month / 12) for month in months]

    
    target_time_index = (year - 2015) * 12 + month
    start_time_index = target_time_index - seq_length
    if start_time_index < 0:
        raise IndexError

    x_temp = nc_file.variables['temperature'][start_time_index:start_time_index + seq_length, lat_index, lon_index]
    x_co2 = co2_sequence[start_time_index:start_time_index + seq_length]
    x_combined = list(zip(x_temp, x_co2))

    # Prepare input features for each timestep
    x_features = [np.append(np.array([temp[0], temp[1]]), [land_mask, lat_norm, cos_months[i], sin_months[i]]) for i, temp in enumerate(x_combined)]
    x_concat = np.stack(x_features)  # Stack to form a 2D array where each row is a timestep

    return x_concat
    
from tqdm import tqdm

def predict_and_update_nc_monthly(model, nc_dataset_path, device, predict_data_dict, start_year=2024, end_year=2150):
    # Open the netCDF dataset
    nc_file = nc.Dataset(nc_dataset_path, 'r+')

    model.eval()
    time_idx = 96  # Initialize time index

    for year in range(start_year, end_year + 1):
        for month in range(12):
            monthly_predictions = np.zeros((180, 360))
            # Use tqdm in the outer loop for latitude
            for lat in tqdm(range(180), desc=f"Processing Year {year}, Month {month}"):
                # Nested tqdm for longitude, you can also remove this if it's too verbose
                for lon in tqdm(range(360), desc="Longitude", leave=False):
                    input_data = get_input_data(year, month, lat, lon, predict_data_dict, nc_file)
                    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        predicted_temp = model(input_tensor).item()
                    monthly_predictions[lat, lon] = predicted_temp

            # Update the netCDF file with the predictions for this month
            nc_file.variables['temperature'][time_idx, :, :] = monthly_predictions
            nc_file.variables['time'][time_idx] = year * 12 + month  # Update time index based on year and month
            time_idx += 1
            nc_file.sync()  # Ensure data is written to disk

    nc_file.close()
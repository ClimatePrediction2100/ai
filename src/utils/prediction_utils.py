import torch
import numpy as np
import netCDF4 as nc

def initialize_netcdf_with_historical_data(source_file_path, new_file_path):
    # Load source netCDF with historical temperature data
    source_nc = nc.Dataset(source_file_path, 'r')

    # Create a new netCDF file
    new_nc = nc.Dataset(new_file_path, 'w', format='NETCDF4_CLASSIC')
    new_nc.createDimension('time', None)  # Unlimited time dimension
    new_nc.createDimension('latitude', 180)
    new_nc.createDimension('longitude', 360)
    
    times = new_nc.createVariable('time', 'i4', ('time',))
    latitudes = new_nc.createVariable('latitude', 'f4', ('latitude',))
    longitudes = new_nc.createVariable('longitude', 'f4', ('longitude',))
    temperatures = new_nc.createVariable('temperature', 'f4', ('time', 'latitude', 'longitude'))
    
    # Copy data for latitudes and longitudes
    latitudes[:] = source_nc.variables['latitude'][:]
    longitudes[:] = source_nc.variables['longitude'][:]
    
    # Load the first 9 years of temperature data (2015-2023)
    # The original file starts from 1850 and each month is sequentially stored
    start_index = 1980 # 2015
    end_index = 12 * 9  # 9 years
    historical_temperatures = source_nc.variables['temperature'][start_index:end_index, :, :]
    times[:] = np.arange(0, end_index)  # Time index from 0 to 119
    temperatures[:, :, :] = historical_temperatures
    
    latitudes.units = 'degree north'
    longitudes.units = 'degree east'
    times.units = 'months since 2015-01'
    temperatures.units = 'degree Celsius'
    
    source_nc.close()
    return new_nc

def get_input_data(year, month, lat_index, lon_index, data_dict, nc_file, seq_length=12):
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
    

def predict_and_update_nc_monthly(model, nc_file, device, predict_data_dict, start_year=2024, end_year=2150):
    model.eval()
    time_idx = 0  # Index to track time dimension in netCDF

    for year in range(start_year, end_year + 1):
        for month in range(12):  # For each month
            monthly_predictions = np.zeros((180, 360))  # Placeholder for one month's grid predictions
            
            for lat in range(180):
                for lon in range(360):
                    # Assume 'get_input_data' fetches or generates the required input for prediction
                    input_data = get_input_data(year, month, lat, lon, predict_data_dict, nc_file)
                    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        predicted_temp = model(input_tensor).item()  # Assuming the model outputs a single temperature value
                    monthly_predictions[lat, lon] = predicted_temp
            
            # Update the netCDF file with the predictions for this month
            nc_file.variables['temperature'][time_idx, :, :] = monthly_predictions
            nc_file.variables['time'][time_idx] = year * 12 + month  # Example time indexing
            time_idx += 1
            
            nc_file.sync()  # Ensure data is written to disk

    nc_file.close()  # Close the file when done
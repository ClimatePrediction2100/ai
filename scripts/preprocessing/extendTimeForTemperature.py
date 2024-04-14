import numpy as np
from netCDF4 import Dataset

# Paths for your original and new NetCDF files
original_file_path = "/Users/jinho/Desktop/climatePrediction/data/raw/globalTemperature/Land_and_Ocean_LatLong1.nc"
new_file_path = "/Users/jinho/Desktop/climatePrediction/data/processed/globalTemperature/Land_and_Ocean_LatLong1_extended.nc"

# Open the original NetCDF file for reading
nc_orig = Dataset(original_file_path, 'r')

# Create a new NetCDF file for writing
nc_new = Dataset(new_file_path, 'w', format='NETCDF4')

# Copy dimensions, and create or modify the time dimension
for name, dimension in nc_orig.dimensions.items():
    if name == 'time':
        nc_new.createDimension(name, None)  # unlimited time dimension
    else:
        nc_new.createDimension(name, len(dimension) if not dimension.isunlimited() else None)

# Copy variables and adjust time variable
for name, variable in nc_orig.variables.items():
    new_var = nc_new.createVariable(name, variable.datatype, variable.dimensions)
    new_var.setncatts({attr: variable.getncattr(attr) for attr in variable.ncattrs()})
    if name == 'time':
        # Initialize time variable with index
        initial_time_length = len(variable)
        time_index = np.arange(1, initial_time_length + 1)
        new_var[:] = time_index
    else:
        new_var[:] = variable[:]

# Extend the time variable to December 2150
# Calculate the total number of months from Jan 1850 to Dec 2150
total_months = (2150 - 1850 + 1) * 12
extended_time_index = np.arange(1, total_months + 1)
nc_new.variables['time'][:] = extended_time_index

# Extend the temperature array with NaNs
temperature_var_orig = nc_orig.variables['temperature']
new_shape = (total_months, temperature_var_orig.shape[1], temperature_var_orig.shape[2])
new_temperature_data = np.full(new_shape, np.nan, dtype='float32')
new_temperature_data[:len(temperature_var_orig), :, :] = temperature_var_orig[:]

temperature_var_new = nc_new.variables['temperature']
temperature_var_new[:, :, :] = new_temperature_data

# Close the original and new NetCDF files
nc_orig.close()
nc_new.close()

print("New file created with extended time data:", new_file_path)

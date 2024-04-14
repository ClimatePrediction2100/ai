import numpy as np
from netCDF4 import Dataset

# Paths to the CO2 files and the temperature file
file1_path = "/Users/jinho/Desktop/climatePrediction/data/raw/globalGhgEmissions/CO2_1deg_month_1850-2013.nc"
file2_path = "/Users/jinho/Desktop/climatePrediction/data/raw/globalGhgEmissions/CO2_SSP119_2015_2150.nc"
temp_file_path = "/Users/jinho/Desktop/climatePrediction/data/processed/Land_and_Ocean_LatLong1_extended_SSP119.nc"

# Open the CO2 files
nc_co2_1 = Dataset(file1_path, 'r')
nc_co2_2 = Dataset(file2_path, 'r')

# Read CO2 data from the first file
co2_data_1 = nc_co2_1.variables['value'][:]

# Read CO2 data from the second file
co2_data_2 = nc_co2_2.variables['CO2'][:]

# Assuming monthly data, calculate how many months are missing in 2014
months_missing = 12

# Initialize a placeholder for missing data with NaNs
missing_data = np.full((months_missing, co2_data_1.shape[1], co2_data_1.shape[2]), np.nan, dtype='float32')

# Combine the datasets
combined_co2_data = np.concatenate((co2_data_1, missing_data, co2_data_2), axis=0)

# Open the temperature file to append the CO2 data
nc_temp = Dataset(temp_file_path, 'a')  # Open in append mode

# Print available dimensions in the temperature file
print("Available dimensions in the temperature file:", nc_temp.dimensions.keys())

# Ensure the necessary dimensions exist, else create them
if 'time' not in nc_temp.dimensions:
    nc_temp.createDimension('time', None)
if 'latitude' not in nc_temp.dimensions:
    nc_temp.createDimension('latitude', co2_data_1.shape[1])
if 'longitude' not in nc_temp.dimensions:
    nc_temp.createDimension('longitude', co2_data_1.shape[2])

# Check if CO2 variable already exists, if not, create it
if 'CO2' not in nc_temp.variables:
    co2_var = nc_temp.createVariable('CO2', 'f4', ('time', 'latitude', 'longitude'))
    co2_var.units = 'ppmv'
    co2_var.long_name = "mole fraction of carbon dioxide in air"
    co2_var.missing_value = 1.e+20
    co2_var.coordinates = "time latitude longitude"
else:
    co2_var = nc_temp.variables['CO2']

# Assign the combined CO2 data to the new variable
co2_var[:] = combined_co2_data

# Close all files
nc_co2_1.close()
nc_co2_2.close()
nc_temp.close()

print("CO2 data successfully combined and appended to the temperature file.")

import netCDF4 as nc

# Open the NetCDF file
file_path = "/Users/jinho/Desktop/climatePrediction/data/raw/globalGhgEmissions/CO2_1deg_month_1850-2013.nc"
ds = nc.Dataset(file_path, 'r+')

# Rename variables
ds.renameVariable('Latitude', 'latitude')
ds.renameVariable('Longitude', 'longitude')

# Update attributes for 'latitude'
ds.variables['latitude'].setncattr('Long_name', 'Latitude')
ds.variables['latitude'].setncattr('units', 'degrees_north')

# Update attributes for 'longitude'
ds.variables['longitude'].setncattr('Long_name', 'Longitude')
ds.variables['longitude'].setncattr('units', 'degrees_east')

# Close the dataset to write changes and free resources
ds.close()

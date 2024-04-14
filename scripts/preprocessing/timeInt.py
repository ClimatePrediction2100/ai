import xarray as xr
import numpy as np
import os

def replace_time_with_int(file_path):
    """
    Replaces the 'time' variable in a NetCDF file from float to int.

    Parameters:
        file_path (str): The path to the NetCDF file.
    """
    # Check if the file is writable
    if not os.access(file_path, os.W_OK):
        print("Error: File is not writable. Check permissions.")
        return

    # Try opening and modifying the NetCDF file
    try:
        with xr.open_dataset(file_path, mode='a') as ds:
            # Convert the time data from float to int
            time_data = ds['time'].values
            time_int = np.floor(time_data).astype(int)
            new_time = xr.DataArray(time_int, dims=["time"], name="time")
            new_time.attrs['units'] = 'months since Jan 1850'

            # Replace the 'time' coordinate in the dataset
            ds = ds.assign_coords(time=new_time)

            # Save the changes
            ds.to_netcdf(file_path, mode='w', format='NETCDF4')
            print("Time variable successfully converted to integer.")
    except PermissionError as e:
        print(f"PermissionError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Path to your NetCDF file
file_path = '/Users/jinho/Desktop/climatePrediction/data/processed/globalTemperature/Land_and_Ocean_LatLong1_extended.nc'

# Run the function to replace the 'time' variable with an integer version
replace_time_with_int(file_path)

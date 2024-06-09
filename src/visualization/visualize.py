import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import os
import pandas as pd
import sys
sys.path.append('../..')
import config

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Define the data path and generate file list
data_path = os.path.join(config.ROOT_DIR, 'results', 'netCDF')
ssp_list = ['SSP119', 'SSP126', 'SSP245', 'SSP370', 'SSP434', 'SSP460', 'SSP534', 'SSP585']
postfix = '_predictions.nc'
file_list = [os.path.join(data_path, ssp + postfix) for ssp in ssp_list]

# Start date set to 2015
start_date = '2015-01'
num_months = 0  # This will be calculated based on the length of the time dimension in the dataset

# Calculate the index to start from January 2024
start_year = 2015
start_month = 1
target_year = 2024
target_month = 1
start_index = (target_year - start_year) * 12 + (target_month - start_month)

vmin = 0
vmax = 3

frames = []
output_dir = os.path.join(config.ROOT_DIR, 'results', 'frames_3')
os.makedirs(output_dir, exist_ok=True)

for file in file_list:
    ds = nc.Dataset(file)
    variable = ds.variables['temperature'][:]  # Assume 'temperature' is your variable
    time_dim = ds.variables['time']  # Get the time variable
    num_months = len(time_dim)  # Update number of months based on time variable length
    
    # Generate the date range for each time step starting from 2015
    dates = pd.date_range(start=start_date, periods=num_months, freq='M')
    
    for month in range(start_index, num_months):
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(40, 20), subplot_kw={'projection': ccrs.PlateCarree()})
        axes = axes.flatten()
        
        for i, file in enumerate(file_list):
            ds = nc.Dataset(file)
            variable = ds.variables['temperature'][:]
            ax = axes[i]
            ax.coastlines()
            im = ax.imshow(variable[month, :, :], cmap='viridis', transform=ccrs.PlateCarree(), extent=[-180, 180, -90, 90], vmin=vmin, vmax=vmax, origin='lower')
            ax.set_title(f"{ssp_list[i]}", fontsize=30)  # Show SSP scenario in title
            
            # Annotate the date at the bottom right corner
            date_str = dates[month].strftime('%Y-%m')
            ax.annotate(date_str, xy=(0.95, 0.05), xycoords='axes fraction', fontsize=20, ha='right', va='bottom', color='white')

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.subplots_adjust(right=0.82)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Temperature', fontsize=20)
        cbar.ax.tick_params(labelsize=20)

        frame_filename = os.path.join(output_dir, f'frame_{month:04d}.png')
        plt.savefig(frame_filename)
        frames.append(frame_filename)
        plt.close(fig)
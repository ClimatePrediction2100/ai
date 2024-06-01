.PHONY: expr setup

expr:
	@bash expr.sh

simulate:
	@bash predict.sh

setup:
	@pip install -r requirements.txt
	@mkdir -p results/models
	@mkdir -p results/globalTemperature
	@mkdir -p data/raw/globalTemperature
	@mkdir -p data/raw/globalGhgEmissions

# https://berkeleyearth.org/data/
# https://zenodo.org/records/5021361

download_data:
	wget -O data/raw/globalTemperature/Land_and_Ocean_LatLong1.nc https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Gridded/Land_and_Ocean_LatLong1.nc
	wget -O data/raw/globalGhgEmissions/CO2_1deg_month_1850-2013.nc https://zenodo.org/records/5021361/files/CO2_1deg_month_1850-2013.nc?download=1
	wget -O data/raw/globalGhgEmissions/CO2_SSP119_2015_2150.nc https://zenodo.org/records/5021361/files/CO2_SSP119_2015_2150.nc?download=1
	wget -O data/raw/globalGhgEmissions/CO2_SSP126_2015_2150.nc https://zenodo.org/records/5021361/files/CO2_SSP126_2015_2150.nc?download=1
	wget -O data/raw/globalGhgEmissions/CO2_SSP245_2015_2150.nc https://zenodo.org/records/5021361/files/CO2_SSP245_2015_2150.nc?download=1
	wget -O data/raw/globalGhgEmissions/CO2_SSP370_2015_2150.nc https://zenodo.org/records/5021361/files/CO2_SSP370_2015_2150.nc?download=1
	wget -O data/raw/globalGhgEmissions/CO2_SSP434_2015_2150.nc https://zenodo.org/records/5021361/files/CO2_SSP434_2015_2150.nc?download=1
	wget -O data/raw/globalGhgEmissions/CO2_SSP460_2015_2150.nc https://zenodo.org/records/5021361/files/CO2_SSP460_2015_2150.nc?download=1
	wget -O data/raw/globalGhgEmissions/CO2_SSP534_2015_2150.nc https://zenodo.org/records/5021361/files/CO2_SSP534_2015_2150.nc?download=1
	wget -O data/raw/globalGhgEmissions/CO2_SSP585_2015_2150.nc https://zenodo.org/records/5021361/files/CO2_SSP585_2015_2150.nc?download=1

download_weights:
	# wget -O results/weights.tar.gz https://github.com/ClimatePrediction2100/data/releases/download/torchmodel/model_weights.tar.gz
	tar -zxvf results/weights.tar.gz -C results

download_results:
	wget -O results/netcdf.tar.gz https://github.com/ClimatePrediction2100/data/releases/download/netCDF4/results_by_ssp.tar.gz
	tar -zxvf results/netcdf.tar.gz -C results
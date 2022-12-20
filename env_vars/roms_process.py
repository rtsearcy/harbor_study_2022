#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 14:02:01 2022

@author: rtsearcy

Convert ROMS salinity and temp data from netcdf to csv time series

SOURCE:
    https://thredds.cencoos.org/thredds/ncss/CENCOOS_CA_ROMS_DAS.nc/dataset.html

"""

import pandas as pd
import numpy as np
import os
import netCDF4 as nc

folder = '/Volumes/GoogleDrive/My Drive/high_frequency_wq/harbor_study_2022/data/hindcast'
file = 'CENCOOS_CA_ROMS_DAS.nc'

data = nc.Dataset(os.path.join(folder, file))

print(data.dimensions)
print('\nVariables:')
print(data.variables.keys())

# date time
datum = data['time'].units.replace('hours since ', '')
dt = pd.to_datetime(datum) + pd.to_timedelta(data['time'][:], 'H')
dt = dt - pd.to_timedelta(8, 'H')   # UTC to PST
dt.name = 'dt'

# Lat/Lon
lat = data['lat'][:].data
lon = data['lon'][:].data

# Salinity
salt = pd.Series(data['salt'][:].data[:,0,0,0])  # All times, surface depth, 1st position
salt.name = 'sal'

salt.loc[salt == -9999] = np.nan

# # temp
temp = pd.Series(data['temp'][:].data[:,0,0,0])  # All times, surface depth, 1st position
temp.name = 'temp'

temp.loc[temp == -9999] = np.nan

df = pd.concat([salt, temp], axis=1)
df.index = dt

df.to_csv(os.path.join(folder, 'ROMS_salinity_temp.csv'))

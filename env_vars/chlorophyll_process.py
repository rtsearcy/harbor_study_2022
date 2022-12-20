#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 14:02:01 2022

@author: rtsearcy

Convert MODIS chl data from netcdf to csv time series

SOURCE:
    https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMBchla1day.html?chlorophyll%5B(2022-10-25T12:00:00Z)%5D%5B(0.0)%5D%5B(37.45):(37.525000000000006)%5D%5B(237.47500000000002):(237.55)%5D&.draw=surface&.vars=longitude%7Clatitude%7Cchlorophyll&.colorBar=%7C%7C%7C%7C%7C&.bgColor=0xffccccff
"""

import pandas as pd
import numpy as np
import os
import netCDF4 as nc

folder = '/Volumes/GoogleDrive/My Drive/high_frequency_wq/harbor_study_2022/data/hindcast'
file = 'MODIS_chl.nc'

data = nc.Dataset(os.path.join(folder, file))

print(data.dimensions)
print('\nVariables:')
print(data.variables.keys())

# date time
datum = data['time'].units.replace('seconds since ', '')
dt = pd.to_datetime(datum) + pd.to_timedelta(data['time'][:], 's')
dt = dt.tz_localize(None)
dt = dt - pd.to_timedelta(8, 'H')   # UTC to PST
dt.name = 'dt'

# Lat/Lon
lat = data['latitude'][:].data
lon = data['longitude'][:].data

# # Chlorophyll
''' Take mean of all points in range'''
chl = data['chlorophyll'][:].data
chl = chl.reshape((chl.shape[0],chl.shape[1]*chl.shape[2]*chl.shape[3]))
chl = pd.DataFrame(chl)
chl = chl.replace(-9999999, np.nan)
chl = chl.mean(axis=1)

df = pd.Series(chl)  # All times, surface depth, 1st position
df.name = 'chl'
df.index = dt

df = df.interpolate(limit=3, limit_area = 'inside') # fill in some NaNs up to three days

df.to_csv(os.path.join(folder, 'MODIS_chlorophyll.csv'))

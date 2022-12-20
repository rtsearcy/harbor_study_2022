#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:31:40 2022

@author: rtsearcy

- Combine hindcast datasets for use in model testing

"""

import pandas as pd
import numpy as np
import os

def lag_vars(df, var_list, n_lags, interval=None, interpolate=False):
    ''' Creates lag variables. If 'interval', shift by that number of minutes'''
    
    for v in var_list:
        for i in range(1, n_lags+1):
            if interval == None:
                df[v+str(i)] = df[v].shift(i)  # variable i time steps previous
            else:
                # Shift data by interval
                df[v+'_' + str(i*interval)+'min'] = df[v].shift(i, freq=str(interval)+'min')
    
    if interpolate:
        df = df.interpolate(axis=1)  # fill in data for lower res variables
    else:
        df = df.dropna(axis=1, how='all') # drops variable with unaligned timestep
    
    return df

def interp(df, var_list, limit):
    # interpolate missing data in variables with few missing
    for v in var_list:
        df.loc[:,v] = df[v].interpolate(limit=limit, limit_area='inside')
        if v == 'wdir':
            df.loc[df['wspd']==0, v] = np.nan  # no wdir if wspd = 0
            df.loc[:,v] = df.loc[:,v].round(0)
    
    return df

### Load data
folder = '/Volumes/GoogleDrive/My Drive/high_frequency_wq/harbor_study_2022/data/hindcast'
files = os.listdir(folder)

sd = '2017-01-01'
ed = '2022-08-01'

### FIB

# Constants
FIB = {'TC':10000,
       'FC': 400,
       'ENT': 104
       }
LOQ = 10
replace = 1
beach_angle = 140

fib = pd.read_csv(os.path.join(folder,[f for f in files if ('FIB_PP7' in f)][0]))
fib['dt'] = pd.to_datetime(fib['date']) + pd.to_timedelta(fib['sample_time'])
fib.set_index('dt', inplace=True)
fib.drop(['date','sample_time'], axis=1, inplace=True)
fib = fib[sd:ed]
fib = fib[~fib.index.duplicated()]

for f in FIB:
    # All samples with qualifier < or <= to 10 (or less), replace
    idx = fib[(fib[f+'_qual'].isin(['<','<='])) & (fib[f] <= LOQ)].index
    fib.loc[idx,f] = replace

fib = fib[FIB]  # drop qualifying info

## Quant, Exceedaces, Antecedent, FIB 1-3 days previous
for f in FIB:
    fib[f + '_ant'] = fib[f].dropna().shift(1)  # antecedent sample, skipping any missed samples in dataset
    fib[f + '_BLOQ'] = (fib[f] == replace).astype(int)  # = or exceeds LOQ? (binary)
    fib.loc[fib[f].isna(),f + 'BLOQ'] = np.nan
    
    fib[f + '_ant_BLOQ'] = (fib[f + '_ant'] == replace).astype(int)
    fib.loc[fib[f+'_ant'].isna(),f + '_ant_BLOQ'] = np.nan
    # previous day quanitfied? (binary) 
    
    fib[f + '_exc'] = (fib[f] > FIB[f]).astype(int)  # exceeds threshold? (binary)
    fib.loc[fib[f].isna(),f + '_exc'] = np.nan
    
    fib[f + '_ant_exc'] = (fib[f + '_ant'] > FIB[f]).astype(int)
    fib.loc[fib[f+'_ant'].isna(),f + '_ant_exc'] = np.nan
    # previous day exceeds threshold? (binary)

    # log10 transform
    fib['log' + f] = round(np.log10(fib[f]), 2)
    fib['log' + f + '_ant'] = round(np.log10(fib[f + '_ant']), 2)


### Tide
tide = pd.read_csv(os.path.join(folder,[f for f in files if ('tide' in f)][0]),
                   index_col=['dt'], parse_dates=['dt'])
tide = tide['tide'].to_frame()
tide['tide_high'] = (tide.tide > 1).astype(int)

tide = lag_vars(tide, var_list=list(tide.columns), n_lags=6, interval=30)
tide = tide.resample('30min').nearest()


### Met
met = pd.read_csv(os.path.join(folder,[f for f in files if ('HMB_met' in f)][0]),
                  index_col=['dt'], parse_dates=['dt'])

met = interp(met, var_list=list(met.columns), limit=2)
met['awind'] = met['wspd'] * round(np.sin(((met['wdir'] - beach_angle) / 180) * np.pi), 1)
met['awind'].fillna(0, inplace=True) # for 0 wind speed values
met['owind'] = met['wspd'] * round(np.cos(((met['wdir'] - beach_angle) / 180) * np.pi), 1)
met['owind'].fillna(0, inplace=True)

met = lag_vars(met, var_list=list(met.columns), n_lags=6, interval=30)


### Rad
rad = pd.read_csv(os.path.join(folder,[f for f in files if ('NREL_hourly' in f)][0]),
                  index_col=['dt'], parse_dates=['dt'])
rad = rad['rad'].to_frame()  # only keep DNI variable
rad = lag_vars(rad, var_list=['rad'], n_lags=6, interval=30, interpolate=True)


### Water Temp (from Buoy)
wtemp = pd.read_csv(os.path.join(folder,[f for f in files if ('waves_raw' in f)][0]),
                  index_col=['dt'], parse_dates=['dt'])
wtemp = wtemp['wtemp_b']
wtemp.name = 'wtemp'
wtemp = wtemp.to_frame()
wtemp = wtemp.resample('30min').nearest()
wtemp = interp(wtemp, var_list=list(wtemp.columns), limit=24)
wtemp = lag_vars(wtemp, var_list=['wtemp'], n_lags=6, interval=30)

### Salinity
sal = pd.read_csv(os.path.join(folder,[f for f in files if ('salinity' in f)][0]),
                  index_col=['dt'], parse_dates=['dt'])
sal = sal['sal']
sal = sal.to_frame()
sal = sal.resample('30min').interpolate()  # 6h to 30m interval
sal = lag_vars(sal, var_list=['sal'], n_lags=6, interval=30, interpolate=True)


### Chl
chl = pd.read_csv(os.path.join(folder,[f for f in files if ('chlorophyll' in f)][0]),
                  index_col=['dt'], parse_dates=['dt'])
# daily data
chl = chl.resample('30min').nearest()
chl = lag_vars(chl, var_list=['chl'], n_lags=6, interval=30, interpolate=True)


### Combine EV Data
evs = pd.concat([tide, met, rad, wtemp, sal, chl], axis=1)
## TODO: RGRESS WQ PARAMETERS

### Combine with FIB variables + Save
df = pd.merge_asof(fib, evs, left_index=True, right_index=True)

# Time vars
df['hour'] = df.index.hour
df['daytime'] = (df.hour.isin([8,9,10,11,12,13,14,15,16,17,18,19])).astype(int)
df['hours_from_noon'] = (12 - df.hour).abs() # hours from solar noon

df.to_csv(os.path.join(folder, 'hindcast_dataset.csv'))

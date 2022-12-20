#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:31:40 2022

@author: rtsearcy

- Combine historical and THFS datasets into  modeling dataset
- Add low-frequency variables:
    Precipitation vars, tide regime, month, day of year, water temperature,
    upwelling?

"""

import pandas as pd
import numpy as np
import os
import datetime
import ephem

def fib_vars(df, FIB, replace):
    '''Adds exceedence, BLOQ, log-transform variables'''
    for f in FIB:
        # Antecedent Samples
        df[f + '_ant'] = df[f].dropna().shift(1)  # antecedent sample, skipping any missed samples in dataset
        df[f + '_BLOQ'] = (df[f] == replace).astype(int)  # = or exceeds LOQ? (binary)
        df.loc[df[f].isna(),f + '_BLOQ'] = np.nan
        
        df[f + '_ant_BLOQ'] = (df[f + '_ant'] == replace).astype(int)
        df.loc[df[f+'_ant'].isna(),f + '_ant_BLOQ'] = np.nan
        # previous day quanitfied? (binary) 
        
        # Exceedances
        df[f + '_exc'] = (df[f] > FIB[f]).astype(int)  # exceeds threshold? (binary)
        df.loc[df[f].isna(),f + '_exc'] = np.nan
        
        df[f + '_ant_exc'] = (df[f + '_ant'] > FIB[f]).astype(int)
        df.loc[df[f+'_ant'].isna(),f + '_ant_exc'] = np.nan
        # previous day exceeds threshold? (binary)
    
        # log10 transform
        df['log' + f] = round(np.log10(df[f]), 2)
        df['log' + f + '_ant'] = round(np.log10(df[f + '_ant']), 2)
    return df

def met_vars(df, rain=True, lag=True):
    '''Low Freq Meteorological variables'''
    
    if rain:
        df_rain = df['rain'].copy()
        
    df = df.drop(['rain'],axis=1, errors='ignore')
    
    df = df.resample('1D').mean().round(1)  # drops cloud variable (OK)
    
    if lag:
        df = lag_vars(df, list(df.columns), n_lags=3, interval=None)
    
    if rain:
        df_rain = df_rain.resample('1D').sum()   # daily precip total
        df_rain = df_rain.to_frame()
        
        for i in range(1, 8):  #lograin1-lograin7
            df_rain['lograin' + str(i)] = round(np.log10(1 + df_rain['rain'].shift(i, freq='D')), 1)
            
        total_list = list(range(2, 8)) + [14]
        for j in total_list:  # rain2T-rain7T, 14T, 30T
            df_rain['rain' + str(j) + 'T'] = 0.0
            for k in range(j, 0, -1):
                df_rain['rain' + str(j) + 'T'] += df_rain['rain'].shift(k, freq='D')
            df_rain['lograin' + str(j) + 'T'] = round(np.log10(1+df_rain['rain' + str(j) + 'T']), 1)
            df_rain.drop(['rain' + str(j) + 'T'], axis=1, inplace=True)
            
        # Wet Days
        # Not including same day rain because not available for daily runs, and samples are typically taken early in the day
        df_rain['wet'] = (df_rain[['lograin3T']] > np.log10(1 + 2.54)).any(axis=1).astype(int)
        
        df = pd.concat([df, df_rain], axis=1)
    
    return df

def lag_vars(df, var_list, n_lags, interval=None, interpolate=False):
    ''' Creates HIGH FREQUENCY lag variables. If 'interval', shift by that number of minutes'''
    
    for v in var_list:
        for i in range(1, n_lags+1):
            if interval == None:
                if v[-1].isalpha():
                    var_name = v+str(i)
                else:
                    var_name = v + '_' + str(i)
                df[var_name] = df[v].shift(i)  # variable i time steps previous
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

def tide_spring(df):
    '''Creates lunar tide variables from the tide datetime index'''
    
    ## Create Date Range
    sd = df.index[0]
    ed = df.index[-1]
    idx = pd.date_range(sd, ed, freq='1D')
    temp = pd.DataFrame(index=idx)
    temp['days_since_full'] = np.nan
    
    for d in range(0,len(idx)):
        date=ephem.Date(datetime.date(idx[d].year,idx[d].month,idx[d].day))
        prev_full = ephem.previous_full_moon(date)
        temp.loc[idx[d],'days_since_full'] = int(np.floor(date - prev_full))
    
    temp['tide_spring'] = 0  ## 0 = neap tide, 1 = spring tide
    temp.loc[temp['days_since_full'].isin([0,1,2,3,12,13,14,15,15,17,18,26,27,28]),'tide_spring'] = 1
    
    temp.index.rename(df.index.name, inplace=True)
    
    return temp['tide_spring']

#%% Setup FIB and Enviro Vars
folder = '/Volumes/GoogleDrive/My Drive/high_frequency_wq/harbor_study_2022/data/hybrid'
files = os.listdir(folder)

sy = 2017
ey= 2022

# Constants
FIB = {'TC':10000,
       'FC': 400,
       'ENT': 104
       }
LOQ = 10
replace = 1
beach_angle = 140

### HF Sampling Event
hf = pd.read_csv(os.path.join(folder,[f for f in files if ('hf_sampling' in f)][0]),
                 parse_dates=['dt'],
                 index_col=['dt'])

hf = hf[(hf.site == 'PP7') & (hf['shift'] != 'sprint')] # PP7 data
#hf = hf.drop(['shift','site','notes'], axis=1) # drop irrelevant columns + MET/TIDE
hf['logchl'] = np.log10(1 + hf['chl'])
hf = hf[list(FIB.keys()) + ['wtemp','sal','logchl']]  # drop turb because don't have proxy
# will add variables back in next
hf = fib_vars(hf, FIB, replace) # add FIB variables
hf = lag_vars(hf, var_list = ['wtemp','sal','logchl'], n_lags=6, interval=30, interpolate=True)
hf['HF'] = True  # HF flag

### Historical FIB
fib = pd.read_csv(os.path.join(folder,[f for f in files if ('FIB_PP7' in f)][0]))
fib['dt'] = pd.to_datetime(fib['date']) + pd.to_timedelta(fib['sample_time'])
fib.set_index('dt', inplace=True)
fib.drop(['date','sample_time'], axis=1, inplace=True)
fib = fib.drop([i for i in fib.index if i in hf.index])
fib = fib['2016':]
fib = fib[~fib.index.duplicated()]

for f in FIB:
    # All samples with qualifier < or <= to 10 (or less), replace
    idx = fib[(fib[f+'_qual'].isin(['<','<='])) & (fib[f] <= LOQ)].index
    fib.loc[idx,f] = replace

fib = fib[FIB]  # drop qualifying info
fib = fib_vars(fib, FIB, replace) # add FIB variables

fib['HF'] = False

### Combine HF and Historical
combo = fib.append(hf)
combo = combo.sort_index(ascending=True)
combo = combo[str(sy):str(ey)]

### Time vars
combo['hour'] = combo.index.hour
combo['daytime'] = (combo.hour.isin([8,9,10,11,12,13,14,15,16,17,18,19])).astype(int)
combo['hours_from_noon'] = (12 - combo.hour).abs() # hours from solar noon
combo['month'] = combo.index.month
combo['dayofyear'] = combo.index.dayofyear

''' Results in a base dataframe which EV data will be added to'''

#%% ENVIRO VARS
### Tide
tide = pd.read_csv(os.path.join(folder,[f for f in files if ('tide' in f)][0]),
                   index_col=['dt'], parse_dates=['dt'])
tide = tide['tide'].to_frame()
tide['tide_high'] = (tide.tide > 1).astype(int)

tide = lag_vars(tide, var_list=list(tide.columns), n_lags=6, interval=30)  # HF variables

tideLF = pd.DataFrame()  # Low Freq Tide Variables
''' Index is a date'''
tideLF['tide_max'] = tide['tide'].resample('1D').max()
tideLF['tide_min'] = tide['tide'].resample('1D').min()
tideLF['tide_range'] = tideLF['tide_max'] - tideLF['tide_min']
tideLF['tide_spring'] = tide_spring(tideLF)

tideLF = lag_vars(tideLF, tideLF.columns, n_lags=3)


### Met / Rad
met = pd.read_csv(os.path.join(folder,[f for f in files if ('HMB_met' in f)][0]),
                  index_col=['dt'], parse_dates=['dt'])
met = interp(met, var_list=list(met.columns), limit=2)

rad = pd.read_csv(os.path.join(folder,[f for f in files if ('NREL_hourly' in f)][0]),
                  index_col=['dt'], parse_dates=['dt'])
rad = rad['rad'].to_frame()  # only keep DNI variable
rad = rad.resample('30min').interpolate()  # upsample to 30m
rad = rad.sort_index(ascending=True)
met = pd.concat([met, rad], axis=1)

met['awind'] = met['wspd'] * round(np.sin(((met['wdir'] - beach_angle) / 180) * np.pi), 1)
met['awind'].fillna(0, inplace=True) # for 0 wind speed values
met['owind'] = met['wspd'] * round(np.cos(((met['wdir'] - beach_angle) / 180) * np.pi), 1)
met['owind'].fillna(0, inplace=True)
met = met.drop(['wspd','wdir'], axis=1)

metLF = met_vars(met, rain=True, lag=True)  # low frequency met variables
metLF = metLF.drop([c for c in metLF.columns if c in met.columns], axis=1)  # drop same day means vars

met = pd.concat([met, pd.get_dummies(met.cloud, prefix = 'cloud')], axis=1).drop(['cloud'],axis=1)
met = lag_vars(met, var_list=list(met.columns), n_lags=6, interval=30)


### Salinity (and Wtemp) from ROMS Model
roms = pd.read_csv(os.path.join(folder,[f for f in files if ('salinity' in f)][0]),
                  index_col=['dt'], parse_dates=['dt'])
roms = roms.rename(columns={'temp':'wtemp'})
roms = roms.resample('30min').interpolate().round(1)  # upsample to 30min

romsLF = roms.resample('1D').mean()
romsLF = lag_vars(romsLF, romsLF.columns, n_lags=3)
romsLF = romsLF.drop([c for c in romsLF.columns if c in roms.columns], axis=1)  # drop same day means vars

roms = lag_vars(roms, var_list=['sal','wtemp'], n_lags=6, interval=30, interpolate=True)


### Chl
chlLF = pd.read_csv(os.path.join(folder,[f for f in files if ('chlorophyll' in f)][0]),
                  index_col=['dt'], parse_dates=['dt']) # daily data
chlLF['logchl'] = np.log10(chlLF['chl']+1)
chlLF = chlLF.drop(['chl'], axis=1)

chl = chlLF.resample('30min').interpolate().round(3)
chl = lag_vars(chl, var_list=['logchl'], n_lags=6, interval=30, interpolate=True)

chlLF = chlLF.resample('1D').nearest().round(3)
chlLF = lag_vars(chlLF, chlLF.columns, n_lags=3)
chlLF = chlLF.drop(['logchl'], axis=1)

## Combine HF WQ params
wq = pd.concat([roms, chl], axis=1)

#%%Combine  Data

### Add HF WQ to historical data
idx = combo.query('HF == 0').index
combo.loc[idx, wq.columns] = wq.reindex(idx, method='nearest')

### add HF met and tide vars
combo = pd.merge_asof(combo, tide, left_index=True, right_index=True)
combo = pd.merge_asof(combo, met, left_index=True, right_index=True)

### Add LF variables
evLF = pd.concat([tideLF, metLF, romsLF, chlLF], axis=1)
combo = pd.merge_asof(combo, evLF, left_index=True, right_index=True)

print(combo.isna().sum().sort_values().tail(50))
combo.to_csv(os.path.join(folder, 'hybrid_dataset.csv'))

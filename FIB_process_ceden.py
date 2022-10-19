#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 11:46:58 2022

@author: rtsearcy

Processes raw FIB data downloaded from CEDEN into beach specific time series
(CEDEN data needs to be in CSV format, beaches labelled in a new 'beach' column)

Output: time series of FIB concentrations + qualifiers 
"""

import pandas as pd
import numpy as np
import os

folder = '/Volumes/GoogleDrive/My Drive/high_frequency_wq/harbor_study_2022/data/hindcast'

### Iterate through raw files
#for f in [c for c in os.listdir(os.path.join(folder,'raw')) if c.endswith('.csv')]:
f = 'FIB_pillar_point_ceden.csv'

### Load Raw File
raw_file = os.path.join(folder, f)
df_raw = pd.read_csv(raw_file)
#df_raw.groupby(['StationName']).count()

raw_cols = ['beach','county', 'SampleDate','CollectionTime','TargetLatitude','TargetLongitude',
            'Analyte','Result','ResultQualCode']
df_raw = df_raw[raw_cols]  # remove uncess. columns
df_raw.columns = ['beach','county','date','sample_time','lat','lon','fib','conc','qual']

### Extract Samples By Time Point and FIB

## Convert Analyte to FIB codes
df_raw.loc[df_raw.fib.str.contains('Total'),'fib'] = 'TC'
#df_raw.loc[df_raw.fib.str.contains('Fecal'),'fib'] = 'FC'
df_raw.loc[df_raw.fib.str.contains('Fecal'),'fib'] = 'EC' ## treat Fecal Coliform = E. Coli
df_raw.loc[df_raw.fib.str.startswith('E.'),'fib'] = 'EC'
df_raw.loc[df_raw.fib.str.contains('Ent'),'fib'] = 'ENT'

## Pivot 
df_raw = df_raw[~df_raw.duplicated(subset=['beach','county','date','sample_time','fib'],
                                           keep='first')]
df_raw = df_raw.pivot(index=['beach','county','date','sample_time','lat','lon'],
                      columns=['fib'], 
                      values=['conc','qual'])

## Rename
df_raw.columns = [c[1]+'_'+c[0] for c in df_raw.columns]
df_raw.columns = [c.replace('_conc','') for c in df_raw.columns]
df_raw = df_raw[list(df_raw.columns.sort_values())]
df_raw.reset_index(inplace=True)

## Convert Datetime
df_raw['date'] = pd.to_datetime(df_raw.date)
df_raw.sort_values(['beach','date','sample_time'], inplace=True)

## Save Countywide data
county = df_raw.county[0]
df_raw.to_csv(os.path.join(folder, county.replace(' ','_') +'_county_FIB.csv'), index=False)

#%% Iterate Through Beaches
for b in df_raw.beach.unique():
    print(b)
    df = df_raw[df_raw.beach==b].copy()
    # df['dt'] = df.date + pd.to_timedelta(df.time)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    #df.drop(['beach','county','lat','lon','date','time'], axis=1, inplace=True)
    df.drop(['beach','county','lat','lon'], axis=1, inplace=True)

### Save Indiv Beach Files
    print(df.index[0])
    print(df.index[-1])
    print(str(len(df)) + ' samples\n')
    df.to_csv(os.path.join(folder,b.replace(' ','_') +'_FIB.csv'))
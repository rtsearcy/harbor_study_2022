#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 12:57:17 2022

@author: rtsearcy
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

## Load Data
file = '/Volumes/GoogleDrive/My Drive/high_frequency_wq/harbor_study_2022/data/all_data.csv'

df = pd.read_csv(file)   # read data using pandas (pd)
df = df.dropna(how='all')

df['dt'] = pd.to_datetime(df['dt'])  # convert to DateTime object
df.set_index('dt', inplace=True)  # Set dt as index

df = df[(df.site == 'PP7') & (df['shift'] != 'Sprint')]



FIB = {'TC':10000,
       'FC': 400,
       'ENT': 104
       }


# df[FIB].plot()  # plot time series
# plt.yscale('log') # log10 scale


# log transform
for f in FIB:
    df['log'+f] = np.log10(df[f])


# Number of exceedances / BLOQ (FLAGS)
for f in FIB:
    df[f + '_exc'] = (df[f] > FIB[f]).astype(int)
    df[f + '_BLOQ'] = (df[f] < 10).astype(int)
    
 
    
# Summarize
print(df[FIB].describe().round(2))
df.sum()[['ENT_BLOQ','ENT_exc']]  # total exc/BLOQ


# Variation
CV =  df[FIB].std() / df[FIB].mean() # coefficienrt of variations (stdev / mean)
# measurement of dispersion (normalized)

# Differences in subsequent samples
df.ENT.diff() # difference in consecutive samples , equivalent to df.ENT - df.ENT.shift()
df.ENT.diff().abs() / df.ENT.mean() # difference normalized by experimental mean
(df.ENT.diff().abs() / df.ENT.mean()).describe().round(3)

## FOR JAKE: % SAMPLES THAT CHANGE BLOQ/EXC STATUS
# hint: use df[f+'_BLOQ'].shift()
df['ENT_BLOQ'].diff().abs().sum() / (len(df) - 1)
df['ENT_exc'].diff().abs().sum() / (len(df) - 1)

# Correlation
C = df.corr(method = 'spearman')
C[FIB].loc[FIB]
C['ENT'].sort_values()

df['tide_stage'] = (df.tide > 1).astype(int)
df.groupby('tide_stage').describe()[['FC','ENT']].round(1)
df.groupby('tide_stage').sum()[['ENT_BLOQ','ENT_exc']]

df['daytime'] = (df.hour.isin([8,9,10,11,12,13,14,15,16,17,18,19])).astype(int)
df.groupby('daytime').describe()[['FC','ENT']].round(1)
df.groupby('daytime').sum()[['ENT_BLOQ','ENT_exc']]

df.groupby('cloud').sum()[['ENT_BLOQ','ENT_exc']]

# Downsampling
df.resample('1D').mean() # aggrefate by day, take mean
df_ds = df.resample('1H').nearest() # take nearest sample to every 1 hour


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_test.py

Playing with preliminary data

Created on Tue Aug 16 10:16:29 2022

@author: rtsearcy
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


folder = '/Volumes/GoogleDrive/My Drive/high_frequency_wq/harbor_study_2022/data/'
FIB = ['TC','E. Coli','ENT']

df = pd.read_csv(os.path.join(folder, 'harbor_study_prelim.csv'))

## Datetime
df['hour'] = pd.to_datetime(df['Time Collected']).dt.hour
df['minute'] = pd.to_datetime(df['Time Collected']).dt.minute
df['dt'] = pd.to_datetime(df['Date']) + pd.to_timedelta(df['hour'], 'H') + pd.to_timedelta(df['minute'], 'm')

## MPN
mpn_table = pd.read_csv(os.path.join(folder, 'mpn_table.csv'), index_col='large\small')
for i in ['0','48']:
    mpn_table[i] = pd.to_numeric([x.replace('\xa0','') for x in mpn_table[i]], errors='ignore')

for f in FIB:   
    big_small = list(zip(list(df[f + ' Big']), list(df[f + ' Small'])))
    data2 = pd.Series([mpn_table.loc[i[0]][i[1]] for i in big_small])
    data2 = pd.to_numeric(data2, errors='ignore')
    df[f] = data2
    
    # ## Check first conversion
    # data1 = df[f + ' MPN']
    # data1 = pd.to_numeric(data1, errors='ignore')
    # diff = (data1 - data2).dropna().sum()
    # if diff != 0:
    #     print(f + ' results not the same')


### Time Series
df = df[~df['Sample #'].isin(['Mav','Cap','Blank'])]
df = df[['dt'] + FIB]

df.set_index('dt',inplace=True)

# tide
tide = pd.read_csv(os.path.join(folder, 'aux_data','San_Francisco_tide_water_level_20220731_20220804.csv'),
                   index_col=['dt'],parse_dates=['dt'])
tide = tide.resample('.5H').nearest()
tide = tide.tide

df = pd.merge(df, tide, how='left', left_index=True, right_index=True)

## LOQ, DIlution
dil = 10
BLOQ = 1
ALOQ = 10000
for f in FIB:
    df.loc[df[f] == '<1', f] = BLOQ / dil
    df.loc[df[f] == '>2419.6', f] = ALOQ / dil
    df[f] = dil * pd.to_numeric(df[f])
    df[f] = np.log10(df[f])

plt.figure(figsize=(10,6))
plt.rcParams['axes.xmargin'] = 0

plt.subplot(4,1,1)
plt.plot(df.tide, color='b')
plt.ylabel('Water Level (m)')
plt.xlabel('')
plt.gca().set_xticklabels([])

plt.subplot(4,1,2)
plt.plot(df.TC, color='k')
plt.axhline(np.log10(10000), color='k', alpha=0.5, ls=':')
plt.ylabel('TC')
plt.xlabel('')
plt.gca().set_xticklabels([])

plt.subplot(4,1,3)
plt.plot(df['E. Coli'], color='g')
plt.axhline(np.log10(400), color='g', alpha=0.5, ls=':')
plt.ylabel('EC')
plt.xlabel('')
plt.gca().set_xticklabels([])

plt.subplot(4,1,4)
plt.plot(df.ENT, color='r')
plt.axhline(np.log10(104), color='r', alpha=0.5, ls=':')
plt.ylabel('ENT')
plt.xlabel('')

plt.suptitle('Harbor Sampling Study - Aug 1 - 3, 2022')
plt.tight_layout()

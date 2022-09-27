#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:31:40 2022

@author: rtsearcy

- Combine field and auxilliary data into a single CSV
- Replace LOD and missing data

"""

import pandas as pd
import numpy as np
import os

basefolder = '/Volumes/GoogleDrive/My Drive/high_frequency_wq/harbor_study_2022/data/'

## Load data
field = pd.read_csv(os.path.join(basefolder,'field_data','field_data.csv'),
                    index_col=['dt'], parse_dates=['dt'])

tide = pd.read_csv(os.path.join(basefolder,'aux_data','sf_tide_6m_220731_220804.csv'),
                   index_col=['dt'], parse_dates=['dt'])
tide = tide['tide']

met = pd.read_csv(os.path.join(basefolder,'aux_data','hmb_met_data_30m_220730_220804.csv'),
                  index_col=['dt'], parse_dates=['dt'])

## Combine Data
df = pd.merge(field, tide, how='left', left_index=True, right_index=True)
df = pd.merge(df, met, how='left', left_index=True, right_index=True)

## Clean Data
df = df.reset_index()

# Replace BLOQ
for f in ['TC','FC','ENT']:
    df.loc[df[f] == '<10', f] = 1  

df.loc[df['shift'] =='Sprint', 'shift'] = 'sprint'
idx = df[(df.site == 'PP7') & (df['shift'] != 'sprint')].index
pp7 = df.iloc[idx]
other = df[~df.index.isin(idx)]

# interpolate missing data in variables with few missing
for v in ['rad','turb','temp','dtemp','pres','wspd','wdir']:
    pp7.loc[:,v] = pp7[v].interpolate(limit=2, limit_area='inside')
    if v == 'wdir':
        pp7.loc[pp7['wspd']==0, v] = np.nan  # no wdir if wspd = 0
        pp7.loc[:,v] = pp7.loc[:,v].round(0)

df = pp7.append(other).sort_index()
cols = [c for c in df.columns if c != 'notes']
df = df[cols + ['notes']]

## Save
df.to_csv(os.path.join(basefolder, 'all_data.csv'), index=False)

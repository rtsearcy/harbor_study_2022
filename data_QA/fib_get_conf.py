#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 12:04:47 2022

@author: rtsearcy

Get upper and lower confidence intervals for FIB data. Note: data already have MPN calculated

"""

import pandas as pd
import numpy as np
import os

folder = '/Volumes/GoogleDrive/My Drive/high_frequency_wq/harbor_study_2022/data/field_data/'
FIB = ['TC','FC','ENT']
dil = 10
BLOQ = 1

## IDEXX Data
df = pd.read_csv(os.path.join(folder, 'fib_wells.csv'))


## MPN + Confidence Tables
mpn_table = pd.read_csv(os.path.join(folder, 'mpn_table.csv'), index_col='large\small')
upper = pd.read_csv(os.path.join(folder, 'upper_conf.csv'), index_col='large\small')
lower = pd.read_csv(os.path.join(folder, 'lower_conf.csv'), index_col='large\small')

for i in ['0','48']:
    mpn_table[i] = pd.to_numeric([x.replace('\xa0','') for x in mpn_table[i]], errors='ignore')

for f in FIB:   
    big_small = list(zip(list(df[f + ' Big']), list(df[f + ' Small'])))
    # data2 = pd.Series([mpn_table.loc[i[0]][i[1]] for i in big_small])
    # data2 = pd.to_numeric(data2, errors='ignore')
    
    f_upper = pd.Series([upper.loc[i[0]][i[1]] for i in big_small])
    f_upper = pd.to_numeric(f_upper, errors='ignore')
    
    f_lower = pd.Series([lower.loc[i[0]][i[1]] for i in big_small])
    f_lower = pd.to_numeric(f_lower, errors='ignore')
    
    df[f + '_lower'] = dil*f_lower
    df[f + '_upper'] = dil*f_upper
    
    

# ## LOQ, DIlution
# for f in FIB:
#     bloq_idx = (df[f] == '<1')
#     df.loc[bloq_idx, f] = np.nan
#     df[f] = dil * pd.to_numeric(df[f])


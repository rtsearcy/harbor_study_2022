#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check if data digitization matches

Created on Mon Aug 29 17:02:23 2022

@author: rtsearcy
"""

import pandas as pd
import numpy as np

# Data Sheets (input file paths here)
file1 = '/Volumes/GoogleDrive/My Drive/high_frequency_wq/harbor_study_2022/data/Idexx and Field Data (1st Upload)/1_Combined_Data.xlsx' 
file2 = '/Volumes/GoogleDrive/My Drive/high_frequency_wq/harbor_study_2022/data/Idexx and Field Data (2nd Upload)/2_Combined_Data.xlsx'

# Load data
df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)

# Account for NaNs
df1[df1 == '-'] = np.nan  # replace dashes with NaN
df2[df2 == '-'] = np.nan

df1.fillna('NaN', inplace=True)  # because NaN cannot be equated
df2.fillna('NaN', inplace=True)

# Check matching
assert all(df1.columns == df2.columns), 'Column names mismatched'
assert len(df1) == len(df2), 'Lengths of datasets mismatched'
n = len(df1)

n_mismatch = n - (df1 == df2).sum()
n_mismatch.drop('notes', inplace=True)
print('N Mismatched:')
print(n_mismatch)

# Iterate by columns through missing values
check_cols = list(n_mismatch[n_mismatch > 0].index)
print('\n\nidx|Sheet 1|Sheet 2')
for c in check_cols:
    idx = np.where(df1[c] != df2[c])  # index where mismatch occurs
    if len(idx) == 0:
        continue

    df_mis = pd.concat([df1.loc[idx, c], df2.loc[idx, c]], axis=1)
    print('\n')
    print(df_mis)

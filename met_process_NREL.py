# met_process_NREL.py
# RTS 2022

# Processes solar rad data from NREL stored in csv files. 

import pandas as pd
import numpy as np
import os

# Inputs #
folder = '/Users/rtsearcy/Box/water_quality_modeling/forecasting/data/met/NREL/'
folder = '/Volumes/GoogleDrive/My Drive/high_frequency_wq/harbor_study_2022/data/hindcast'
beach = 'HMB'

### Read Raw Files
raw_folder = os.path.join(folder,'NREL_raw')
files = os.listdir(raw_folder)

df = pd.DataFrame()
for f in files:
    temp = pd.read_csv(os.path.join(raw_folder, f), skiprows=2)
    df = df.append(temp)

## Datetime
df['dt'] = pd.to_datetime( df[['Year', 'Month', 'Day','Hour','Minute']])
df.set_index('dt', inplace=True)
df.drop(['Year', 'Month', 'Day','Hour','Minute'], axis=1, inplace=True)
df.drop([c for c in df.columns if 'Unnamed' in c], axis=1, inplace=True)

## Daily mean/max value
df_daily = df.resample('D').mean()

# Save to file
df.to_csv(os.path.join(folder, 'NREL_hourly_' + beach + '.csv'))  # PD Series
df_daily.to_csv(os.path.join(folder, 'NREL_daily_mean_' + beach + '.csv'))  # PD Series

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 15:42:25 2022

@author: rtsearcy

Regressions on WQ Sonde parameters so external data can be used for prediction

"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV

import beach_model_harbor_study as bm


#%% Load Data
folder = '/Volumes/GoogleDrive/My Drive/high_frequency_wq/harbor_study_2022/data/param_regressions'

df = pd.read_csv(os.path.join(folder, 'data.csv'), 
                 index_col=['dt'], parse_dates=['dt'])

### Drop NA variables
to_drop = ['shift','day']
to_drop += [c for c in df if ('TC' in c) or ('FC' in c) or ('ENT' in c) or ('cloud' in c)]

df.drop(to_drop, axis=1, inplace=True)

### Log transform

for v in ['chl','turb']:
    df['log'+v] = np.log10(df[v])


#%% Cholorophyll

v = 'logchl'
EV = ['rad','dtemp', 'tide_30min']
## Linear Regression
lm = sm.GLSAR(endog = df[v], exog=sm.add_constant(df[EV]), 
              rho=10, hasconst=True).iterative_fit(maxiter=10)


print(lm.summary())
print('N - ' + str(len(lm.predict())))

print('\nRMSE: ' + str(round(lm.mse_resid**.5, 3)))  # OLS
print('\nVIFs:')
print(lm.params.index)
variables = lm.model.exog
print([round(VIF(variables, i),3) for i in range(variables.shape[1])])


### Plot Regression Output
#plot_regression(df_combo,lm)

#%% Turbidity
v = 'logturb'
EV = ['rad', 'temp', 'tide_high','pres', 'owind']
## Linear Regression
lm = sm.GLSAR(endog = df[v], exog=sm.add_constant(df[EV]), 
              rho=11, hasconst=True).iterative_fit(maxiter=10)


print(lm.summary())
print('N - ' + str(len(lm.predict())))

print('\nRMSE: ' + str(round(lm.mse_resid**.5, 3)))  # OLS
print('\nVIFs:')
print(lm.params.index)
variables = lm.model.exog
print([round(VIF(variables, i),3) for i in range(variables.shape[1])])


### Plot Regression Output
#plot_regression(df_combo,lm)

#%% Salinity
v = 'sal'
EV = ['owind', 'temp']
## Linear Regression
lm = sm.GLSAR(endog = df[v], exog=sm.add_constant(df[EV]), 
              rho=1, hasconst=True).iterative_fit(maxiter=10)


print(lm.summary())
print('N - ' + str(len(lm.predict())))

print('\nRMSE: ' + str(round(lm.mse_resid**.5, 3)))  # OLS
print('\nVIFs:')
print(lm.params.index)
variables = lm.model.exog
print([round(VIF(variables, i),3) for i in range(variables.shape[1])])


### Plot Regression Output
#plot_regression(df_combo,lm)









#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 12:57:17 2022

@author: rtsearcy

Performs multiple analyses on the Harbor Study data
- # samples BLOQ/exceeding threshold

"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf
import seaborn as sns

### Constants
FIB = {'TC':10000,
       'FC': 400,
       'ENT': 104
       }
LOQ = 10
beach_angle = 140

### Plot Specs
params = {
   'axes.labelsize': 11,
   'font.size': 11,
   'legend.fontsize': 10.5,
   'xtick.labelsize': 10,
   'ytick.labelsize': 11,
   'font.family'  : 'sans-serif',
   'font.sans-serif':'Helvetica',
   'axes.axisbelow': True
   }
plt.rcParams.update(params)

## Colors
pal_grey = ['#969696','#525252']  # grey, black
pal = ['#de425b','#2c8380']
pal3c = ['#003f5c', '#565089','#b1568f']
pal4c = ['#253494','#2c7fb8','#41b6c4','#a1dab4'] # 4 color blue tone
#pal = sns.color_palette(pal)

## Functions
def caps_off(axx): ## Turn off caps on boxplots
    lines = axx.lines
    for i in range(0, int(len(lines)/6)):
        lines[(i*6)+2].set_color('none')
        lines[(i*6)+3].set_color('none')

def flier_shape(axx, shape='.'):  ## Set flier shape on boxplots
    lines = axx.lines
    for i in range(0, int(len(lines)/6)):
        lines[(i*6)+5].set_marker(shape)

def plot_spines(axx, offset=8): # Offset position, Hide the right and top spines
    axx.spines['left'].set_position(('outward', offset))
    axx.spines['bottom'].set_position(('outward', offset))
    
    axx.spines['right'].set_visible(False)
    axx.spines['top'].set_visible(False)


#%% Load Data
folder = '/Volumes/GoogleDrive/My Drive/high_frequency_wq/harbor_study_2022/data/'
file = os.path.join(folder, 'all_data.csv')

df = pd.read_csv(file)   # read data using pandas (pd)
df = df.dropna(how='all')

df['dt'] = pd.to_datetime(df['dt'])  # convert to DateTime object
df.set_index('dt', inplace=True)  # Set dt as index

### Analysis variables
# Time
df['day'] = df.index.day
df['hour'] = df.index.hour
df['daytime'] = (df.hour.isin([8,9,10,11,12,13,14,15,16,17,18,19])).astype(int)
df['hours_from_noon'] = (12 - df.hour).abs() # hours from solar noon

# FIB
for f in FIB:
    # log transform
    df['log'+f] = np.log10(df[f])
    # Number of exceedances / BLOQ (FLAGS)
    df[f + '_exc'] = (df[f] > FIB[f]).astype(int)
    df[f + '_BLOQ'] = (df[f] < 10).astype(int)
    
# Tide
df['tide_high'] = (df.tide > 1).astype(int)
# df['tide_stage'] = 'ebb_flood'
# low_idx = [0,1,2,
#            20,21,22,23,24,25,26,27,
#            44,45,46,47,48,49,50,51,
#            68,69,70,71,72,73,74,75,76,
#            92,93,94,95]
# df.loc[df.iloc[low_idx].index, 'tide_stage'] = 'low'
# high_idx = [8,9,10,11,12,13,14,
#             32,33,34,35,36,37,38,
#             59,60,61,62,63,64,
#             80,81,82,83,84,85,86,87,88]
# df.loc[df.iloc[high_idx].index,'tide_stage'] = 'high'

# Met
df['awind'] = df['wspd'] * round(np.sin(((df['wdir'] - beach_angle) / 180) * np.pi), 1)
df['awind'].fillna(0, inplace=True) # for 0 wind speed values
df['owind'] = df['wspd'] * round(np.cos(((df['wdir'] - beach_angle) / 180) * np.pi), 1)
df['owind'].fillna(0, inplace=True)
# df['owind_bin'] = (df.owind > 0).astype(int)
# df.loc[df['owind'].isna(),'owind_bin'] = np.nan

### Separate data
df = df[df.site != 'Control']  # drop controls
df_spatial = df.copy().loc[df[df.site=='Mav'].index]
df_sprint = df.copy().loc['2022-08-02 11:00:00':'2022-08-02 11:30:00']
df = df[(df.site == 'PP7') & (df['shift'] != 'sprint')]

### Save for modeling
df_save = df.copy()
df_save.drop(['TC','FC','ENT', 'site','gust','wdir',
              'ceiling', 'rain', 'notes'], axis=1, inplace=True) # drop
df_save.to_csv(os.path.join(folder, 'PP7_variables.csv'))

#%% Plot FIB Time Series
plot_type = 'stem'  # line, stem

plt.figure(figsize=(9,5.5))

c = 1
for f in FIB:
    plt.subplot(4,1,c)
    if plot_type == 'line':
        plt.plot(df['log'+ f], 'k') # line plot 
        #plt.plot(df[f])
        ll = 0
    elif plot_type == 'stem':
        plt.bar(df.index,df['log'+f] + 0.5, bottom = -.5, width=.0075, color=pal3c[c-1], alpha=0.75)
        #plt.scatter(df.index,df['log'+f], color='k', s=3)
        ll = -0.15
    
    plt.axhline(np.log10(FIB[f]), color=pal3c[c-1], ls=':', lw=1)
    
    if c==2:
        plt.ylabel(r'log$_{10}$ MPN/100 ml')
    else:
        plt.ylabel('')
    plt.ylim(ll,3.5)

    plt.xlim(df.index[0], df.index[-1])
    plt.gca().set_xticklabels([])
        
    plt.text(.95, .85, f, transform=plt.gca().transAxes)
    
    plot_spines(plt.gca(), offset=0)
    c+=1

# Tide
plt.subplot(4,1,4)
plt.plot(df.tide, 'b', alpha=0.85)

# Sunrise Sunset
plt.fill_betweenx([-5,5], df.index[25], df.index[45], color='gray', alpha=0.4)
plt.fill_betweenx([-5,5], df.index[73], df.index[93], color='gray', alpha=0.4)

plt.ylabel('Water Level [m]')
plt.xlim(df.index[0], df.index[-1])
plt.ylim(0,2)

plt.text(.95, .85, 'tide', transform=plt.gca().transAxes)
plot_spines(plt.gca(), offset=0)
    
plt.tight_layout()


#%% Summary Stats / Variation

## Overall
print(df[FIB].describe().round(0))
print('\n')
for f in FIB:
    print(f + ' BLOQ: ' + str(df[f+'_BLOQ'].sum()) + ' (' + str((100*df[f+'_BLOQ'].sum()/len(df)).round(1)) + '%)'  )
    print(f + ' EXC: ' + str(df[f+'_exc'].sum()) + ' (' + str((100*df[f+'_exc'].sum()/len(df)).round(1)) + '%)'  )


## Coefficient of Variation
CV =  100 * df[FIB].std() / df[FIB].mean() # coefficienrt of variations (stdev / mean)
# measurement of dispersion (normalized)
print('\nCV (%, All Data)')
print(CV.round(1))

## Differences in subsequent samples
for f in FIB:
    print('\n' + f)
    print('delta:') # difference in consec. samples, normalized by experimental mean
    print((100*df[f].diff().abs() / df[f].mean()).describe().round(1))

    print('\n% samples changing status (BLOQ/EXC):') #  % SAMPLES THAT CHANGE BLOQ/EXC STATUS
    print(round(100* df[f + '_BLOQ'].diff().abs().sum() / (len(df) - 1), 1))
    print(round(100* df[f + '_exc'].diff().abs().sum() / (len(df) - 1), 1))
    
    # Shanon Entropy
    vals, counts = np.unique(df[f],return_counts=True)
    print('\nShannon Entropy: ')
    print(round(stats.entropy(counts/len(df[f])),3))
    
## FIB by shift
#sns.boxplot(x='shift', y = 'value', hue='variable', data = df.melt(id_vars=['shift'],value_vars=['logTC','logFC','logENT']))


#%% Autocorrelation
acf_type = 'auto'  # auto, partial

plt.figure(figsize=(7.5,5))
c=1
for f in FIB:
    plt.subplot(3,1,c)
    #plot_acf(df.dropna(), zero=False, ax=plt.gca(), marker='')
    if acf_type == 'auto':
        rho = acf(df['log'+f], nlags=12)
    elif acf_type == 'partial':
        rho = pacf(df['log'+f],nlags=12)  # Partial autocorrelation
    
    plt.stem(range(0,len(rho)), rho, linefmt='k-', markerfmt=' ', basefmt='k-')
    ax = plt.gca()

    plt.ylabel(r'$\rho$', rotation=0)      
    plt.ylim(-.55,1.05)
    
    plt.text(.95, .85, f, transform=plt.gca().transAxes)
    
    if c == [3]:
        plt.xlabel('Lag')
    
    #plt.xlim(.75,20.25)
    #plt.xticks(ticks=range(1,21), 
    #           labels =[1,'',3,'',5,'',7,'',9,'',11,'',13,'',15, '',17,'',19,''])
                        #21,'',23,''])
    
    plt.axhline(1.96/(len(df)**.5), ls='--', color='grey', alpha=0.7)
    plt.axhline(-1.96/(len(df)**.5), ls='--', color='grey', alpha=0.7)
    
    
    #ax.axes.get_lines()[0].set_color('k')  # color x axis black
    
    plot_spines(plt.gca(), offset=0)
    c+=1

if acf_type == 'auto':
    plt.suptitle('Autocorrelation')
elif acf_type == 'partial':
    plt.suptitle('Partial Autocorrelation')
plt.tight_layout()


#%% Downsampling Analysis 
'''CV / delta of FIB by different sampling rates
Sampling Rates: 30m, 1h, 2h, 3h, 6h, 12h, 24h
There will be error around these numbers because of choice of start time
'''
sample_rates = ['30m', '1H', '2H', '3H', '6H', '12H','24H']
CV_mean = pd.DataFrame()
CV_std = pd.DataFrame()

D_mean = pd.DataFrame()
D_std = pd.DataFrame()

cv = lambda x: 100 * np.std(x, ddof=1) / np.mean(x) # coefficient of variations (stdev / mean)
delta = lambda x: 100 * (x.diff().abs() / x.mean()).mean().round(3)

for s in sample_rates:
    if s == '30m':
        df_ds = df.copy()
        CV = cv(df_ds[FIB]) 
        std = pd.Series(dtype=float)
        
        D = delta(df_ds[FIB])
        D_std_temp = pd.Series(dtype=float)
        
        
    else:
        temp = pd.Series(dtype=float)
        temp_d = pd.Series(dtype=float)
        for i in range(0, 2*int(s.replace('H',''))):  # iterate through shift choices
            df_ds = df.shift(-i).resample(s, origin='start').nearest() # take nearest sample to every 1 hour
            
            temp = pd.concat([temp, cv(df_ds[FIB])], axis=1)
            temp_d = pd.concat([temp_d, delta(df_ds[FIB])], axis=1)
            
        CV = temp.mean(axis=1)
        std = temp.std(axis=1)
            
        D = temp_d.mean(axis=1)
        D_std_temp = temp_d.std(axis=1)
            
            
    CV_mean = pd.concat([CV_mean, CV.to_frame(name=s)], axis=1)
    CV_std = pd.concat([CV_std, std.to_frame(name=s)], axis=1)
    CV_std = CV_std.fillna(0)
    
    D_mean = pd.concat([D_mean, D.to_frame(name=s)], axis=1)
    D_std = pd.concat([D_std, D_std_temp.to_frame(name=s)], axis=1)
    D_std = D_std.fillna(0)


## Plot
plt.figure(figsize=(8,6))
x = range(0,len(CV_mean.columns))

# CV
plt.subplot(2,1,1)
c = 0
for f in FIB:
    plt.plot(x, CV_mean.loc[f], marker='o', color=pal3c[c], label=f) # mean
    plt.fill_between(x, CV_mean.loc[f]-CV_std.loc[f], CV_mean.loc[f]+CV_std.loc[f], color=pal3c[c], alpha=0.3) # error
    c+=1

plot_spines(plt.gca())
plt.ylabel('%')
plt.xlabel('')
#plt.xticks(ticks=x, labels=sample_rates)
plt.gca().set_xticklabels([])
plt.text(-.0, .9, 'CV', transform=plt.gca().transAxes)
plt.legend(frameon=False)

# Delta
plt.subplot(2,1,2)
c = 0
for f in FIB:
    plt.plot(x, D_mean.loc[f], marker='o', color=pal3c[c], label=f) # mean
    plt.fill_between(x, D_mean.loc[f]-D_std.loc[f], D_mean.loc[f]+D_std.loc[f], color=pal3c[c], alpha=0.3) # error
    c+=1

plot_spines(plt.gca())
plt.ylabel('%')
plt.xlabel('')
plt.xticks(ticks=x, labels=sample_rates)
plt.text(-.0, .9, 'delta', transform=plt.gca().transAxes)

plt.tight_layout()

#%% Multivariate Stats

### Correlations
print('\nCorrelations')
C = df.corr(method = 'spearman')
print(C[FIB].loc[FIB].round(2)) # w FIB
print('\n')

evs = ['tide','wtemp','sal','turb','chl','rad','temp','dtemp', 'pres','owind','awind']
print(C[FIB].loc[evs + ['hours_from_noon']].round(2)) # w Enviro Vars

### Cross correlation
lags = range(0,12)
sig_p = 0.05

for f in FIB:
    df_rho = pd.DataFrame(index=[evs],columns = list(lags))
    df_p = df_rho.copy()
    
    for e in evs:
        for i in lags:        
            temp = pd.concat([df[f], df[e].shift(i)],axis=1).dropna()
            rho, p = stats.spearmanr(temp[f], temp[e])
            df_rho.loc[e,i] = rho
            df_p.loc[e,i] = p
    
    plt.figure(figsize=(7,5))
    
    df_annot = df_rho.astype(float).round(2).astype(str) + (df_p < 0.05).astype(int).replace([0,1],['','*']).astype(str)

    cmap = sns.diverging_palette(220, 20, as_cmap=True) 
    sns.heatmap(df_rho.astype(float), annot=df_annot, fmt='s',
                linecolor='k', linewidths=.25,
                cmap= cmap, 
                cbar_kws={'label':'Spearman Rank Correlation'},
                vmin=-0.3,vmax=0.3, robust=False, center=0) # YlGnBu_r
    plt.xlabel('Lag')
    plt.ylabel('')
    plt.title(f)

    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        left=False) 
    
    plt.tight_layout()


### Enviro Groupings
''' NOTE: OWIND_BIN ALWAYS OFFSHORE
tide_high when tide above/below 1m
'''

for v in ['tide_high', 'tide_stage', 'daytime']:
    print('\n'+v)
    print(df.groupby(v).describe()[['FC','ENT']].T.round(0))
    print(df.groupby(v).sum()[['ENT_BLOQ','ENT_exc']])
    
    ## Hypothesis Testing (EV Regimes)
    temp = df.copy().melt(id_vars=v, value_vars=['logTC','logFC','logENT'], var_name='FIB', value_name='conc').dropna()
    print('\n'+v + ' (N=' + str(len(temp)/2) + ')')
    for f in FIB:
        kw_temp = temp[temp.FIB == 'log'+f]
        if v == 'tide_stage':
            kw = stats.kruskal(kw_temp[kw_temp[v]=='low']['conc'],
                               kw_temp[kw_temp[v]=='high']['conc'],
                               kw_temp[kw_temp[v]=='ebb_flood']['conc'])
        else:
            kw = stats.kruskal(kw_temp[kw_temp[v]==0]['conc'],kw_temp[kw_temp[v]==1]['conc'])
        print('\n' + f + ' - ' + str(round(kw[1],3)))
        print('mean/med:')
        print(kw_temp.groupby([v]).describe().round(2)['conc'][['mean','50%']])
    
    ## Plots
    plt.figure(figsize=(6,3))
    sns.boxplot(x='variable',y='value',hue=v,
                data=df.melt(value_vars=['logTC','logFC','logENT'], id_vars=[v]))
    
    plt.ylabel(r'log$_{10}$ MPN/100 ml')
    plt.xlabel('')
    plt.xticks(ticks=[0,1,2], labels=FIB)
    
    plot_spines(plt.gca())
    caps_off(plt.gca())
    flier_shape(plt.gca())
    plt.tight_layout()


#%% Sprint
plot_type = 'stem'  # line, stem

## TS Plot
plt.figure(figsize=(9,4))

c = 1
for f in FIB:
    plt.subplot(3,1,c)
    
    if plot_type == 'line':
        plt.plot(df_sprint['log'+ f], 'k') # line plot 
        #plt.plot(df[f])
        ll = 0
    elif plot_type == 'stem':
        plt.bar(df_sprint.index,df_sprint['log'+f] + 0.5, bottom = -.5, width=.0001, color='k', alpha=0.75)
        #plt.scatter(df_sprint.index,df_sprint['log'+f], color='k', s=3)
        ll = -0.15
    
    plt.axhline(np.log10(FIB[f]), color='k', ls=':', lw=1)
    
    if c==2:
        plt.ylabel(r'log$_{10}$ MPN/100 ml')
    else:
        plt.ylabel('')
    plt.ylim(ll, 4)

    plt.xlim(df_sprint.index[0], df_sprint.index[-1])
    if c < 3:
        plt.gca().set_xticklabels([])
    
    plt.text(.95, .85, f, transform=plt.gca().transAxes)
    plot_spines(plt.gca(), offset=0)
    c+=1

plt.suptitle('Sprint Sampling')
plt.tight_layout()


## Coefficient of Variation
CV =  df_sprint[FIB].std() / df_sprint[FIB].mean() # coefficienrt of variations (stdev / mean)
# measurement of dispersion (normalized)
print('CV')
print(CV.round(2))

## Differences in subsequent samples
for f in FIB:
    print('\n' + f)
    print('delta:') # difference in consec. samples, normalized by experimental mean
    print((df_sprint[f].diff().abs() / df_sprint[f].mean()).describe().round(3))

    print('\n% samples changing status (BLOQ/EXC):') #  % SAMPLES THAT CHANGE BLOQ/EXC STATUS
    print(round(100* df_sprint[f + '_BLOQ'].diff().abs().sum() / (len(df_sprint) - 1), 1))
    print(round(100* df_sprint[f + '_exc'].diff().abs().sum() / (len(df_sprint) - 1), 1))
    
    # Shanon Entropy
    vals, counts = np.unique(df_sprint[f],return_counts=True)
    print('\nShannon Entropy: ')
    print(round(stats.entropy(counts/len(df_sprint[f])),3))

#%% Spatial Data
df_spatial.groupby('site').describe()[FIB].T.round(0)


# Time Series
plt.figure(figsize=(9,5.5))
c = 1
for f in FIB:
    plt.subplot(4,1,c)
    plt.plot(df_spatial[df_spatial.site == 'PP7']['log'+ f], pal3c[0], marker='.', label='PP7') # line plot 
    plt.plot(df_spatial[df_spatial.site == 'Cap']['log'+ f], pal3c[1], marker='.', label='Cap')
    plt.plot(df_spatial[df_spatial.site == 'Mav']['log'+ f], pal3c[2], marker='.', label='Mav')
    plt.axhline(np.log10(FIB[f]), color='k', ls=':', lw=1)
    
    plt.ylabel(r'log$_{10}$ MPN/100 ml')
    if f == 'TC':
        plt.ylim(0,6)
    else:
        plt.ylim(0,4)

    plt.xlim(df.index[0], df.index[-1])
    plt.gca().set_xticklabels([])
        
    plt.text(.95, .85, f, transform=plt.gca().transAxes)
    
    if c == 1:
        plt.legend(frameon=False, loc='upper center', ncol=3)
    
    plot_spines(plt.gca(), offset=0)
    c+=1

# Tide
plt.subplot(4,1,4)
plt.plot(df.tide, 'b', alpha=0.85)

# Sunrise Sunset
plt.fill_betweenx([-5,5], df.index[25], df.index[45], color='gray', alpha=0.4)
plt.fill_betweenx([-5,5], df.index[73], df.index[93], color='gray', alpha=0.4)

plt.ylabel('Water Level [m]')
plt.xlim(df.index[0], df.index[-1])
plt.ylim(0,2)

plt.text(.95, .85, 'tide', transform=plt.gca().transAxes)
plot_spines(plt.gca(), offset=0)
    
plt.tight_layout()

# Boxplots
plt.figure(figsize=(6,3))
sns.boxplot(x='site',y='value', hue='variable',palette=pal3c,
            data=df_spatial.copy().melt(id_vars='site', value_vars=['logTC','logFC','logENT']).dropna())
plt.ylabel('rlog$_{10}$ MPN/100 ml')
plt.xlabel('')

plot_spines(plt.gca())
caps_off(plt.gca())
flier_shape(plt.gca())

plt.legend(frameon=False, loc='upper left', ncol=3)
plt.tight_layout()

## Relationship w EVs
df_spatial.groupby(['site','tide_high']).describe()['ENT'].round()


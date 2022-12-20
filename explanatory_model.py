#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 15:09:53 2022

@author: rtsearcy

Explanatory Modeling - 2022 HMB Harbor Study

"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import confusion_matrix, roc_curve, r2_score, roc_auc_score, balanced_accuracy_score
from sklearn.feature_selection import RFE, RFECV, VarianceThreshold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures



def lag_vars(df, var_list, n_lags, interval=None):
    ''' Creates lag variables. If 'interval', shift by that number of minutes'''
    
    for v in var_list:
        for i in range(1, n_lags+1):
            if interval == None:
                df[v+str(i)] = df[v].shift(i)  # variable i time steps previous
            else:
                # Shift data by interval
                df[v+'_' + str(i*interval)+'min'] = df[v].shift(i, freq=str(interval)+'min')
    
    df = df.dropna(axis=1, how='all') # drops variable with unaligned timestep
    
    return df

def get_interaction(x, interact_var):
    # Create interaction variables
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_vars = pd.DataFrame(poly.fit_transform(x[interact_var]), 
                             columns=poly.get_feature_names(interact_var), index=x.index)
    poly_vars = poly_vars[[v for v in poly_vars.columns if v not in x.columns]]
    return poly_vars

def add_var(y, X, model_type, score, selected_vars=[], skip_vars=[]):
    # empty dictionary that will be used to store results
    metrics_dict = {'predictor': [], 'r2':[] ,'aic':[], 'p_val':[]}  # 'rmse'
    
    # Iterate through every column in X
    for col in X.columns:
        if col not in selected_vars + skip_vars:
            selected_X = X[[col] + selected_vars] #X[[col]]
            
            #Fit a model for target and selected columns
            # model = sm.OLS(y, sm.add_constant(selected_X)).fit()
            if model_type == 'GLSAR':
                model = sm.GLSAR(endog = y, 
                                 exog=sm.add_constant(selected_X), 
                                 rho=2, 
                                 hasconst=True).iterative_fit(maxiter=10)
            elif model_type == 'BLR':
                model = sm.Logit(endog = y>0,
                                 exog = sm.add_constant(selected_X)).fit(disp=False)
            
            #Predict what our target would be for our model
            y_pred = model.predict(sm.add_constant(selected_X))
            #Add the column name to our dictionary
            metrics_dict['predictor'].append(col)
            
            #Calculate metrics between the target and predicted target
            if model_type == 'GLSAR':
                r2 = model.rsquared_adj
            else:
                r2 = model.prsquared
            #rmse = np.sqrt(((y - y_pred)**2).mean())
            aic = - round(model.aic, 3)
            p = - model.pvalues.loc[col]
            
            #Add values to our dictionary
            metrics_dict['r2'].append(r2)
            # metrics_dict['rmse'].append(rmse)
            metrics_dict['aic'].append(aic)
            metrics_dict['p_val'].append(p)
        
    #Once iterated through every column, turn dictionary into a DataFrame and sort it
    mets = pd.DataFrame(metrics_dict).sort_values(by=[score], ascending = False).reset_index()
    top_var = mets.iloc[0]['predictor']
    top_p = -mets.iloc[0]['p_val']
    print('Selected: ' + top_var + \
          ' (R2: ' + str(round(mets.iloc[0]['r2'],3)) + ', ' + \
          ' Neg. AIC: ' + str(mets.iloc[0]['aic']))
    
    return top_var, top_p

def stepwise_select(y, X, model_type='GLSAR', score='p_val', alpha=0.05):
    ''' Use stepwise variable selection to find optimal variables. Check VIF '''
    ind_vars = []
    skip_vars = []
    c = 0
    while c < len(X.columns):
        # Find variable that minimizes p_val
        top_var, top_p = add_var(y, X, 
                                 model_type=model_type,
                                 score=score,
                                 selected_vars=ind_vars, 
                                 skip_vars=skip_vars)
        
        # Test if p_val < alpha
        if top_p > alpha:
            print('  not significant')
            break
        else:
            ind_vars += [top_var]
        
        # Test VIF of current ind_vars
        if c > 0:
            vif = [VIF(X[ind_vars], i) for i in range(len(X[ind_vars].columns))]
            if any(v > 5 for v in vif):
                print('  VIF > 5. Removed.')
                skip_vars += [ind_vars.pop()]
            
        c+=1
        
    return ind_vars

def fit_rf(X, y, perm_thresh=1.5):  
    
    '''Fits Random Forest model'''

    if y.dtype == 'float':
        rf = RandomForestRegressor(n_estimators=1000, 
                                   oob_score=True,
                                   #max_depth=3,  # None - expands until all leaves are pure
                                   max_features=.75,
                                   #max_samples=.75,
                                   random_state=0)
            
        score_metric = 'r2' #'neg_root_mean_squared_error' # r2
        
    else:
        rf = RandomForestClassifier(n_estimators=1000, 
                                   oob_score=True,
                                   #max_depth=3,  # None - expands until all leaves are pure
                                   max_features=.75,
                                   #max_samples=.75,
                                   random_state=0)
        score_metric = 'accuracy' # accuracy
        
    ## Variable Selection    
    # Random Forest Permutation method (narrow down big variable amounts)
    print('\nNarrowing down variables w/ permutation importances (scoring: ' + score_metric + ')')
    rf.fit(X,y)
    temp = permutation_importance(rf, X, y, 
                                  scoring=score_metric, 
                                  random_state=0,
                                  n_repeats=5)['importances_mean']
    temp = pd.Series(data=temp, index=X.columns).sort_values(ascending=False)

    features = list(temp[temp > perm_thresh*temp.mean()].index)  # Select the variables > X times the mean importance
    assert len(features) > 0, 'Random Forest Regression failed to select any variables'
    # if len(features) < 3:
    #     features = list(temp.index[0:3])
    
    print('  ' + str(len(features)) + ' features selected')
    # print(*features)
    X = X[features]
    
    # ## Recursive Feature Elimination - Stepwise variable selection
    # print('\nRFECV Variable Selection (scoring: ' + score_metric + ')')
    # S = RFECV(rf, 
    #           cv=cv, 
    #           scoring=score_metric,
    #           min_features_to_select=3,
    #           verbose=0).fit(np.array(X), np.array(y))
    
    # features = X.columns[list(np.where(S.support_)[0])]
    # print('\n' + str(len(features)) + ' feature(s) selected')
    # print(*features)
        
    ### Fit Model
    rf = rf.fit(X[features],y)

    return list(features), rf 

def plot_regression(y, pred, f, model_type):
    resid = y - pred
    
    plt.figure(figsize=(12, 4))

    plt.subplot(1,3,1)
    plt.scatter(y, pred)
    plt.plot(np.arange(0,4), np.arange(0,4), ls = ':', color='k')
    plt.ylabel('Modeled')
    plt.xlabel('Actual')
    plt.title(f + ' - Actual vs Predicted')
    
    plt.subplot(1,3,2)
    plt.scatter(pred, resid)
    plt.axhline(0, ls=':', color='k')
    plt.title(f + ' - residual vs fitted')
    plt.xlabel('Modeled')
    plt.ylabel('Residuals')
    
    plt.subplot(1,3,3)
    plt.scatter(resid.index, resid)
    plt.axhline(0, ls=':', color='k')
    plt.title(f + ' - residual vs time')
    plt.ylabel('Residuals')
    
    plt.suptitle(model_type + ' Model')
    plt.tight_layout()
    
#%% Inputs + Load Data
folder = '/Volumes/GoogleDrive/My Drive/high_frequency_wq/harbor_study_2022/data/'
f ='ENT'
output_bin = False

# Constants
FIB = {'TC':10000,
       'FC': 400,
       'ENT': 104
       }
LOQ = 10
alpha = 0.05  # p value maximum

# Dependent Variable
if output_bin:
    dep_var = f + '_exc'
else:
    dep_var = 'log' + f


# Load df
df = pd.read_csv(os.path.join(folder, 'PP7_variables.csv'),
                 index_col=['dt'],
                 parse_dates=['dt'])
#print(df.info())

#%% Pre-Process

### Drop unessecary vars
drop_list = ['shift', 'hour','day']
 
# Drop other FIB variables
# We do not consider other FIB in the models
other_fib = [x for x in FIB if x != f]
for x in other_fib:
    drop_list += [v for v in df.columns if x in v]

# Drop same FIB variables that do not apply
drop_list +=  [v for v in df.columns if (f in v) and (v != dep_var)] 

df = df.drop(drop_list, axis=1)

### Salinity fill
df['sal'] = df.sal.fillna(method='bfill')  # only 1 missing value at beginning

### Cloud Dummies
df.loc[df.cloud.isin(['few','broken','scattered']),'cloud'] = 'partly'  # combine minority classes
cloud_vars = pd.get_dummies(df['cloud'], prefix='cloud') #.drop(['cloud_clear'], axis=1)
# dummies for overcast and partly cloudy. 0 for both indicates clear skies

df = df.drop(['cloud'], axis=1).merge(cloud_vars, left_index=True, right_index=True)

### Sal Pressure adjustment
# df['sal'] = 34 - df.sal
# df['pres'] = df.pres - 1000  # units of mbar - 1000

### Log Transform
for v in ['chl','turb']:
    df['log'+v] = np.log10(df[v] + 1).round(3)
    df = df.drop([v], axis = 1)

print(*df.columns)

### Create lag variables
evs = ['tide','tide_high', 'wtemp','logchl','logturb','sal',
       'rad', 'awind','owind', 'pres','temp','dtemp', 
       'vis', 'cloud_overcast','cloud_partly','cloud_clear']

df = lag_vars(df, var_list = evs, n_lags = 6, interval=30).dropna()
print('  added lag variables up to 1hr previous\n')

### Narrow down lags
# Select lag most correlated w FIB
# Reduce multicollinearity
lag_drop = []
corr_method = 'spearman'
df_corr = df.corr(method=corr_method)[dep_var]
for e in evs:
    temp = df_corr.loc[[v for v in df_corr.index if e in v]].abs().sort_values(ascending=False)
    lag_drop += list(temp.index[1:])
    
df = df.drop(lag_drop, axis=1)

print('Lags parsed. ' + str(len(df.columns)) + ' vars remaining.\n' + corr_method.capitalize() + ' corrleations w ' + dep_var)
print(df.corr()[dep_var].sort_values().round(2))
print('\n')

### Create interaction variables (Skip)

### Split
y = df[dep_var].copy()
X = df.drop([dep_var], axis=1).copy()

### Standardize
scaler = StandardScaler()
scaler.fit(X)
cols = X.columns
X = scaler.transform(X)
X = pd.DataFrame(data=X, columns = cols, index=y.index)

#%% Explanatory Regressions
if f == 'TC':
    # some methods from https://medium.com/@garrettwilliams90/stepwise-feature-selection-for-statsmodels-fda269442556
    print('\n- - GLS Regression - -')
    ### Variable Selection
    ind_vars = stepwise_select(y, X, 
                               model_type='GLSAR', 
                               alpha=alpha)
    
    ### Fit Model
    # Generalized Linear Regression
    lm = sm.GLSAR(endog = y, 
                  exog=sm.add_constant(X[ind_vars]), 
                  rho=2, 
                  hasconst=True).iterative_fit(maxiter=10)
    
    y_pred = lm.predict(sm.add_constant(X[ind_vars]))
    print(lm.summary2())
    #print(pd.concat([lm.params.round(2), lm.pvalues.round(2)], axis=1))
    print('R2: ' + str(round(np.corrcoef(y, y_pred)[0, 1]**2, 3)))
    print('RMSE: ' + str(round((((y-y_pred)**2).mean())**.5, 3)))
    
    print('\nVIFs:')
    print(lm.params.index)
    variables = lm.model.exog
    print([round(VIF(variables, i),3) for i in range(variables.shape[1])])
        
    ### Plot Regression Output
    plot_regression(y, y_pred, f, 'GLS Regression')


#%% Random Forest Regression
# print('\n- - Random Forest - -')
# rf_vars, rf = fit_rf(X, y, perm_thresh=1.5) #neg_root_mean_squared_error

# print(rf)
# print(pd.DataFrame(data=rf.feature_importances_, index=rf_vars, columns=['importances']))

# rf_pred = rf.predict(X[rf_vars])
# rf_resid = y - rf_pred

# print('\nR2: ' + str(round(np.corrcoef(y, rf_pred)[0, 1]**2, 3)))
# print('RMSE: ' + str(round((((y-rf_pred)**2).mean())**.5, 3)))  # OLS


# ### Plot RF Regression
# plot_regression(y, rf_pred, f, 'Random Forest')

# ### Plot Partial Dependence Plots
# # https://christophm.github.io/interpretable-ml-book/ice.html#ice
# # https://www.tandfonline.com/doi/full/10.1080/07350015.2019.1624293
# PartialDependenceDisplay.from_estimator(rf, 
#                                         X[rf_vars], 
#                                         rf_vars, 
#                                         kind='both',
#                                         pd_line_kw = {'color':'k'})
# figure = plt.gcf()
# #figure.set_size_inches(9,3)

# axes = figure.axes
# c=0
# for a in figure.axes[1:]:
#     plt.sca(a)
#     plt.xlabel('')
#     plt.ylabel('')
#     plt.title(rf_vars[c]) 
#     plt.legend([], frameon=False)
#     c+=1

# plt.suptitle('Random Forest')
# plt.tight_layout()


#%% Hurdle Model
else:
    
    '''Logistic model to predict whether FIB > 0, Regression to predict concentration
    Prediction is the product of the two models
    Cite: https://data.library.virginia.edu/getting-started-with-hurdle-models/'''
    print('\n- - Hurdle Model - -')
    
    ### Binary Model
    print('Binary Model')
    
    # # Logistic Model
    # bin_vars = stepwise_select(y, X, model_type='BLR', alpha=alpha)
    # bin_model = sm.Logit(endog = y>0,
    #                      exog = sm.add_constant(X[bin_vars])).fit(disp=False)
    # bin_pred = bin_model.predict(sm.add_constant(X[bin_vars]))
    
    # print(bin_model.summary2())
    # print('Observed/Predicted:')
    # pt = bin_model.pred_table()
    
    
    # Random Forest Classifier
    bin_vars, bin_model = fit_rf(X, y>0, perm_thresh=1.5)
    print(bin_model)
    print(pd.DataFrame(data=bin_model.feature_importances_, index=bin_vars, columns=['importances']))
    print('\n')
    
    bin_pred = bin_model.predict_proba(X[bin_vars])[:,1]
    pt = confusion_matrix(y>0, bin_pred > 0.5)
    
    print(pt)
    print('acc: ' + str(round((pt[1,1] + pt[0,0]) / pt.sum(), 3)))
    print('sens: ' + str(round(pt[1,1] / (pt[1,1] + pt[1,0]), 3)))
    print('spec: ' + str(round(pt[0,0] / (pt[0,0] + pt[0,1]), 3)))
    
    
    ### Concentration Model
    print('\nConcentration Model')
    conc_vars = stepwise_select(y[y>0], X[y>0], model_type='GLSAR', alpha=alpha)
    conc_model = sm.GLSAR(endog = y[y>0], 
                 exog=sm.add_constant(X[y>0][conc_vars]), 
                 rho=2, 
                 hasconst=True).iterative_fit(maxiter=10)
    
    conc_pred = conc_model.predict(sm.add_constant(X[conc_vars]))
    print(conc_model.summary2())
    print('N - ' + str(len(conc_model.predict())))
    print('R2: ' + str(round(np.corrcoef(y, conc_pred)[0, 1]**2, 3)))
    print('RMSE: ' + str(round(conc_model.mse_resid**.5, 3)))  # OLS
    print('VIFs:')
    print(conc_model.params.index)
    variables = conc_model.model.exog
    print([round(VIF(variables, i),3) for i in range(variables.shape[1])])
    
    
    ### Combined into Hurdle Model
    print('\nHurdle Model Results:')
    hurdle_pred = (bin_pred > 0.5).astype(int) * conc_pred
    #hurdle_pred = bin_pred * conc_pred
    hurdle_resid = y - hurdle_pred
    
    print('R2: ' + str(round(np.corrcoef(y, hurdle_pred)[0, 1]**2, 3)))
    print('RMSE: ' + str(round((((y-hurdle_pred)**2).mean())**.5, 3)))  # OLS
    
    ### Plot Hurdle Output
    plot_regression(y, hurdle_pred, f, 'Hurdle')



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 15:42:25 2022

@author: rtsearcy

Explanatory/Predictive modeling for Harbor Study FIB

"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoCV, ElasticNet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, roc_curve, r2_score, roc_auc_score, balanced_accuracy_score
from sklearn.feature_selection import RFE, RFECV, VarianceThreshold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)

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

def select_vars(y, X, method='forest', interaction=True, interact_var=[], corr_check=True):
    '''
    Selects best variables for modeling from dataset.
    
    Parameters:
        - y = Dependant variable
        
        - X = Independant variables
        
        - method = Variable selection method
            - 'lasso' - Lasso Regressions - insignificant variables automatically 
               assigned 0 coefficient
            - 'rfe' - Recursive Feature Elimination - Selects best features
            - 'forest' - Random Forest
                
    Output:
        - Dataset with the best variables to use for modeling
    '''
    print('\n\n- - | Selecting Variables | - -')
    print('\nOriginal # of Variables - ' + str(len(X.columns)))
    
    # Interaction terms
    if interaction:    
        poly_vars = get_interaction(X, interact_var)
        X = X.merge(poly_vars, left_index=True, right_index=True)
    
    # Check similarly correlated vars to FIB 
    if corr_check:
        X = check_corr(y, X)  
        
    # Select variables
    print('\nVariable Selection Method: ' + method.upper() + '\n')
    multi=True
    c=0
    while multi:
        if method == 'lasso':  # LASSO
            lm = LassoCV(cv=5, normalize=True).fit(X, y)
            new_vars = list(X.columns[list(np.where(lm.coef_)[0])]) # vars Lasso kept
            assert len(new_vars) > 0, 'Lasso regression failed to select any variables'
            X = X[new_vars]
        elif (method == 'forest') & (c==0): # Random Forest
        # Ref: Jones et al 2013 - Hydrometeorological variables predict fecal indicator bacteria densities in freshwater: data-driven methods for variable selection
            # Only run once
            rf = RandomForestRegressor(n_estimators=500, 
                                       oob_score=True, 
                                       random_state=0, 
                                       #max_samples=.75,
                                       max_features=.75
                                       )
            #Xs = X
            # Scale by mean and std dev
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            
            rf.fit(Xs,y)
            temp = permutation_importance(rf,Xs,y,random_state=0, n_repeats=10)['importances_mean']
            temp = pd.Series(data=temp, index=X.columns).sort_values(ascending=False)
            print('  Mean Importance: ' + str(round(temp.mean(),3)))
            #new_vars = list(temp.index[0:10])
            new_vars = list(temp[temp>1.5*temp.mean()].index)  # Select the variables > 1.5 of th emean importance
            assert len(new_vars) > 0, 'Random Forest Regression failed to select any variables'
            c+=1
            X = X[new_vars]
            print('  Out of Bag R-sq: ' + str(round(rf.oob_score_,3)))
                          
        elif method == 'rfe':  # Recursive Feature Elimination w. Linear Regression
            # Credit: https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
            low_score=1*10**6
            nof=0           
            for n in range(1,10):
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
                lm = LinearRegression(normalize=True)
                rfe = RFE(lm)
                rfe.n_features_to_select=n
                X_train_rfe = rfe.fit_transform(X_tr,y_tr)
                X_test_rfe = rfe.transform(X_te)
                lm.fit(X_train_rfe,y_tr)
                score = ((((lm.predict(X_test_rfe) - y_te)**2).sum())**0.5) / len(y_te)  # RMSE
                #score = lm.score(X_test_rfe,y_te)  # R2 of prediction
                #print(str(score))
                if(score<low_score):
                    low_score = score
                    nof = n
            print("Optimum number of features: %d" %nof)
            print("R2 with %d features: %f" % (nof, low_score))
            
            lm = LinearRegression(normalize=True)
            rfe = RFE(lm)
            rfe.n_features_to_select=n
            S = rfe.fit(X, y)
            new_vars = list(X.columns[list(np.where(S.support_)[0])])
            X = X[new_vars]
             
        # Check VIF
        X_multi = multicollinearity_check(X)
        if len(X_multi.columns) == len(X.columns):
            multi = False
        else:
            X = X_multi
    
    print('\nFinal Variables Selected - ' + str(len(X.columns)))
    print(X.columns.values)    
    return X

def check_corr(dep, ind, thresh=.95):
    '''
    Check if confounding variables have correlations > thresh, and drop the one with 
    least correlation to the dependnet variable
    
    Parameters:
        - dep - Pandas Series of the dependant variable
        
        - ind - Dataset (Pandas DF) containing modeling variables to be checked against the dependent
          variables
        
        - thresh - Threshold for Pearson Correlation Coefficient between two variables
          above which the least correlated variable to FIB will be dropped
          
    Output:
        - DataFrame with the best correlated variables included
    
    '''
    print('\nChecking variable correlations against threshold (PCC > ' + str(thresh) + '): ')
    c = ind.corr()  # Pearson correlation coefs.
    to_drop = []

    for ii in c.columns:  # iterate through all variables in correlation matrix except dependant variable
        temp = c.loc[ii]
        temp = temp[temp.abs() > thresh]  # .5 removed a lot of variables
        temp = temp.drop(ii, errors='ignore')  # Remove the variable itself
        i_corr = dep.corr(ind[ii])
        if len(temp) > 0:
            for j in temp.index:
                j_corr = dep.corr(ind[j])
                if ii not in to_drop and abs(i_corr) < abs(j_corr):  # Drop variable if its corr. with logFIB is lower
                    to_drop.append(ii)

    print('  Variables dropped - ' + str(len(to_drop)))
    print(to_drop)
    ind = ind.drop(to_drop, axis=1, errors='ignore')  # Drop variables
    print('Remaining variables - ' + str(len(ind.columns) - 1))
    print(ind.columns.values)
    return ind

def multicollinearity_check(X, thr=5):  
    '''
    Check VIF of model variables, drop if any above threshold
    
    Parameters:
        - X = Variable dataset
        
        - thr = threshold VIF maximum
        
    Output:
        - Dataset with no multicolinear variables
        
    '''
    variables = list(X.columns)
    print('\nChecking multicollinearity of ' + str(len(variables)) + ' variables for VIF:')
    if len(variables) > 1:
        vif_model = LinearRegression()
        v = [1 / (1 - (r2_score(X[ix], vif_model.fit(X[variables].drop(ix, axis=1), X[ix]).
                                predict(X[variables].drop(ix, axis=1))))) for ix in variables]
        maxloc = v.index(max(v))  # Drop max VIF var if above 'thr'
        if max(v) > thr:
            print(' Dropped: ' + X[variables].columns[maxloc] + ' (VIF - ' + str(round(max(v), 3)) + ')')
            variables.pop(maxloc)  # remove variable with maximum VIF
        else:
            print(' VIFs for all variables less than ' + str(thr))
        X = X[[i for i in variables]]
        return X
    else:
        return X

def fit_lm(X, y, score_metric, output_bin=True, seed=0, select_vars='all', multi=False, cv=5):  
    '''Fits Regression model after selecting variables
    X - calibration independent data; 
    y - calibration dependant variable;
    output_bin - is y a binary output? (or continuous)
    C - model regularization coefficient 
        - float: smaller - more regularization of variables);
        - If integer, test grid of size C to find optimal regulatization
    select_vars - variable selection method
        - ' all' - use all variables in X_train
        - 'force' - use only keep_vars to model with
        - else, perform feature selection
    cv - number of cross-validation steps
    scorer - model evaluation metric (roc_auc, accuracy)
    
    '''
    
    if output_bin:
        #score_metric='f1' # accuracy, f1, recall
        reg_param = 'C'
        lm = LogisticRegression(C = 1, 
                                penalty = 'elasticnet', # l1, l2, elasticnet, None 
                                l1_ratio = 0.5,
                                class_weight = 'balanced', # None, balanced
                                solver = 'saga',
                                max_iter = 1000,
                                random_state=seed)
        
    else:
        reg_param = 'alpha'
        #scorer='neg_root_mean_squared_error'  # r2, 
        lm = ElasticNet(alpha = 0.1, 
                        l1_ratio = 0.5,
                        random_state=seed)

    if select_vars in ['all','force']:
        features = X.columns
        
    else:
        ## Permutation method (narrow down big variable amounts)
        print('\nNarrowing down variables w/ permutation importances (scoring: ' + score_metric + ')')
        lm.fit(X,y)
        temp = permutation_importance(lm, X, y, 
                                      scoring=score_metric, 
                                      random_state=seed,
                                      n_repeats=cv)['importances_mean']
        temp = pd.Series(data=temp, index=X.columns).sort_values(ascending=False)
        #features = list(temp.index[0:10])
        features = list(temp[temp > 1.5*temp.mean()].index)  # Select the variables > X times the mean importance
        assert len(features) > 0, 'Linear Regression failed to select any variables'
        if len(features) < 3:
            features = list(temp.index[0:3])
        
        print('  ' + str(len(features)) + ' features selected')
        print(*features)
        X = X[features]
        

        c=0
        print('\nRFECV Variable Selection (scoring: ' + score_metric + ')')
        while c<1:
        
            ## Recursive Feature Elimination - Stepwise variable selection
            S = RFECV(lm, 
                      cv=cv, 
                      scoring=score_metric,
                      min_features_to_select=3,
                      verbose=0).fit(np.array(X), np.array(y))
            
            if multi:  # Check multicolinearity 
                old_len = len(features)    
                X = multicollinearity_check(X[features], thr=5)
                features = X.columns
                if len(features)<old_len:
                    c-=1
            c+=1   
            
        features = X.columns[list(np.where(S.support_)[0])]
        print('  ' + str(len(features)) + ' feature(s) selected')
        print(*features)
        
    ### Fit Model
    
    ## Grid Search for best parameters
    print('\nGrid Search for best model parameters (scoring: ' + score_metric + ')')
    gs = GridSearchCV(lm, 
                          param_grid={reg_param:[0.0001, 0.001, 0.01, 0.1, 1, 10],
                                      'l1_ratio': [0, 0.1, 0.25, .5, .75, 0.9, 1]},
                          cv=cv, 
                          scoring = score_metric,
                          verbose = 1)
    
    gs.fit(X[features], y)
    lm = gs.best_estimator_
    print('\n')
    print(lm)
    
    #lm.fit(X[features], y) # Fit model
    
    return list(features), lm


def fit_rf(X, y, score_metric, output_bin=True, n_trees=300, max_depth=3, max_features=.75, max_samples=.75, min_samples_leaf=3, seed=0, select_vars='all', cv=5):  
    '''Fits Random Forest model
    
    X - calibration independent data; 
    y - calibration dependant variable;
    select_vars - variable selection method
        - ' all' - use all variables in X_train
        - 'force' - use only keep_vars to model with
        - else, perform feature selection
    '''
    if output_bin:
        rf = RandomForestClassifier(n_estimators=n_trees, 
                                    oob_score=True,
                                    max_depth=max_depth,  # None - expands until all leaves are pure
                                    max_features=max_features,
                                    max_samples=max_features,
                                    min_samples_leaf=min_samples_leaf,
                                    class_weight='balanced', # None, 'balanced'
                                    random_state=seed,
                                    n_jobs=-1)
        #score_metric = 'recall' # accuracy, recall, f1, roc_auc
        
    else:
        rf = RandomForestRegressor(n_estimators=n_trees, 
                                   oob_score=True,
                                   max_depth=max_depth,  # None - expands until all leaves are pure
                                   max_features=max_features,
                                   max_samples=max_features,
                                   min_samples_leaf=min_samples_leaf,
                                   random_state=seed)
        
        #score_metric = 'neg_root_mean_squared_error' # r2, max_error, neg_mean_absolute_error, neg_root_mean_squared_error
        
    if select_vars in ['all','force']:
        features = X.columns
        
    else:  # Use variable selection method
    
        ## Random Forest Permutation method (narrow down big variable amounts)
        print('\nNarrowing down variables w/ permutation importances (scoring: ' + score_metric + ')')
        rf.fit(X,y)
        temp = permutation_importance(rf, X, y, 
                                      scoring=score_metric, 
                                      random_state=seed,
                                      n_repeats=cv)['importances_mean']
        temp = pd.Series(data=temp, index=X.columns).sort_values(ascending=False)
        #features = list(temp.index[0:10])
        features = list(temp[temp > 1.5*temp.mean()].index)  # Select the variables > X times the mean importance
        #assert len(features) > 0, 'Random Forest Regression failed to select any variables'
        if len(features) < 3:
            features = list(temp.index[0:10])
        
        print('  ' + str(len(features)) + ' features selected')
        print(*features)
        X = X[features]
        
        ## Recursive Feature Elimination - Stepwise variable selection
        print('\nRFECV Variable Selection (scoring: ' + score_metric + ')')
        S = RFECV(rf, 
                  cv=cv, 
                  scoring=score_metric,
                  min_features_to_select=3,
                  verbose=0).fit(np.array(X), np.array(y))
        
        features = X.columns[list(np.where(S.support_)[0])]
        print('\n' + str(len(features)) + ' feature(s) selected')
        print(*features)
        
    ### Fit Model
    
    ## Grid Search for best parameters
    print('\nGrid Search for best model parameters (scoring: ' + score_metric + ')')
    gs = GridSearchCV(rf, 
                          param_grid={'max_depth':[3,5],
                                      'max_features': [0.25, .5],
                                      #'max_samples': [.5,.75],
                                      'min_samples_leaf':[1, 3, 5]},
                          cv=cv, 
                          scoring = score_metric,
                          verbose = 1)
    
    gs.fit(X[features], y)
    rf = gs.best_estimator_
    print('\n')
    print(rf)
    
    # rf.fit(X[features], y) # Fit Model (simple)

    return list(features), rf 

def fit_ann(X, y, score_metric, output_bin=True, seed=0, select_vars='all', cv=5):  
    '''Fits Neural Network model
    
    X - calibration independent data; 
    y - calibration dependant variable;
    select_vars - variable selection method
        - ' all' - use all variables in X_train
        - 'force' - use only keep_vars to model with
        - else, perform feature selection
    '''
    if output_bin:
        ann = MLPClassifier(hidden_layer_sizes = (8,), 
                            activation='relu',  #tanh, logistic
                            solver='adam',   # adam, sgd, lbfgs
                            alpha=0.0001,
                            #learning_rate_init=0.1,
                            early_stopping=True,
                            max_iter=500,
                            random_state=0)
    else:
        ann = MLPRegressor(hidden_layer_sizes = (8,), 
                            activation='relu',  #tanh, logistic
                            solver='adam',   # adam, sgd, lbfgs
                            alpha=0.0001,
                            early_stopping=True,
                            #learning_rate_init=0.1,
                            max_iter=500,
                            random_state=0)
        
        #score_metric = 'neg_root_mean_squared_error' # r2, max_error, neg_mean_absolute_error, neg_root_mean_squared_error
        
    if select_vars in ['all','force']:
        features = X.columns
        
    else:  # Use variable selection method
    
        ## Random Forest Permutation method (narrow down big variable amounts)
        print('\nNarrowing down variables w/ permutation importances (scoring: ' + score_metric + ')')
        ann.fit(X,y)
        temp = permutation_importance(ann, X, y, 
                                      scoring=score_metric, 
                                      random_state=seed,
                                      n_repeats=cv)['importances_mean']
        temp = pd.Series(data=temp, index=X.columns).sort_values(ascending=False)
        #features = list(temp.index[0:10])
        features = list(temp[temp > 1.5*temp.mean()].index)  # Select the variables > X times the mean importance
        #assert len(features) > 0, 'Random Forest Regression failed to select any variables'
        if len(features) < 3:
            features = list(temp.index[0:10])
        
        print('  ' + str(len(features)) + ' features selected')
        print(*features)
        X = X[features]
        
        
        # ## Recursive Feature Elimination - Stepwise variable selection
        # print('\nRFECV Variable Selection (scoring: ' + score_metric + ')')
        # S = RFECV(ann, 
        #           cv=cv, 
        #           scoring=score_metric,
        #           min_features_to_select=3,
        #           verbose=0).fit(np.array(X), np.array(y))
        
        # features = X.columns[list(np.where(S.support_)[0])]
        # print('\n' + str(len(features)) + ' feature(s) selected')
        # print(*features)
        
    ### Fit Model
    
    ## Grid Search for best parameters
    print('\nGrid Search for best model parameters (scoring: ' + score_metric + ')')
    
    layers = [(2*len(X.columns),), 
              (4, (2*len(X.columns),)), 
              (4,8,4,),
              (16,32,),
              (8,8,3,)]
    
    gs = GridSearchCV(ann, 
                          param_grid={
                                      'hidden_layer_sizes':layers,
                                      'activation':['relu','logistic','tanh'],
                                      'alpha': [0.00001, 0.0001, 0.001],
                                      },
                          cv=cv, 
                          scoring = score_metric,
                          verbose = 1)
    
    gs.fit(X[features], y)
    ann = gs.best_estimator_
    print('\n')
    print(ann)

    return list(features), ann 

def model_tuner(y, y_pred, maximize='sens', min_sens=0.3, min_spec=0.5):  
    fpr, tpr, thresholds = roc_curve(y, y_pred)  # need probability predictions
    spec = np.round(1 - fpr, 3)
    sens = np.round(tpr, 3)
    
    df_tune = pd.DataFrame(data=[thresholds,sens,spec], index=['thresh','sens','spec']).T
        
    if maximize == 'sens':
        df_tune = df_tune[(df_tune.spec >= min_spec)]
        assert len(df_tune) > 0, 'No threshold available to tune spec >= ' + min_spec
        thresh = df_tune.iloc[-1]['thresh']
        
    elif maximize == 'spec':
        df_tune = df_tune[(df_tune.sens >= min_sens)]
        assert len(df_tune) > 0, 'No threshold available to tune sens >= ' + min_sens
        thresh = df_tune.iloc[0]['thresh']
    
    return thresh

def model_eval(true, predicted, thresh=0.5, output_bin=True):  # Evaluate Model Performance
    # if true.dtype == 'float':
        
    if not output_bin:
        r2 = r2_score(true, predicted)
        rmse = np.sqrt(((true - predicted)**2).mean())

    samples = len(true)  # number of samples
    exc = (true>thresh).sum()  # number of exceedances
    
    if exc == 0:
        auc = np.nan
    else:
        auc = roc_auc_score(true, predicted)
        
    true = (true > thresh).astype(int)  # Convert to binary if predicted.dtype == 'float':
    predicted = (predicted >= thresh).astype(int)

    cm = confusion_matrix(true, predicted)   # Lists number of true positives, true negatives,false pos,and false negs.
    if cm.size == 1: ## No exc observed or predicted
        sens = np.nan
        spec = 1.0
        acc = 1.0
        bal_acc = 1.0
    else: 
        sens = cm[1, 1] / (cm[1, 1] + cm[1, 0])  # sensitivity - TP / TP + FN
        spec = cm[0, 0] / (cm[0, 1] + cm[0, 0])  # specificity - TN / TN + FP
        acc = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + cm[0, 1] + cm[1, 0])
        bal_acc = balanced_accuracy_score(true, predicted) # balanced acc -> deals with class imbalance


    if output_bin:
        out = {'sens': round(sens, 3), 'spec': round(spec, 3), 'acc': round(acc, 3),
               'bal_acc': round(bal_acc, 3), 'AUC': round(auc, 3), 'N': samples, 'exc': exc}
    else:
        out = {'sens': round(sens, 3), 'spec': round(spec, 3), 'acc': round(acc, 3),
               'R2': round(r2, 3), 'RMSE': round(rmse, 3), 'AUC': round(auc, 3),
               'N': samples, 'exc': exc}
 
    return out

#%% Inputs
folder = '/Volumes/GoogleDrive/My Drive/high_frequency_wq/harbor_study_2022/data/'

f = 'ENT'
output_bin = True

miss_allow = .15
keep_vars = []

model_types = ['LM','RF'] #['LM','RF','ANN']
scale = True

hf_test = True  # split HF data into train and test subsets
hf_test_frac = 0.25

tune = True

save = False

### Constants
FIB = {'TC':10000,
       'FC': 400,
       'ENT': 104
       }
LOQ = 10

#%% Load Data

### HF Sampling Data
df = pd.read_csv(os.path.join(folder, 'PP7_variables.csv'), 
                 index_col=['dt'], parse_dates=['dt'])


print('HF data loaded. ' + str(len(df)) + ' obs. ' + str(len(df.columns)) + 
      ' vars. Vars with missing values: ' + str((df.isna().sum() > 0).sum()))

### Create lag variables
evs = ['tide','tide_high', 'wtemp','chl','turb','sal',
       'rad', 'awind','owind', 'pres','temp','dtemp']

df = lag_vars(df, var_list = evs, n_lags = 6, interval=30)

df = df.fillna(method='backfill', limit=3)  # only for EVs

for f in FIB:
    for i in range(1,5):
        df['log'+f + '_' + str(i*30)+'min'] = df['log'+f].shift(i)
        df[f+'_exc_'+ str(i*30)+'min'] = df[f+'_exc'].shift(i)
        df[f+'_BLOQ_'+ str(i*30)+'min'] = df[f+'_BLOQ'].shift(i)

df.dropna(inplace=True)

print('Vars created. ' + str(len(df)) + ' obs. ' + str(len(df.columns)) + 
      ' vars. Vars with missing values: ' + str((df.isna().sum() > 0).sum()))


### Load Test Datasets

## Hindcast
hind = pd.read_csv(os.path.join(folder, 'hindcast', 'hindcast_dataset.csv'), 
                   index_col=['dt'], parse_dates=['dt'])
print('\nHindcast data loaded. ' + str(len(hind)) + ' obs. ' + str(len(hind.columns)) + 
      ' vars.')
print(str(hind.index[0]) + ' - ' + str(hind.index[-1]))


## TODO: Nowcast


#%% Pre-Process Data
if output_bin:
    dep_var = f + '_exc'
    exc_thresh = 0.5
else:
    dep_var = 'log' + f  
    exc_thresh = np.log10(FIB[f])
    

### Non-Applicable Variables
drop_list = []

## Default
drop_default = [v for v in df.columns if any(x in v for x in ['sample_time', 'shift','cloud','day','sal','chl','turb'])] + \
             [v for v in df.columns if any(x in v for x in ['rain','flow','chl','turb']) and (('log' not in v))] 
drop_list += drop_default
 
## Drop other FIB variables
# We do not consider other FIB in the models
other_fib = [x for x in FIB if x != f]
drop_other_fib = []
for x in other_fib:
    drop_other_fib += [v for v in df.columns if x in v]
drop_list += drop_other_fib
    
## Drop same FIB variables that do not apply
drop_same_fib = [v for v in df.columns if (f in v) and (v != dep_var) and (v not in keep_vars)] # Drop non-transformed fib
drop_list += drop_same_fib
# varies depending on lead_time

## Drop from df
print('\nDropped ' + str(len(np.unique(drop_list))) + ' irrelevant features')
df = df[[c for c in df if c not in drop_list]]


### Remove 0 variance variables
vt = VarianceThreshold().fit(df)
if (vt.variances_ == 0).sum() > 0:
    print('Dropped ' + str((vt.variances_ == 0).sum()) + ' zero variance features:')
    print(list(df.columns[(vt.variances_ == 0)]))
df = pd.DataFrame(data= vt.transform(df), 
                  index=df.index, 
                  columns = df.columns[vt.get_support(indices=True)])

### Deal w Missing Data
df = df.dropna(subset=[dep_var])  # drop all rows where FIB == NaN

## Drop columns with high missing %
miss_frac = df.isnull().sum() / len(df)
drop_missing = miss_frac[(miss_frac > miss_allow)].index
drop_missing = [c for c in drop_missing if (c not in keep_vars) and (c != dep_var)]

df = df.drop(drop_missing, axis=1)
print('\nDropped ' + str(len(drop_missing)) + ' features with data > ' + str(100*miss_allow) + '% missing\n')

## Drop rows (days) with missingness in vars in keep_vars
len_temp = len(df)
df = df.dropna(axis=0)
print('Dropped ' + str(len_temp - len(df)) + ' rows; ' + str(len(df)) + ' rows remaining')
print(str(len(df.columns)) + ' variables remaining:')
print(*df.columns, sep=', ')

### Split into y and X
if hf_test:
    y_temp = df[dep_var]
    X_temp = df[[c for c in df if c != dep_var]]
    
    X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, 
                                                        test_size=hf_test_frac, 
                                                        shuffle=True,
                                                        random_state=0)
    
else:
    y_train = df[dep_var]
    X_train = df[[c for c in df if c != dep_var]]
    y_test = []
    X_train = []
    

#%% Variable Selection (Prior to Model Fitting)
# ''' Per Searcy and Boehm 2021, select variables using RF first, then fit other
# models'''
# var_select_method = 'forest'
# interact_var = ['rad', 'hours_from_noon', 'tide','tide_high', 'awind']
# # skip interaction variables for now, check correlations,

# X_train = select_vars(y_train, X_train, method=var_select_method, interaction=False, interact_var=interact_var)
# features = list(X_train.columns)


#%% Train and Test Models
if output_bin:
    score_metric = 'balanced_accuracy'
else:
    score_metric = 'neg_root_mean_squared_error'

print('\nModeling: ')

### Scale Variables
if scale:
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(data=scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(data=scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

### Train
for m in model_types:
    ### Regression
    if m == 'LM':
        print('\n\n* * Regression  * * ')
        features, model = fit_lm(X_train, y_train, score_metric, output_bin=output_bin, select_vars='select')
        
        if output_bin:
            #print('\nBinary Logistic Regression:')
            print(str(round(model.intercept_[0], 2)) + ' - intercept')
            for x in range(0,len(model.coef_[0])):
                print(str(round(model.coef_[0][x], 4)) + ' - ' + features[x])
            
        else:
            #print('\nMultiple Linear Regression:')
            print(str(round(model.intercept_, 2)) + ' - intercept')
            for x in range(0,len(model.coef_)):
                print(str(round(model.coef_[x], 2)) + ' - ' + features[x])
            #print('\nSelected alpha: ' + str(model.alpha_))
    
    ### Random Forest
    elif m == 'RF':
        print('\n\n* * Random Forest * *')
        features, model = fit_rf(X_train, y_train, score_metric, output_bin=output_bin, select_vars='select')
        
        print('\nFeature Importances: ')
        for x in range(0,len(features)):
            print(str(round(model.feature_importances_[x], 2)) + ' - ' + features[x])
    
    ### Neural Network
    elif m == 'ANN':
        print('\n\n* * Neural Network * *')
        features, model = fit_ann(X_train, y_train, score_metric, output_bin=output_bin, select_vars='select')
     

### Model Perf in Training
    if output_bin:
        #model.fit(X_train, y_train)
        pred_train = model.predict_proba(X_train[features])[:,1]
        
    else:
        #if m in ['RF','ANN']:
            #print('\nSummary of Model Fit')
            #model.fit(X_train[features], y_train) 
        pred_train = model.predict(X_train[features])
            
        # elif m in ['LM']:
        #     #print(model.summary2())
        #     pred_train = model.predict(sm.add_constant(X_train[features]))        
    
    
    print('\nTraining:')
    print(model_eval(y_train, pred_train, thresh=exc_thresh, output_bin=output_bin))
    
### HF Test (Validation) Set
    if hf_test:
        if output_bin:
            pred_test = model.predict_proba(X_test[features])[:,1]
        else:
            pred_test = model.predict(X_test[features])
    
        print('Test (Validation):')
        print(model_eval(y_test, pred_test, thresh=exc_thresh, output_bin=output_bin))


### Tuning
    if tune:
        DTP = model_tuner(y_test, pred_test)
        print('\nTuning factor: ' + str(round(DTP,5)))
        print('Test (Validation) [TUNED]:')
        print(model_eval(y_test, pred_test, thresh=DTP, output_bin=output_bin))


### Save data and models
    # if save:  
    #     # Save model
    #     model_file = 'model_' + f + '_' + m + '_' + t + '.pkl'
    #     if m in ['RF','ANN']:
    #         joblib.dump(model, os.path.join(tc_folder, model_file))
    #         # use joblib.load to load this file in the model runs script 
    #     elif m in ['MLR','GLS']:
    #         model.save(os.path.join(tc_folder, model_file))



### Hindcast Test Set
    start_yr = '2020'
    end_yr = '2021'
    
    hc = hind[start_yr:end_yr].copy()
    #hc = hc.merge(get_interaction(hc, interact_var), left_index=True, right_index=True)
    
    y_hc = hc[dep_var].copy()
    X_hc = hc.copy().drop([dep_var], axis=1)
    if scale:
        X_hc = X_hc[scaler.feature_names_in_]
        X_hc = pd.DataFrame(data=scaler.transform(X_hc), index=X_hc.index, columns=X_hc.columns) 
    X_hc = X_hc[features].dropna()
    y_hc = y_hc.reindex_like(X_hc)
    
        
    if output_bin:
        y_hc_persist = hc[f+'_ant_exc'].reindex_like(X_hc)
        pred_hc = model.predict_proba(X_hc[features])[:,1]
    else:
        y_hc_persist = hc[dep_var+'_ant'].reindex_like(X_hc)
        pred_hc = model.predict(X_hc)
        # if m not in ['LM']:
        #     pred_hc = model.predict(X_hc)
        # else:
        #     pred_hc = model.predict(sm.add_constant(X_hc))
    
    print('\n\nHindcast (' + str(start_yr) + '-' + str(end_yr) + '):')
    print(model_eval(y_hc, pred_hc, thresh=exc_thresh, output_bin=output_bin))
    if tune:
        print('Hindcast (TUNED):')
        print(model_eval(y_hc, pred_hc, thresh=DTP, output_bin=output_bin))    
    print('Persistence:')
    print(model_eval(y_hc, y_hc_persist, thresh=exc_thresh, output_bin=output_bin))


### Nowcast Test Set (August - October)




# #%% Explanatory Regressions
# f = 'ENT'
# EV = ['tide','chl','sal','wtemp','turb','owind','pres']
# ## Linear Regression
# #lm = smf.ols(formula, data=df_combo).fit()
# lm = sm.GLSAR(endog = df['log'+f], exog=sm.add_constant(df[EV]), 
#               rho=2, hasconst=True).iterative_fit(maxiter=10)

# ## Logistic Regression
# lm = sm.Logit(endog = df[f+'_exc'], exog=sm.add_constant(df[EV])).fit()

# print(lm.summary())
# print('N - ' + str(len(lm.predict())))

# if 'Binary' in str(type(lm)):
#     print('\nAIC: ' + str(round(lm.aic, 3)))
#     print('\nObserved/Predicted:')
#     pt = lm.pred_table()
#     print(pt)
#     print('acc: ' + str(round((pt[1,1] + pt[0,0]) / pt.sum(), 3)))
#     print('sens: ' + str(round(pt[1,1] / (pt[1,1] + pt[1,0]), 3)))
#     print('spec: ' + str(round(pt[0,0] / (pt[0,0] + pt[0,1]), 3)))

# else:
#     print('\nRMSE: ' + str(round(lm.mse_resid**.5, 3)))  # OLS
#     print('\nVIFs:')
#     print(lm.params.index)
#     variables = lm.model.exog
#     print([round(VIF(variables, i),3) for i in range(variables.shape[1])])
    

# ### Plot Regression Output
# #plot_regression(df_combo,lm)
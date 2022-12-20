#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 11:19:12 2022

@author: rtsearcy

Try applying a LSTM model to the data

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, roc_curve, r2_score, roc_auc_score, balanced_accuracy_score
#from datetime import datetime

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

#Read the csv file
folder = '/Volumes/GoogleDrive/My Drive/high_frequency_wq/harbor_study_2022/data/'
df = pd.read_csv(os.path.join(folder, 'PP7_variables.csv'),
                              parse_dates=['dt'],
                              index_col = ['dt'])

dep_var = 'ENT_exc'
df = df[[dep_var, 'sal', 'chl','tide','dtemp',
         'owind','tide_high', 'daytime', 'hours_from_noon']]

train = df.copy()
y_train = train.pop(dep_var)
features = train.columns

# scale data
scaler = StandardScaler()
scaler = scaler.fit(train)

## TODO: ADD PREVIOUS TIME STEPS
r = 6
for i in range(0,r):
    temp = train.copy()
    temp = temp.shift(i)
    temp = temp.fillna(-9999)
    temp = scaler.transform(temp)
    temp = np.array(temp).reshape(-1,1,len(features))
    if i == 0:
        X_train = temp.copy()
    else:
        X_train = np.append(X_train, temp, axis=1)
#X_train = X_train[r:,:,:]
#y_train = y_train.iloc[r:]


model = Sequential()
model.add(tf.keras.layers.Input(shape=(X_train.shape[1], len(features)))) # sjape: time steps, features
model.add(tf.keras.layers.Masking(mask_value=-9999))  
model.add(LSTM(64, activation='relu', return_sequences=False))
model.add(Dropout(0.4))
model.add(Dense(1))

model.compile(optimizer='adam', 
              loss='BinaryCrossentropy',
              metrics = ['Recall'])
model.summary()

# fit the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.3, verbose=1)

#evaluate
predictions = model.predict(X_train)
predictions = predictions.reshape(-1)
print(model_eval(y_train, predictions, thresh = 0.5, output_bin = True))


# # test data
# test = pd.read_csv('/Volumes/GoogleDrive/My Drive/high_frequency_wq/harbor_study_2022/data/hindcast/hindcast_dataset.csv',
#                    parse_dates=['dt'],
#                    index_col = ['dt'])
# test = test['2021':].dropna(subset=features)
# y_test = test[dep_var].copy()
# X_test = test[features].copy()
# X_test = scaler.transform(X_test)
# X_test = np.array(X_test).reshape(-1,1,len(features))

# test_pred = model.predict(X_test)
# test_pred = test_pred.reshape(-1)
# print(model_eval(y_test, test_pred, thresh = 0.5, output_bin = True))


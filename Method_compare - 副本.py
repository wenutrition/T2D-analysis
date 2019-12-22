# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:29:04 2019

@author: xihuyy8
"""


import pandas as pd 
import matplotlib
import numpy as np 
import sklearn 
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score 
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import seaborn as sns

##-----------------------------------------Random forest------------------------------------------

def modelfit(alg, feature, target, feature_names,performCV=True, printFeatureImportance=True, cv_folds=10):
    #Fit the algorithm on the data
    alg.fit(feature, target)
#
#    #Predict training set:
    feature_predictions = alg.predict(feature)

    #Perform cross-validation:
    if performCV:
       cv_score = cross_val_score(alg, feature, target, cv=cv_folds, scoring='roc_auc') #accuracy
       print ("error : %.4g" % metrics.mean_absolute_error(target, feature_predictions))
#       print ("AUC Score (Train): %f" % metrics.roc_auc_score(target, feature_predprob))
       print ("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))

#    Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, feature_names).sort_values(ascending=False)
        print(feat_imp)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.show()
    return feat_imp 

dff = pd.read_excel('D:/GNHS/data_pre/data_for_predict.xlsx') 
dff=dff.replace('.', np.nan)
dff =dff.convert_objects(convert_numeric=True) 

dff.index=dff['SampleID']

target_dia =dff.DM_outcome.astype('int')
feature_dia =dff.drop(columns='DM_outcome')
feature_dia =feature_dia.drop(columns='SampleID')

## train on NL

gbm0 =RandomForestClassifier(random_state=99)
feature=feature_dia
feature[feature==np.inf]=np.nan
feature.fillna(feature.mean(), inplace=True)
target=target_dia
feature_names=list(feature.columns.values)
micrfeat_im1=modelfit(gbm0, feature, target,feature_names)  ##min-0.785;mean-0.8445;max-0.8775
feaim_RF=pd.DataFrame(micrfeat_im1)
feaim_RF.to_excel('D:/GNHS/data_for_plot/RF_featureim.xlsx') 

## test on FH

df_fh = pd.read_excel('D:/GNHS/data_pre/F2FH_predict_basediet.xlsx') 
df_fh=df_fh.replace('.', np.nan)
df_fh =df_fh.convert_objects(convert_numeric=True) 
df_fh.index=df_fh['SampleID']
target_dia =df_fh.DM_outcome.astype('int')
feature_dia =df_fh.drop(columns='DM_outcome')
feature_dia =feature_dia.drop(columns='SampleID')
feature_dia[feature_dia==np.inf]=np.nan
feature_dia.fillna(feature_dia.mean(), inplace=True)
y_pred = gbm0.predict(feature_dia)
metrics.roc_auc_score(target_dia, y_pred)  ##0.5287

##----------------------------------------------LightGBM-------------------------------------------------------------

dff = pd.read_excel('D:/GNHS/data_pre/data_for_predict.xlsx') 
dff=dff.replace('.', np.nan)
dff =dff.convert_objects(convert_numeric=True) 

dff.index=dff['SampleID']

target_dia =dff.DM_outcome.astype('int')
feature_dia =dff.drop(columns='DM_outcome')
feature_dia =feature_dia.drop(columns='SampleID')

gbm1 = lgb.LGBMClassifier()
feature=feature_dia
target=target_dia
feature_names=list(feature.columns.values)
micrfeat_im=modelfit(gbm1, feature, target,feature_names) ##mean 0.9122 ,min 0.8502,max 0.975

df_fh = pd.read_excel('D:/GNHS/data_pre/F2FH_predict_basediet.xlsx') 
df_fh=df_fh.replace('.', np.nan)
df_fh =df_fh.convert_objects(convert_numeric=True) 
df_fh.index=df_fh['SampleID']
target_dia =df_fh.DM_outcome.astype('int')
feature_dia =df_fh.drop(columns='DM_outcome')
feature_dia =feature_dia.drop(columns='SampleID')
y_pred = gbm1.predict(feature_dia)
metrics.roc_auc_score(target_dia, y_pred)   ##0.71875



















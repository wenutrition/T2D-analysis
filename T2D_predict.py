# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 15:34:48 2019

@author: gouwanglong
"""

## import the needed library 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import shap
from sklearn import metrics 
shap.initjs()

###-------------------------------AUC compare (All features vs Selected features)-------------------------------------

##----------Discovery cohort + all features---------

#-Train the machine learning model

dff = pd.read_excel('D:/GNHS/data_pre/data_for_predict.xlsx') 
dff=dff.replace('.', np.nan)
dff =dff.convert_objects(convert_numeric=True) 

dff.index=dff['SampleID']

target_dia =dff.DM_outcome.astype('int')
feature_dia =dff.drop(columns='DM_outcome')
feature_dia =feature_dia.drop(columns='SampleID')

X_train, X_test, y_train, y_test = train_test_split(feature_dia, target_dia, test_size=0.4, random_state=7)

X_validate, X_test1, y_validate, y_test1 = train_test_split(X_test, y_test, test_size=0.5, random_state=7)

d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_validate, label=y_validate)


params = {'metric' : 'auc',
          'boosting_type' : 'gbdt',
          'colsample_bytree' : 0.9234,
          'num_leaves' : 13,
          'max_depth' : -1,
          'n_estimators' : 200,
          'min_child_samples': 399, 
          'min_child_weight': 0.1,
          'reg_alpha': 2,
          'reg_lambda': 5,
          'subsample': 0.855,
          'verbose' : -1,
          'num_threads' : 4
}

model = lgb.train(params, d_train, 10000, valid_sets=[d_test], early_stopping_rounds=50, verbose_eval=1000)
test_labels =y_test1
test_features = X_test1
y_pred = model.predict(test_features)
metrics.roc_auc_score(y_test1, y_pred)


#-Export the results for PROC analysis---

test_labels1 =y_validate
test_features1 = X_validate
y_pred1= model.predict(test_features1)

data_main_val=pd.DataFrame()
data_main_val['outcome_val_true']=test_labels1
data_main_val['outcome_val_predict']=y_pred1
data_main_val.to_excel('D:/GNHS/data_pre/AUC_calculate and compare/NL_all_features_validate.xlsx')


test_labels2 =y_test1
test_features2 = X_test1
y_pred2 = model.predict(test_features2)

data_main_test=pd.DataFrame()
data_main_test['outcome_val_true']=test_labels2
data_main_test['outcome_val_predict']=y_pred2

data_main_test.to_excel('D:/GNHS/data_pre/AUC_calculate and compare/NL_all_features_test.xlsx')

##-------Test on the external validation cohort 1---------

df_fh = pd.read_excel('D:/GNHS/data_pre/F2FH_predict_basediet.xlsx') 
df_fh=df_fh.replace('.', np.nan)
df_fh =df_fh.convert_objects(convert_numeric=True) 
df_fh.index=df_fh['SampleID']
target_dia =df_fh.DM_outcome.astype('int')
feature_dia_ =df_fh.drop(columns='DM_outcome')
feature_dia_ =feature_dia_.drop(columns='SampleID')
y_pred3 = model.predict(feature_dia_)

data_main_fh_test=pd.DataFrame()
data_main_fh_test['outcome_val_true']=target_dia
data_main_fh_test['outcome_val_predict']=y_pred3

data_main_fh_test.to_excel('D:/GNHS/data_pre/AUC_calculate and compare/FH_all_features_test.xlsx')

##-------------Model interpreting------------------------

#-Ranking the features impact-

X=feature_dia
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X,max_display=21)  ##just show the features with an average absolute SHAP value greater than 0

#-Selecting features 

shap_value=pd.DataFrame(shap_values)
shap_value.columns=X.columns
shap_value.index=X.index
shap_mean=shap_value.mean()
shap_index=[]
for i in shap_mean.index:
    if shap_mean[i]!=0:
        shap_index.append(i)     
shap_value=shap_value.loc[:,shap_index]        
shap_value['DM_outcome']=dff['DM_outcome']
df_select=dff.loc[:,shap_index]        
df_select['DM_outcome']=dff['DM_outcome']

##-------------------------Discovery cohort + Selected features------------------

#-Train model-

target_dia =df_select.DM_outcome.astype('int')
feature_dia =df_select.drop(columns='DM_outcome')
X_train, X_test, y_train, y_test = train_test_split(feature_dia, target_dia, test_size=0.4, random_state=7)

X_validate, X_test1, y_validate, y_test1 = train_test_split(X_test, y_test, test_size=0.5, random_state=7)

d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_validate, label=y_validate)

params = {'metric' : 'auc',
          'boosting_type' : 'gbdt',
          'colsample_bytree' : 0.9234,
          'num_leaves' : 13,
          'max_depth' : -1,
          'n_estimators' : 200,
          
          
          
          'min_child_samples': 399, 
          'min_child_weight': 0.1,
          'reg_alpha': 2,
          'reg_lambda': 5,
          'subsample': 0.855,
          'verbose' : -1,
          'num_threads' : 4
}

model = lgb.train(params, d_train, 10000, valid_sets=[d_test], early_stopping_rounds=50, verbose_eval=1000)
y_pred = model1.predict(X_test1)
metrics.roc_auc_score(y_test1, y_pred)

test_labels1 =y_validate
test_features1 = X_validate
y_pred1= model.predict(test_features1)

#--------Export the results for PROC analysis---------

data_main_val=pd.DataFrame()
data_main_val['outcome_val_true']=test_labels1
data_main_val['outcome_val_predict']=y_pred1
data_main_val.to_excel('D:/GNHS/data_pre/AUC_calculate and compare/NL_select_features_validate.xlsx')


test_labels2 =y_test1
test_features2 = X_test1
y_pred2 = model.predict(test_features2)

data_main_test=pd.DataFrame()
data_main_test['outcome_val_true']=test_labels2
data_main_test['outcome_val_predict']=y_pred2

data_main_test.to_excel('D:/GNHS/data_pre/AUC_calculate and compare/NL_select_features_test.xlsx')

##-----------------Test on external validation cohort 1----------

df_fh = pd.read_excel('D:/GNHS/data_pre/F2FH_predict_basediet_select.xlsx') 
df_fh=df_fh.replace('.', np.nan)
df_fh =df_fh.convert_objects(convert_numeric=True) 
df_fh.index=df_fh['SampleID']
target_dia =df_fh.DM_outcome.astype('int')
feature_dia =df_fh.drop(columns='DM_outcome')
feature_dia =feature_dia.drop(columns='SampleID')
y_pred = model.predict(feature_dia)
metrics.roc_auc_score(target_dia, y_pred)

#--------Export the results for PROC analysis---------

data_main_fh_test=pd.DataFrame()
data_main_fh_test['outcome_val_true']=target_dia
data_main_fh_test['outcome_val_predict']=y_pred

data_main_fh_test.to_excel('D:/GNHS/data_pre/AUC_calculate and compare/FH_select_features_test.xlsx')


##----------------------Compare selected genetic,microbiome,FORS+dietary+lifestyle factors AUC------------------------------------

#---------------------------Genetic factors--------------------------------

#-Train on discovery cohort--

dff = pd.read_stata("H:/data/GNHS/data_pre/select_gene.dta")
dff.index=dff['SampleID']

target_dia =dff.DM_outcome.astype('int')
feature_dia =dff.drop(columns='DM_outcome')
feature_dia =feature_dia.drop(columns='SampleID')

X_train, X_test, y_train, y_test = train_test_split(feature_dia, target_dia, test_size=0.4, random_state=7)

X_validate, X_test1, y_validate, y_test1 = train_test_split(X_test, y_test, test_size=0.5, random_state=7)

d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_validate, label=y_validate)

params = {'metric' : 'auc',
          'boosting_type' : 'gbdt',
          'colsample_bytree' : 0.9234,
          'num_leaves' : 13,
          'max_depth' : -1,
          'n_estimators' : 200,
          'min_child_samples': 399, 
          'min_child_weight': 0.1,
          'reg_alpha': 2,
          'reg_lambda': 5,
          'subsample': 0.855,
          'verbose' : -1,
          'num_threads' : 4
}

model = lgb.train(params, d_train, 10000, valid_sets=[d_test], early_stopping_rounds=50, verbose_eval=1000)
test_labels =y_test1
test_features = X_test1
y_pred = model.predict(test_features)
metrics.roc_auc_score(y_test1, y_pred)

test_labels1 =y_validate
test_features1 = X_validate
y_pred1= model.predict(test_features1)

data_main_val=pd.DataFrame()
data_main_val['outcome_val_true']=test_labels1
data_main_val['outcome_val_predict']=y_pred1
data_main_val.to_excel('D:/GNHS/data_pre/AUC_calculate and compare/select_gene_valid.xlsx')


test_labels2 =y_test1
test_features2 = X_test1
y_pred2 = model.predict(test_features2)

data_main_test=pd.DataFrame()
data_main_test['outcome_val_true']=test_labels2
data_main_test['outcome_val_predict']=y_pred2

data_main_test.to_excel('D:/GNHS/data_pre/AUC_calculate and compare/select_gene_test.xlsx')

#-Validating on external validation cohort 1

df_fh = pd.read_stata("D:/GNHS/raw data/FH_GRS_score.dta") 
df_fh=df_fh.replace('.', np.nan)
df_fh =df_fh.convert_objects(convert_numeric=True) 
df_fh.index=df_fh['SampleID']
target_dia =df_fh.DM_outcome.astype('int')
feature_dia =df_fh.drop(columns='DM_outcome')
feature_dia =feature_dia.drop(columns='SampleID')
y_pred = model.predict(feature_dia)
metrics.roc_auc_score(target_dia, y_pred)
data_main_val=pd.DataFrame()
data_main_val['outcome_val_true']=target_dia
data_main_val['outcome_val_predict']=y_pred
data_main_val.to_excel('D:/GNHS/data_pre/AUC_calculate and compare/FH_gene.xlsx')


#------------------Selected microbiome factors-------------------------

#-Train 

dff = pd.read_stata("H:/data/GNHS/data_pre/select_microbiota_MRS.dta")
dff.index=dff['SampleID']

target_dia =dff.DM_outcome.astype('int')
feature_dia =dff.drop(columns='DM_outcome')
feature_dia =feature_dia.drop(columns='SampleID')

X_train, X_test, y_train, y_test = train_test_split(feature_dia, target_dia, test_size=0.4, random_state=7)

X_validate, X_test1, y_validate, y_test1 = train_test_split(X_test, y_test, test_size=0.5, random_state=7)

d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_validate, label=y_validate)

params = {'metric' : 'auc',
          'boosting_type' : 'gbdt',
          'colsample_bytree' : 0.9234,
          'num_leaves' : 13,
          'max_depth' : -1,
          'n_estimators' : 200,
          'min_child_samples': 399, 
          'min_child_weight': 0.1,
          'reg_alpha': 2,
          'reg_lambda': 5,
          'subsample': 0.855,
          'verbose' : -1,
          'num_threads' : 4
}

model = lgb.train(params, d_train, 10000, valid_sets=[d_test], early_stopping_rounds=50, verbose_eval=1000)
test_labels =y_test1
test_features = X_test1
y_pred = model.predict(test_features)
metrics.roc_auc_score(y_test1, y_pred)

test_labels1 =y_validate
test_features1 = X_validate
y_pred1= model.predict(test_features1)

data_main_val=pd.DataFrame()
data_main_val['outcome_val_true']=test_labels1
data_main_val['outcome_val_predict']=y_pred1
data_main_val.to_excel('D:/GNHS/data_pre/AUC_calculate and compare/micro_val.xlsx')


test_labels2 =y_test1
test_features2 = X_test1
y_pred2 = model.predict(test_features2)

data_main_test=pd.DataFrame()
data_main_test['outcome_val_true']=test_labels2
data_main_test['outcome_val_predict']=y_pred2

data_main_test.to_excel('D:/GNHS/data_pre/AUC_calculate and compare/micro_test.xlsx')

#-External validation

df_fh = pd.read_stata("H:/data/GNHS/data_pre/fh_select_microbiota_MRS.dta") 
df_fh=df_fh.replace('.', np.nan)
df_fh =df_fh.convert_objects(convert_numeric=True) 
df_fh.index=df_fh['SampleID']
target_dia =df_fh.DM_outcome.astype('int')
feature_dia =df_fh.drop(columns='DM_outcome')
feature_dia =feature_dia.drop(columns='SampleID')
y_pred = model.predict(feature_dia)
metrics.roc_auc_score(target_dia, y_pred)

data_main_val=pd.DataFrame()
data_main_val['outcome_val_true']=target_dia
data_main_val['outcome_val_predict']=y_pred
data_main_val.to_excel('D:/GNHS/data_pre/AUC_calculate and compare/FH_micro_test.xlsx')

##---------------------- FORS+Lifestyle+Dietary factors-------------------------

#-Train model

dff = pd.read_stata("H:/data/GNHS/data_pre/traditional_factors_lifestyle.dta")
dff.index=dff['SampleID']

target_dia =dff.DM_outcome.astype('int')
feature_dia =dff.drop(columns='DM_outcome')
feature_dia =feature_dia.drop(columns='SampleID')

X_train, X_test, y_train, y_test = train_test_split(feature_dia, target_dia, test_size=0.4, random_state=7)

X_validate, X_test1, y_validate, y_test1 = train_test_split(X_test, y_test, test_size=0.5, random_state=7)

d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_validate, label=y_validate)

params = {'metric' : 'auc',
          'boosting_type' : 'gbdt',
          'colsample_bytree' : 0.9234,
          'num_leaves' : 13,
          'max_depth' : -1,
          'n_estimators' : 200,
          'min_child_samples': 399, 
          'min_child_weight': 0.1,
          'reg_alpha': 2,
          'reg_lambda': 5,
          'subsample': 0.855,
          'verbose' : -1,
          'num_threads' : 4
}

model = lgb.train(params, d_train, 10000, valid_sets=[d_test], early_stopping_rounds=50, verbose_eval=1000)
test_labels =y_test1
test_features = X_test1
y_pred = model.predict(test_features)
metrics.roc_auc_score(y_test1, y_pred)

test_labels1 =y_validate
test_features1 = X_validate
y_pred1= model.predict(test_features1)

data_main_val=pd.DataFrame()
data_main_val['outcome_val_true']=test_labels1
data_main_val['outcome_val_predict']=y_pred1
data_main_val.to_excel('D:/GNHS/data_pre/AUC_calculate and compare/select_traditional_lifestyle_val.xlsx')


test_labels2 =y_test1
test_features2 = X_test1
y_pred2 = model.predict(test_features2)

data_main_test=pd.DataFrame()
data_main_test['outcome_val_true']=test_labels2
data_main_test['outcome_val_predict']=y_pred2

data_main_test.to_excel('D:/GNHS/data_pre/AUC_calculate and compare/select_traditional_lifestyle_test.xlsx')

#-validate on external cohort 1---

df_fh = pd.read_stata("H:/data/GNHS/data_pre/FH_traditional_factors_lifestyle.dta") 
df_fh=df_fh.replace('.', np.nan)
df_fh =df_fh.convert_objects(convert_numeric=True) 
df_fh.index=df_fh['SampleID']
target_dia =df_fh.DM_outcome.astype('int')
feature_dia =df_fh.drop(columns='DM_outcome')
feature_dia =feature_dia.drop(columns='SampleID')
y_pred = model.predict(feature_dia)
metrics.roc_auc_score(target_dia, y_pred)

data_main_val=pd.DataFrame()
data_main_val['outcome_val_true']=target_dia
data_main_val['outcome_val_predict']=y_pred
data_main_val.to_excel('D:/GNHS/data_pre/AUC_calculate and compare/FH_traditional_factors_lifestyle_test.xlsx')

##-----------------------Selected microbiome+FORS+Lifestyle+Dietary factors------------------------

#-Train model

dff = pd.read_stata("H:/data/GNHS/data_pre/select_microbiota_traditionalfactors_lifestyle.dta")
dff.index=dff['SampleID']

target_dia =dff.DM_outcome.astype('int')
feature_dia =dff.drop(columns='DM_outcome')
feature_dia =feature_dia.drop(columns='SampleID')

X_train, X_test, y_train, y_test = train_test_split(feature_dia, target_dia, test_size=0.4, random_state=7)

X_validate, X_test1, y_validate, y_test1 = train_test_split(X_test, y_test, test_size=0.5, random_state=7)

d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_validate, label=y_validate)

params = {'metric' : 'auc',
          'boosting_type' : 'gbdt',
          'colsample_bytree' : 0.9234,
          'num_leaves' : 13,
          'max_depth' : -1,
          'n_estimators' : 200,
          'min_child_samples': 399, 
          'min_child_weight': 0.1,
          'reg_alpha': 2,
          'reg_lambda': 5,
          'subsample': 0.855,
          'verbose' : -1,
          'num_threads' : 4
}

model = lgb.train(params, d_train, 10000, valid_sets=[d_test], early_stopping_rounds=50, verbose_eval=1000)
test_labels =y_test1
test_features = X_test1
y_pred = model.predict(test_features)
metrics.roc_auc_score(y_test1, y_pred)

test_labels1 =y_validate
test_features1 = X_validate
y_pred1= model.predict(test_features1)

data_main_val=pd.DataFrame()
data_main_val['outcome_val_true']=test_labels1
data_main_val['outcome_val_predict']=y_pred1
data_main_val.to_excel('D:/GNHS/data_pre/AUC_calculate and compare/select_traditional_lifestyle__micro_val.xlsx')


test_labels2 =y_test1
test_features2 = X_test1
y_pred2 = model.predict(test_features2)

data_main_test=pd.DataFrame()
data_main_test['outcome_val_true']=test_labels2
data_main_test['outcome_val_predict']=y_pred2

data_main_test.to_excel('D:/GNHS/data_pre/AUC_calculate and compare/select_traditional_lifestyle_micro_test.xlsx')

#-validate on external cohort 1

df_fh = pd.read_stata("H:/data/GNHS/data_pre/fh_select_microbiota_traditionalfactors_lifestyle.dta") 
df_fh=df_fh.replace('.', np.nan)
df_fh =df_fh.convert_objects(convert_numeric=True) 
df_fh.index=df_fh['SampleID']
target_dia =df_fh.DM_outcome.astype('int')
feature_dia =df_fh.drop(columns='DM_outcome')
feature_dia =feature_dia.drop(columns='SampleID')
y_pred = model.predict(feature_dia)
metrics.roc_auc_score(target_dia, y_pred)
data_main_val=pd.DataFrame()
data_main_val['outcome_val_true']=target_dia
data_main_val['outcome_val_predict']=y_pred
data_main_val.to_excel('D:/GNHS/data_pre/AUC_calculate and compare/FH_traditional_factors_lifestyle_micro_test.xlsx')

##-------------------------Calculate and visualize the correlation matrices for selected microbiome features---------------

#-Discovery cohort-

df = pd.read_stata("D:/GNHS/data_pre/NL_mrs_visual.dta")
sns.catplot(x="MRS", y="micro_score", hue="T2D",
            kind="violin", data=df);

d = pd.read_stata("D:/GNHS/data_pre/nl_micro.dta")

sns.set(font='Arial',font_scale =2) 
sns.set(style="white")

# Compute the correlation matrix

corr = d.corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize = (10, 4))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.6, center=0,
            square=True, linewidths=.8, cbar_kws={"shrink": .5})

f.savefig('H:/GNHS/final figures/NL_micro_micro_interaction.pdf', bbox_inches='tight')


corr.to_excel('D:/GNHS/tables/NL_micro_relation.xlsx')

#-External validation cohort 1-


d = pd.read_stata("D:/GNHS/data_pre/fh_micro.dta")

sns.set(font='Arial',font_scale = 2) 
sns.set(style="white")


# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize = (10, 4))
ax.set_xlabel('X Label',fontsize=34)

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.6, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
f.savefig('H:/GNHS/final figures/FH_micro_micro_interaction.pdf', bbox_inches='tight')

corr.to_excel('D:/GNHS/tables/FH_micro_relation.xlsx')
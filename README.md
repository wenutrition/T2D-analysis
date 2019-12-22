# T2D-analysis
Introduction
Analytical framework for the investigation of microbiome-disease relationship
# Copyright
   Copyright:     Prof. Ju-Sheng Zheng  
 Institution:     School of Life Sciences, Westlake University, Hangzhou, China
       Email:     zhengjusheng@westlake.edu.cn
# Author
     Author:      Wanglong Gou    
Last update:     2019-12-22   
      Email:     gouwanglong@westlake.edu.cn
# Environment     
      Python version: 3.7.3
      Stata  version: 15
      R      version: 3.5.3
      Main   packages: sklearn, shap, pandas, numpy,lightgbm
# 1. Scripts
# 1.1 python scripts
# 1.1.1 T2D_predict.py
  Function: 
  Using LightGBM to infer the relationship between incorporated features and T2D;
  Using SHapley Additive explanation(SHAP) to interpret the machine learning model results;
  Algorithm performance in different cohorts based on the selected microbiome features, host genetics, lifestyle and diet, T2D traditional   risk factors (FORS), and their combination; 
  Calculate and visualize the correlation matrices for selected microbiome features.
# 1.1.2 Method_compare.py
  Function: 
  Compared our model performance with that of a random forest algorithm, applying the same evaluation criteria (tenfold cross-validation     in the discovery cohort, independent validation in the external cohort 1)
# 1.2 stata scripts
# 1.2.1 data_pre.do
  Function: 
  Pipline of preprocessing  
 

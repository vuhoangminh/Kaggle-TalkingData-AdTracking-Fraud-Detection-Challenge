"""
A non-blending lightGBM model that incorporates portions and ideas from various public kernels
This kernel gives LB: 0.977 when the parameter 'debug' below is set to 0 but this implementation requires a machine with ~32 GB of memory
"""

import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
import os
import pickle
import psutil
import datetime

SEED = 1988
process = psutil.Process(os.getpid())

now = datetime.datetime.now()
if now.month<10:
    month_string = '0'+str(now.month)
else:
    month_string = str(now.month)
if now.day<10:
    day_string = '0'+str(now.day)
else:
    day_string = str(now.day)
yearmonthdate_string = str(now.year) + month_string + day_string

boosting_type = 'gbdt'
# boosting_type = 'dart'
frac = 0.7

frm=10
to=180000010

# frm=1000
# to=1001000



DATATYPE_LIST = {
    'ip'                : 'uint32',
    'app'               : 'uint16',
    'device'            : 'uint16',
    'os'                : 'uint16',
    'channel'           : 'uint16',
    'is_attributed'     : 'uint8',
    'click_id'          : 'uint32',
    'mobile'            : 'uint16',
    'mobile_app'        : 'uint16',
    'mobile_channel'    : 'uint16',
    'app_channel'       : 'uint16',
    'category'          : 'category',
    'epochtime'         : 'int64',
    'nextClick'         : 'int64',
    'nextClick_shift'   : 'float64',
    'min'               : 'uint8',
    'day'               : 'uint8',
    'hour'              : 'uint8',
    'ip_mobile_day_count_hour'                  : 'uint32',
    'ip_mobile_app_day_count_hour'              : 'uint32',
    'ip_mobile_channel_day_count_hour'          : 'uint32',
    'ip_app_channel_day_count_hour'             : 'uint32',
    'ip_mobile_app_channel_day_count_hour'      : 'uint32',
    'ip_mobile_day_var_hour'                    : 'float16',
    'ip_mobile_app_day_var_hour'                : 'float16',
    'ip_mobile_channel_day_var_hour'            : 'float16',
    'ip_app_channel_day_var_hour'               : 'float16',
    'ip_mobile_app_channel_day_var_hour'        : 'float16',
    'ip_mobile_day_std_hour'                    : 'float16',
    'ip_mobile_app_day_std_hour'                : 'float16',
    'ip_mobile_channel_day_std_hour'            : 'float16',
    'ip_app_channel_day_std_hour'               : 'float16',
    'ip_mobile_app_channel_day_std_hour'        : 'float16',
    'ip_mobile_day_cumcount_hour'               : 'uint32',
    'ip_mobile_app_day_cumcount_hour'           : 'uint32',
    'ip_mobile_channel_day_cumcount_hour'       : 'uint32',
    'ip_app_channel_day_cumcount_hour'          : 'uint32',
    'ip_mobile_app_channel_day_cumcount_hour'   : 'uint32',
    'ip_mobile_day_nunique_hour'                : 'uint32',
    'ip_mobile_app_day_nunique_hour'            : 'uint32',
    'ip_mobile_channel_day_nunique_hour'        : 'uint32',
    'ip_app_channel_day_nunique_hour'           : 'uint32',
    'ip_mobile_app_channel_day_nunique_hour'    : 'uint32'
    }

PREDICTORS = [
    'mobile', 'mobile_app', 'mobile_channel', 'app_channel',
    'nextClick', 'nextClick_shift', 'hour',
    'ip_mobile_day_var_hour', 'ip_mobile_app_day_var_hour', 'ip_mobile_channel_day_var_hour',      
    'ip_app_channel_day_var_hour', 'ip_mobile_app_channel_day_var_hour',  
    'ip_mobile_day_cumcount_hour', 'ip_mobile_channel_day_cumcount_hour',      
    'ip_app_channel_day_cumcount_hour',
    'ip_mobile_day_nunique_hour', 'ip_mobile_channel_day_nunique_hour',      
    'ip_mobile_app_channel_day_nunique_hour' 
    ]  

CATEGORICAL = [
    'ip', 'app', 'device', 'os', 'channel',     
    'mobile', 'mobile_app', 'mobile_channel', 'app_channel',
    'category', 'epochtime', 'min', 'day', 'hour'
    ]

TARGET = ['is_attributed']

debug=0
if not debug:
    print('=======================================================================')
    print('process on server...')
    print('=======================================================================')
else:
    print('=======================================================================')
    print('for testing only...')
    print('=======================================================================')


def print_memory(print_string=''):
    print('Total memory in use ' + print_string + ': ', process.memory_info().rss/(2**30), ' GB')


def get_predictors():
    predictors = PREDICTORS
    print('------------------------------------------------')
    print('predictors:')
    for feature in predictors:
        print (feature)
    print('number of features:', len(predictors))            
    return predictors 

def get_categorical(predictors):
    predictors = get_predictors()
    categorical = []
    for feature in predictors:
        if feature in CATEGORICAL:
            categorical.append(feature)
    print('------------------------------------------------')
    print('categorical:')
    for feature in categorical:
        print (feature)
    print('number of categorical features:', len(categorical))                        
    return categorical            

def drop_features(df):
    for feature in df.columns:
        if feature not in PREDICTORS and feature not in TARGET:
            df = df.drop([feature], axis=1)
    return df

def lgb_modelfit_nocv(params, train_df_array, train_df_labels, val_df_array, val_df_labels, predictors, target='target', objective='binary', metrics='auc',
                 feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': boosting_type,
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.2,
        # 'drop_rate': 0.2, 
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4,
        'verbose': 0,
        'metric':metrics
    }

    lgb_params.update(params)
    print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')
    print("preparing validation datasets")
    
    xgtrain = lgb.Dataset(train_df_array, label=train_df_labels,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(val_df_array, label=val_df_labels,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')                          

    evals_results = {}

    bst1 = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgtrain, xgvalid], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=feval)

    print("\nModel Report")
    print("bst1.best_iteration: ", bst1.best_iteration)
    print(metrics+":", evals_results['valid'][metrics][bst1.best_iteration-1])

    return (bst1,bst1.best_iteration)


def predict(modelname,subfilename,num_iteration):
    # load model to predict
    print('Load model to predict')
    bst = lgb.Booster(model_file=modelname)

    # can only predict with the best iteration (or the saving iteration)
    print('prepare predictors...')
    predictors = get_predictors()
  
    print('reading test')
    test_h5 = pd.HDFStore('test_day9.h5')
    print(test_h5)
    test_df = test_h5.select('test') 
    print(test_df.info()); print(test_df.head())
    print_memory()

    print("test size : ", len(test_df))
    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')

    print("Predicting...")
    sub['is_attributed'] = bst.predict(test_df[predictors],num_iteration=num_iteration)
    # if not debug:
    print("writing...")
    sub.to_csv(subfilename,index=False,compression='gzip')
    print("done...")
    


# num_leaves_list = [7,9,11,13,15,31,31,9]
# max_depth_list = [3,4,5,6,7,5,6,5]
num_leaves_list = [7]
max_depth_list = [3]
num_iteration_list = [245]

predictors = get_predictors()
print(predictors)


for i in range(len(num_leaves_list)):
    print ('==============================================================')
    # num_leaves = num_leaves_list[len(num_leaves_list)-1-i]
    # max_depth = max_depth_list[len(num_leaves_list)-1-i]
    num_leaves = num_leaves_list[i]
    max_depth = max_depth_list[i]
    num_iteration = num_iteration_list[i]
    print('num leaves:', num_leaves)
    print('max depth:', max_depth)
    print('iteration:', num_iteration)
    predictors = get_predictors()
    subfilename = yearmonthdate_string + '_' + str(len(predictors)) + \
            'features_' + boosting_type + '_minh_hope_' + str(int(100*frac)) + \
            'percent_day9_%d_%d.csv.gz'%(num_leaves,max_depth)
    modelfilename = yearmonthdate_string + '_' + str(len(predictors)) + \
            'features_' + boosting_type + '_minh_hope_' + str(int(100*frac)) + \
            'percent_day9_%d_%d.txt'%(num_leaves,max_depth)          
    predict(modelfilename,subfilename,num_iteration)   
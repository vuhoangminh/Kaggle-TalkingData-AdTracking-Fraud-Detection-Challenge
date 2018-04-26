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

process = psutil.Process(os.getpid())
# boosting_type = 'rf'
boosting_type = 'gbdt'
# boosting_type = 'dart'
frac = 0.01
noday = False

frm=0
to=1

debug=0
if debug:
    print('*** debug parameter set: this is a test run for debugging purposes ***')

def get_predictors():
    predictors=[]
    new_feature = 'nextClick'
    predictors.append(new_feature)
    predictors.append(new_feature + '_shift')
    predictors.extend(['app','device','os', 'channel', 'hour', 'day', 
                'ip_tcount', 'ip_tchan_count', 'ip_app_count',
                'ip_app_os_count', 'ip_app_os_var',
                'ip_app_channel_var_day','ip_app_channel_mean_hour',
                'mode_channel_ip', 'mode_device_ip', 'mode_os_ip', 'mode_app_ip',
                'numunique_device_ip','mode_channel_app','mode_device_app','mode_os_app',
                'mode_ip_app','mode_channel_device','mode_app_device','mode_os_device',
                'mode_ip_device','mode_channel_os','mode_app_os','mode_device_os',
                'mode_ip_os','numunique_device_os','mode_os_channel','mode_app_channel',
                'mode_device_channel','mode_ip_channel','numunique_device_channel',
                'numunique_app_channel','category','epochtime'])   
    naddfeat=9
    for i in range(0,naddfeat):
        predictors.append('X'+str(i))

    predictors_removed = ['mode_channel_ip', 'mode_device_ip', 'mode_os_ip', 'mode_app_ip',
                'numunique_device_ip','mode_channel_app','mode_device_app','mode_os_app',
                'mode_ip_app','mode_channel_device','mode_app_device','mode_os_device',
                'mode_ip_device','mode_channel_os','mode_app_os','mode_device_os',
                'mode_ip_os','numunique_device_os','mode_os_channel','mode_app_channel',
                'mode_device_channel','mode_ip_channel','numunique_device_channel',
                'numunique_app_channel','epochtime',
                'X3','X5','ip_tcount', 'ip_app_channel_mean_hour','day']                  

    predictors_kept = predictors
    for item in predictors_removed:
        while predictors_kept.count(item) > 0:
            predictors_kept.remove(item)

    return predictors_kept 


def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
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

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

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

def DO(frm,to,fileno,num_leaves,max_depth):
    print('------------------------------------------------')
    print('start...')
    print('fraction:', frac)
    
    print('prepare predictors...')
    predictors = get_predictors()
    print('predictors:',predictors)
    if 'day' in predictors:
        categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']
    else:
        categorical = ['app', 'device', 'os', 'channel', 'hour']       
    target = 'is_attributed'

    subfilename = yearmonthdate_string + '_' + str(len(predictors)) + 'features_' + boosting_type + '_sub_it_' + str(int(100*frac)) + 'percent_day9_%d_%d.csv'%(num_leaves,max_depth)
    modelfilename = yearmonthdate_string + '_' + str(len(predictors)) + 'features_' + boosting_type + '_sub_it_' + str(int(100*frac)) + 'percent_day9_%d_%d'%(num_leaves,max_depth)
    
    print('submission file name:', subfilename)
    print('model file name:', modelfilename)


    print('read val...')
    save_name='val_%d_%d'%(frm,to)
    val_df = pd.read_pickle(save_name)
    val_df = val_df.sample(frac=frac)
    print('Total memory in use after reading val: ', process.memory_info().rss/(2**30), ' GB\n')
    print('read train...')
    save_name='train_%d_%d'%(frm,to)
    train_df = pd.read_pickle(save_name)
    train_df = train_df.sample(frac=frac)
    print('Total memory in use after reading train: ', process.memory_info().rss/(2**30), ' GB\n')
    train_df = train_df.append(val_df)
    del val_df; gc.collect()
    print('Splitting train and val...')
    train_df, val_df = train_test_split(train_df, test_size=0.2,
                random_state=42)

   

    gc.collect()

    print('------------------------------------------------')
    print("Training...")
    start_time = time.time()

    params = {
        'learning_rate': 0.05,
        #'is_unbalance': 'true', # replaced with scale_pos_weight argument
        # 'num_leaves': 7,  # 2^max_depth - 1
        # 'max_depth': 3,  # -1 means no limit
        'num_leaves': num_leaves,  # 2^max_depth - 1
        'max_depth': max_depth,  # -1 means no limit
        # 'num_leaves': 9,  # 2^max_depth - 1
        # 'max_depth': 5,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight':200 # because training data is extremely unbalanced 
    }
    print (params)
    (bst,best_iteration) = lgb_modelfit_nocv(params, 
                            train_df, 
                            val_df, 
                            predictors, 
                            target, 
                            objective='binary', 
                            metrics='auc',
                            early_stopping_rounds=30, 
                            verbose_eval=True, 
                            num_boost_round=1000, 
                            categorical_features=categorical)

    print('[{}]: model training time'.format(time.time() - start_time))
    del train_df
    del val_df
    gc.collect()

    save_name='test_%d_%d'%(frm,to)
    test_df = pd.read_pickle(save_name)
    print('Total memory in use after reading test: ', process.memory_info().rss/(2**30), ' GB\n')
    print("test size : ", len(test_df))

    
    return 0

# num_leaves_list = [7,9,20,31,35]
# max_depth_list = [3,5,9,17,27]
num_leaves_list = [7,9,11,13,15]
max_depth_list = [3,4,5,6,7]

# num_leaves_list = [7,9,15,31,41]
# max_depth_list = [5,5,5,5,5]

# num_leaves_list = [35]
# max_depth_list = [27]

for i in range(len(num_leaves_list)):
    print ('==============================================================')
    # num_leaves = num_leaves_list[len(num_leaves_list)-1-i]
    # max_depth = max_depth_list[len(num_leaves_list)-1-i]
    num_leaves = num_leaves_list[i]
    max_depth = max_depth_list[i]
    print('num leaves:', num_leaves)
    print('max depth:', max_depth)
    predictors = get_predictors()
    subfilename = yearmonthdate_string + '_' + str(len(predictors)) + 'features_' + boosting_type + '_sub_it_' + str(int(100*frac)) + 'percent_day9_%d_%d.csv'%(num_leaves,max_depth)
    if os.path.isfile(subfilename):
        print('Already trained...')
    else:        
        sub=DO(frm,to,0,num_leaves,max_depth)
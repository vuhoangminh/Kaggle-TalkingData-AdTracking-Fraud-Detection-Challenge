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

process = psutil.Process(os.getpid())
# boosting_type = 'rf'
boosting_type = 'gbdt'
# boosting_type = 'dart'
frac = 0.8
# num_leaves = 31
# max_depth = 5
# num_leaves = 38
# max_depth = 21
noday = False

frm=10
to=180000010

# frm=1000
# to=1001000

debug=1
if debug:
    print('*** debug parameter set: this is a test run for debugging purposes ***')

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

def DO(frm,to,fileno,num_leaves,max_depth):

    if noday:
        subfilename = boosting_type + '_noday_sub_it_' + str(int(100*frac)) + 'percent_%d_%d_%d_%d.csv.gz'%(frm,to,num_leaves,max_depth)
        modelfilename = boosting_type + '_noday_sub_it_' + str(int(100*frac)) + 'percent_%d_%d_%d_%d'%(frm,to,num_leaves,max_depth)
    else:
        subfilename = boosting_type + '_sub_it_' + str(int(100*frac)) + 'percent_%d_%d_%d_%d.csv.gz'%(frm,to,num_leaves,max_depth)
        modelfilename = boosting_type + '_sub_it_' + str(int(100*frac)) + 'percent_%d_%d_%d_%d'%(frm,to,num_leaves,max_depth)
    
    print('submission file name:', subfilename)
    print('model file name:', modelfilename)
    print('fraction:', frac)

    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32',
            }


    print('doing nextClick')
    predictors=[]
    
    new_feature = 'nextClick'
    predictors.append(new_feature)
    predictors.append(new_feature+'_shift')
    
 
    target = 'is_attributed'
    if noday:
        predictors.extend(['app','device','os', 'channel', 'hour', 
                    'ip_tcount', 'ip_tchan_count', 'ip_app_count',
                    'ip_app_os_count', 'ip_app_os_var',
                    'ip_app_channel_var_day','ip_app_channel_mean_hour'])
        categorical = ['app', 'device', 'os', 'channel', 'hour']
    else:
        predictors.extend(['app','device','os', 'channel', 'hour', 'day', 
                    'ip_tcount', 'ip_tchan_count', 'ip_app_count',
                    'ip_app_os_count', 'ip_app_os_var',
                    'ip_app_channel_var_day','ip_app_channel_mean_hour'])
        categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']

    naddfeat=9
    for i in range(0,naddfeat):
        predictors.append('X'+str(i))
        
    print('predictors',predictors)

    
    if debug:
        save_name='train_1000_1001000'
    else:        
        save_name='train_%d_%d_reduced'%(frm,to)
    train_df = pd.read_pickle(save_name)
    train_df = train_df.sample(frac=frac)
    print('Total memory in use after reading train: ', process.memory_info().rss/(2**30), ' GB\n')
    
    if debug:
        save_name='val_1000_1001000'
    else:    
        save_name='val_%d_%d_reduced'%(frm,to)
    val_df = pd.read_pickle(save_name)
    val_df = val_df.sample(frac=frac)
    print('Total memory in use after reading val: ', process.memory_info().rss/(2**30), ' GB\n')
    
    


    print("train size: ", len(train_df))
    print("valid size: ", len(val_df))


    gc.collect()

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

    print('cleaning train...')
    train_df_array = train_df[predictors].values
    train_df_labels = train_df[target].values.astype('int').flatten()
    del train_df; gc.collect()
    print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')
    print('cleaning val...')
    val_df_array = val_df[predictors].values
    val_df_labels = val_df[target].values.astype('int').flatten()
    del val_df; gc.collect()
    print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')


    (bst,best_iteration) = lgb_modelfit_nocv(params, 
                            train_df_array, 
                            train_df_labels,
                            val_df_array, 
                            val_df_labels,
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

    print('Save model...')
    # save model to file
    bst.save_model(modelfilename+'.txt')


    save_name='test_%d_%d'%(frm,to)
    test_df = pd.read_pickle(save_name)
    print('Total memory in use after reading test: ', process.memory_info().rss/(2**30), ' GB\n')
    print("test size : ", len(test_df))
    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')

    print("Predicting...")
    sub['is_attributed'] = bst.predict(test_df[predictors],num_iteration=best_iteration)
    # if not debug:
    print("writing...")
    sub.to_csv(subfilename,index=False,compression='gzip')
    print("done...")
    return sub

num_leaves_list = [7,9,11,13,15]
max_depth_list = [3,4,5,6,7]
# num_leaves_list = [9]
# max_depth_list = [4]

for i in range(len(num_leaves_list)):
    print ('==============================================================')
    # num_leaves = num_leaves_list[len(num_leaves_list)-1-i]
    # max_depth = max_depth_list[len(num_leaves_list)-1-i]
    num_leaves = num_leaves_list[i]
    max_depth = max_depth_list[i]
    print('num leaves:', num_leaves)
    print('max depth:', max_depth)
    sub=DO(frm,to,0,num_leaves,max_depth)

# sub=DO(frm,to,0)
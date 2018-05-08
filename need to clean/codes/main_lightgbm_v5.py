"""
Adding improvements inspired from:
Ravi Teja's fe script: https://www.kaggle.com/rteja1113/lightgbm-with-count-features?scriptVersionId=2815638
"""

import pandas as pd
import time
import numpy as np
from numpy import random
from sklearn.cross_validation import train_test_split
import json
import lightgbm as lgb
import gc

def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                 feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        # 'learning_rate': 0.001,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 128,  # Number of bucketed bin for feature values
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
                     learning_rates=lambda iter: 0.3 * (0.997 ** iter),
                     feval=feval)

    n_estimators = bst1.best_iteration
    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics+":", evals_results['valid'][metrics][n_estimators-1])

    return bst1

# path = 'D:/Users/RD/Documents/GitHub/Kaggle-datasets/Kaggle-TalkingData-AdTracking-Fraud-Detection-Challenge/input/'
path = 'E:/kaggle/input/'

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

def train(iSplit):
    # 
    train_set_name = 'train_0'
    train_set_name2 = 'train_1'
    

    print (train_set_name)
    print (train_set_name2)
            
    print ("load train...")
    train_df_full = pd.read_pickle(train_set_name)
    train_df_full2 = pd.read_pickle(train_set_name2)
    # print(train_df_full.head())
    train_df_full=train_df_full.append(train_df_full2)
    del train_df_full2
    gc.collect()


    train_df_full, train_df_delete = train_test_split(train_df_full, test_size=0.2)

    del train_df_delete
    gc.collect()



    # split
    train_df, val_df = train_test_split(train_df_full, test_size=0.3)
    del train_df_full
    gc.collect()

    print("after splitted: ")
    # print(train_df.head())

    print("train size: ", len(train_df))
    print("valid size: ", len(val_df))
    

    target = 'is_attributed'
    # predictors = ['app','device','os', 'channel', 'min' ,'hour', 'day', 'wday', 'qty', 'ip_app_count', 'ip_app_os_count']
    # categorical = ['app', 'device', 'os', 'channel', 'min', 'hour', 'day', 'wday']
    predictors = ['app','device','os', 'channel','hour', 'qty', 'ip_app_count', 'ip_app_os_count']
    categorical = ['app', 'device', 'os', 'channel', 'hour']

    gc.collect()

    print("Training...")
    start_time = time.time()


    params = {
        # 'learning_rate': 0.15,
        #'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 15,  # 2^max_depth - 1
        'max_depth': 4,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': .7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight': 99, # because training data is extremely unbalanced 
        'bagging_fraction': 0.7, # randomly select part of data without resampling
        'bagging_freq': 1, # frequency for bagging
        'save_binary': True
    }

    # params = {}

    bst = lgb_modelfit_nocv(params, 
                            train_df, 
                            val_df, 
                            predictors, 
                            target, 
                            objective='binary', 
                            metrics='auc',
                            # metrics='binary_logloss',
                            early_stopping_rounds=50, 
                            verbose_eval=True, 
                            num_boost_round=700, 
                            categorical_features=categorical)

    print('[{}]: model training time'.format(time.time() - start_time))
    del train_df
    del val_df
    gc.collect()

    print ("load test...")
    test_df = pd.read_pickle('test')
    # print(test_df.head())
    print("test size : ", len(test_df))
    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')


    print('Save model...')
    # save model to file
    save_name = 'lightgbm_v5_split_' + str(iSplit)
    bst.save_model(save_name + '.txt')

    print("Predicting...")
    sub['is_attributed'] = bst.predict(test_df[predictors])
    print("writing...")
    sub.to_csv(save_name + '.csv',index=False)
    print("done...")

# train
for iSplit in range(3):
    print (iSplit)   
    train_set_name = 'train_' + str(iSplit)
    train(iSplit)
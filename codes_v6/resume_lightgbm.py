import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

debug=0
frac=1
print('debug', debug)
OPTION = 3


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
import h5py

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


frm = 1; to = 100

TARGET = ['is_attributed']

if debug==1:
    DATASET = 'day9'
else:    
    DATASET = 'full'

TRAIN_HDF5 = 'train_' + DATASET + '.h5'
TEST_HDF5 = 'test_' + DATASET + '.h5'
if debug == 0:
    TRAIN_HDF5 = 'converted_' + TRAIN_HDF5
    TEST_HDF5 = 'converted_' + TEST_HDF5


if OPTION == 1:
    # OPTION 1 - core
    PREDICTORS = [
        # core 9
        'ip', 'app', 'os', 'channel', 'hour',
        'mobile',
        'ip_os_device_app_nextclick',
        'ip_device_os_nunique_app',
        'ip_nunique_channel'
    ]        


if OPTION == 3:
    # OPTION 3 - PREVIOUS RESULT
    PREDICTORS = [
        # 'ip', 
        'app', 'device', 'os', 'channel', 'hour',
        'ip_nunique_channel',   # X0
        'ip_device_os_cumcount_app',
        'ip_day_nunique_hour',
        'ip_nunique_app',
        'ip_app_nunique_os',
        'ip_nunique_device',
        'app_nunique_channel',
        # 'ip_cumcount_os', # X6
        'ip_device_os_nunique_app', # X8
        'ip_os_device_app_nextclick',
        'ip_day_hour_count_channel',
        'ip_app_count_channel',
        'ip_app_os_count_channel',
        # 'ip_day_channel_var_hour', # miss
        'ip_app_os_var_hour',
        'ip_app_channel_var_day',
        'ip_app_channel_mean_hour'
        ]     

if OPTION == 4:
    # OPTION 4 - RFE
    PREDICTORS = ['ip', 'app', 'os', 'channel', 'hour',
        'ip_os_device_app_nextclick',
        'ip_day_hour_count_mobile_channel',
        'ip_day_count_mobile',
        'ip_day_hour_count_mobile_app',
        'ip_day_count_mobile_app',
        'ip_day_hour_count_mobile',
        'ip_day_count_hour',
        'ip_day_count_app',
        'ip_app_channel_var_day',
        'ip_mobile_day_std_hour',
        'ip_day_hour_count_channel',
        'ip_nunique_channel',
        'ip_device_os_nunique_app',
        'ip_app_nextclick'
        ]  

if OPTION == 5:
    # OPTION 4 - RFE
    PREDICTORS = ['ip', 'app', 'os', 'channel', 'hour',
        'device', 'mobile',
        'ip_os_device_app_nextclick',
        'ip_nextclick',
        'ip_app_nextclick',
        'ip_device_os_nextclick',
        'ip_channel_nextclick',
        'ip_os_device_app_nextclick',
        'ip_os_device_channel_app_nextclick',
        'ip_os_device_channel_nextclick',
        'ip_day_hour_var_channel',
        'ip_day_hour_nunique_channel',
        'ip_app_nunique_channel'
        ] 

if OPTION == 6:
    PREDICTORS = [
        # core 9
        'ip', 'app', 'os', 'channel', 'hour',
        'mobile',
        'ip_os_device_app_nextclick',
        'ip_device_os_nunique_app',
        'ip_nunique_channel',

        # add 10
        'ip_mobile_day_cumcount_hour',
        'ip_var_os',
        'ip_day_count_channel',
        'ip_app_channel_day_cumcount_hour',
        'ip_day_count_mobile_channel',
        'ip_mobile_day_count_hour',
        'ip_mobile_day_var_hour',
        'ip_mobile_day_nunique_hour',
        'ip_app_os_nunique_channel',
        'ip_mobile_channel_day_cumcount_hour'
        ] 

if OPTION == 7:
    PREDICTORS = [
        # core 9
        'ip', 'app', 'os', 'channel', 'hour',
        'mobile',
        'ip_os_device_app_nextclick',
        'ip_device_os_nunique_app',
        'ip_nunique_channel',

        # add 10
        'ip_mobile_app_channel_day_count_hour',
        'ip_mobile_channel_day_count_hour',
        'ip_mobile_app_day_count_hour',
        'ip_mobile_app_channel_day_cumcount_hour',
        'ip_app_channel_day_count_hour',
        'ip_mobile_app_day_cumcount_hour',
        'ip_app_channel_day_std_hour',
        'ip_app_channel_day_var_hour',
        'ip_app_channel_day_nunique_hour',
        'ip_app_os_var_hour',

        ]         



CATEGORICAL = [
    'ip', 'app', 'device', 'os', 'channel',     
    'mobile', 'mobile_app', 'mobile_channel', 'app_channel',
    'category', 'epochtime', 'min', 'day', 'hour'
    ]

TARGET = ['is_attributed']

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
    del train_df_array, train_df_labels, val_df_array, val_df_labels
    gc.collect()
    
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


def read_processed_h5(filename, predictors):
    with h5py.File(filename,'r') as hf:
        feature_list = list(hf.keys())
    train_df = pd.DataFrame()
    t0 = time.time()
    for feature in feature_list:
        if feature!='dump_later' and feature in predictors:
            print('>> adding', feature)
            if debug==2:
                train_df[feature] = pd.read_hdf(filename, key=feature, 
                        start=0, stop=100)        
            else:
                # where = [0].append(range(9308569, 10000000))
                train_df[feature] = pd.read_hdf(filename, key=feature)   
            if feature=='day' or feature=='hour' or feature=='min':
                train_df[feature] = train_df[feature].fillna(0)
                train_df[feature] = train_df[feature].astype('uint8')                                                              
            print_memory()
    t1 = time.time()
    total = t1-t0
    print('total reading time:', total)
    print(train_df.info())   
    return train_df


def DO(frm,to,fileno,num_leaves,max_depth):

    print('------------------------------------------------')
    print('start...')
    print('fraction:', frac)
    print('prepare predictors, categorical and target...')
    predictors = get_predictors()
    categorical = get_categorical(predictors)
    target = TARGET
    print('taget', target)


    if debug==0:
        print('=======================================================================')
        print('process on server...')
        print('=======================================================================')
    if debug==1:
        print('=======================================================================')
        print('for testing only...')
        print('=======================================================================')
    if debug==2:
        print('=======================================================================')
        print('for LIGHT TEST only...')
        print('=======================================================================')
        print('reading train')
    print(predictors)
    print('option:', OPTION)
    train_df = read_processed_h5(TRAIN_HDF5, predictors+target)

    train_df = train_df.sample(frac=frac, random_state = SEED)
    print_memory('afer reading train')
    # train_df = drop_features(train_df)
    # print_memory('afer drop unused features')
    train_df, val_df = train_test_split(train_df, test_size=0.33, random_state=SEED)
    print(train_df.head())
    print_memory('afer split')
    print("train size: ", len(train_df))
    print("valid size: ", len(val_df))

    gc.collect()

    subfilename = yearmonthdate_string + '_' + str(len(predictors)) + \
            'features_' + boosting_type + '_minh_hope_' + str(int(100*frac)) + \
            'percent_full_%d_%d'%(num_leaves,max_depth) + '_OPTION' + str(OPTION) + '.csv.gz'
    modelfilename = yearmonthdate_string + '_' + str(len(predictors)) + \
            'features_' + boosting_type + '_minh_hope_' + str(int(100*frac)) + \
            'percent_full_%d_%d'%(num_leaves,max_depth) + '_OPTION' + str(OPTION)

    print('submission file name:', subfilename)
    print('model file name:', modelfilename)
    print('fraction:', frac)

    start_time = time.time()

    params = {
        'learning_rate': 0.2,
        #'is_unbalance': 'true', # replaced with scale_pos_weight argument
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

    print('>> cleaning train...')
    train_df_array = train_df[predictors].values
    train_df_labels = train_df[target].values.astype('int').flatten()
    del train_df; gc.collect()
    print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')
    print('>> cleaning val...')
    val_df_array = val_df[predictors].values
    val_df_labels = val_df[target].values.astype('int').flatten()
    del val_df; gc.collect()
    print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')

    print('--------------------------------------------------------------------') 
    print(">> Training...")
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

    print('--------------------------------------------------------------------')                            
    print('Feature importances:', list(bst.feature_importance()))              

    print('[{}]: model training time'.format(time.time() - start_time))
    del train_df_array, train_df_labels, val_df_array, val_df_labels
    gc.collect()

    print('--------------------------------------------------------------------') 
    print('>> Save model...')
    # save model to file
    bst.save_model(modelfilename+'.txt')

    # print('--------------------------------------------------------------------') 
    # print('>> Plot feature importances...')
    # lgb.plot_importance(bst)
    # fig=plt.gcf()
    # fig.set_size_inches(50,50)
    # savename = modelfilename + '.png'
    # plt.savefig(savename)
    # print('done')     


    print('--------------------------------------------------------------------') 
    print('reading test')
    test_df = read_processed_h5(TEST_HDF5,predictors+['click_id'])
    print(test_df.info()); print(test_df.head())
    print_memory()
    print("test size : ", len(test_df))
    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')

    print(">> Predicting...")
    sub['is_attributed'] = bst.predict(test_df[predictors],num_iteration=best_iteration)
    # if not debug:
    print("writing...")
    sub.to_csv(subfilename,index=False,compression='gzip')
    print("done...")
    return sub

# num_leaves_list = [7,9,11,13,15,31,31,9]
# max_depth_list = [3,4,5,6,7,5,6,5]

num_leaves_list = [7,9,31]
max_depth_list = [3,4,5]

for i in range(len(num_leaves_list)):
    print ('==============================================================')
    num_leaves = num_leaves_list[i]
    max_depth = max_depth_list[i]
    print('num leaves:', num_leaves)
    print('max depth:', max_depth)
    predictors = get_predictors()
    subfilename = yearmonthdate_string + '_' + str(len(predictors)) + \
            'features_' + boosting_type + '_minh_hope_' + str(int(100*frac)) + \
            'percent_full_%d_%d'%(num_leaves,max_depth) + '_OPTION' + str(OPTION) + '.csv.gz'
    if os.path.isfile(subfilename):
        print('--------------------------------------')
        print('Already trained...')
    else:             
        sub=DO(frm,to,0,num_leaves,max_depth)

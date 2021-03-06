import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

debug=0
frac=1
print('debug', debug)

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

frm = 1; to = 100

TARGET = ['is_attributed']

if debug==1:
    DATASET = 'day9'
else:    
    DATASET = 'full'

PATH = '../codes_v6/'
TRAIN_HDF5 = 'train_' + DATASET + '.h5'
TEST_HDF5 = 'test_' + DATASET + '.h5'
if debug == 0:
    TRAIN_HDF5 = 'converted_' + TRAIN_HDF5
    TEST_HDF5 = 'converted_' + TEST_HDF5
TRAIN_HDF5 = PATH + TRAIN_HDF5
TEST_HDF5 = PATH + TEST_HDF5   

# OPTION 3 - PREVIOUS RESULT - 31_5_100_9781
PREDICTORS3 = [
    'app', 'device', 'os', 'channel', 'hour',
    'ip_nunique_channel',   # X0
    'ip_device_os_cumcount_app',
    'ip_day_nunique_hour',
    'ip_nunique_app',
    'ip_app_nunique_os',
    'ip_nunique_device',
    'app_nunique_channel',
    'ip_device_os_nunique_app', # X8
    'ip_os_device_app_nextclick',
    'ip_day_hour_count_channel',
    'ip_app_count_channel',
    'ip_app_os_count_channel',
    'ip_app_os_var_hour',
    'ip_app_channel_var_day',
    'ip_app_channel_mean_hour'
    ]     

# OPTION 15 - remove some cat to avoid overfitting?
PREDICTORS15 = [
    # core 9
    'app', 'os', 'hour', 'device', 
    'ip_os_device_app_nextclick',
    'ip_device_os_nunique_app',
    'ip_nunique_channel',
    'ip_nunique_app', 
    # add
    'ip_app_count_channel',
    'ip_nunique_device',
    'ip_cumcount_os',
    'app_nunique_channel',
    'ip_device_os_nextclick',
    'ip_nextclick',
    'ip_os_device_channel_app_nextclick',
    ]

PREDICTORS18 = [
    # core 9
    'app', 'os', 'device', 'channel', 'hour',
    'ip_os_device_app_nextclick',
    'ip_device_os_nunique_app',
    'ip_nunique_channel',
    'ip_nunique_app', 
    # add
    'ip_nunique_device',
    'ip_cumcount_os',
    'ip_device_os_nextclick',
    'ip_os_device_channel_app_nextclick',
    'ip_app_os_count_channel',
    'ip_count_app',
    'app_count_channel',
    'ip_device_os_nunique_channel',
    'ip_nextclick',
    'ip_channel_nextclick'
    ]	
	
NEW_FEATURE = [    
    'channel_count_app',
    'ip_count_app',
    'ip_app_count_os',
    'ip_count_device',
    'app_count_channel',
    'ip_device_os_nunique_channel',
    'channel_nunique_app'
    ]



NEW_FEATURE = [    
    'channel_count_app',
    'ip_count_app',
    'ip_app_count_os',
    'ip_count_device',
    'app_count_channel',
    'ip_device_os_nunique_channel',
    'channel_nunique_app'
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

def get_predictors(option):
    if option==3:
        predictors = PREDICTORS3
    if option==15:
        predictors = PREDICTORS15
    if option==18:
        predictors = PREDICTORS18  	

    print('------------------------------------------------')
    print('predictors:')
    for feature in predictors:
        print (feature)
    print('number of features:', len(predictors))            
    return predictors 

def get_categorical(predictors):
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
            if debug==1:
                train_df[feature] = pd.read_hdf(filename, key=feature, 
                        start=0, stop=10000000)                         
            if debug==0:
                train_df[feature] = pd.read_hdf(filename, key=feature)   

            if feature=='day' or feature=='hour' or feature=='min':
                train_df[feature] = train_df[feature].fillna(0)
                train_df[feature] = train_df[feature].astype('uint8')   
            if feature in NEW_FEATURE:
                print('convert {} to uint32'.format(feature))
                train_df[feature] = train_df[feature].fillna(0)
                train_df[feature] = train_df[feature].astype('uint32')                                                                                              
            print_memory()
    t1 = time.time()
    total = t1-t0
    print('total reading time:', total)
    print(train_df.info())   
    return train_df


def TRAIN(boosting_type, num_leaves,max_depth, option, num_boost_rounds_lgb, max_bin, min_data_in_leaf):
    print('------------------------------------------------')
    print('start...')
    print('fraction:', frac)
    print('prepare predictors, categorical and target...')
    predictors = get_predictors(option)
    categorical = get_categorical(predictors)
    target = TARGET
   
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

    subfilename = 'final_{}_numleave{}_maxdepth{}_maxbin{}_mininleaf{}_round{}_option{}.csv.gz'. \
            format(
                boosting_type,
                num_leaves,
                max_depth,
                max_bin,
                min_data_in_leaf,
                num_boost_rounds_lgb,
                option
            )
    modelfilename = 'final_{}_numleave{}_maxdepth{}_maxbin{}_mininleaf{}_round{}_option{}.txt'. \
            format(
                boosting_type,
                num_leaves,
                max_depth,
                max_bin,
                min_data_in_leaf,
                num_boost_rounds_lgb,
                option
            )

    print('----------------------------------------------------------')
    print('SUMMARY:')
    print('----------------------------------------------------------')
    print('predictors:',predictors)
    print('taget', target)
    print('categorical', categorical)
    print('submission file name:', subfilename)
    print('model file name:', modelfilename)
    print('fraction:', frac)
    print('option:', option)

    print('----------------------------------------------------------')
    train_df = read_processed_h5(TRAIN_HDF5, predictors+target)
    train_df = train_df.sample(frac=frac, random_state = SEED)
    print_memory('afer reading train:')
    print(train_df.head())
    print("train size: ", len(train_df))
    gc.collect()

    print('----------------------------------------------------------')
    print("Training...")
    
    params = {
        'boosting_type': boosting_type,
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.02,
        'num_leaves': num_leaves,  # we should let it be smaller than 2^(max_depth)
        'max_depth': max_depth,  # -1 means no limit
        'min_data_in_leaf': min_data_in_leaf,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': max_bin,  # Number of bucketed bin for feature values
        'subsample': 0.5,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'feature_fraction': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 10,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4,
        'verbose': 0,
        'scale_pos_weight': 200, # because training data is extremely unbalanced
    }
    print ('params:', params)

    print('>> cleaning train...')
    train_df_array = train_df[predictors].values
    train_df_labels = train_df[target].values.astype('int').flatten()
    del train_df; gc.collect()
    print_memory()

    print('>> prepare dataset...')
    dtrain_lgb = lgb.Dataset(train_df_array, label=train_df_labels,
                        feature_name=predictors,
                        categorical_feature=categorical)
    del train_df_array, train_df_labels; gc.collect()                        
    print_memory()                        
    
    print ('>> start trainning... ')
    start_time = time.time()
    model_lgb = lgb.train(
                        params, dtrain_lgb, 
                        num_boost_round=num_boost_rounds_lgb,
                        feature_name = predictors,
                        categorical_feature = categorical
                        )
    del dtrain_lgb; gc.collect()
    print('[{}]: model training time'.format(time.time() - start_time))

    print('--------------------------------------------------------------------') 
    print('>> save model...')
    # save model to file
    model_lgb.save_model(modelfilename)

    print('--------------------------------------------------------------------') 
    print('>> reading test')
    test_df = read_processed_h5(TEST_HDF5,predictors+['click_id'])
    print(test_df.info()); print(test_df.head())
    print_memory()
    print("test size : ", len(test_df))
    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')

    print(">> predicting...")
    sub['is_attributed'] = model_lgb.predict(test_df[predictors])
    # if not debug:
    print("writing...")
    sub.to_csv(subfilename,index=False,compression='gzip')
    print("done...")
    

# num_leaves_list =           [512,512,2048,2048]
# max_depth_list =            [64,64,256,256]
# num_boost_rounds_lgb_list = [400,400,400,400]
# max_bin_list =              [1024,1024,2048,2048]
# min_data_in_leaf_list =     [128,128,64,64]


# num_leaves_list =           [7,7,7,16,16,16,128,128,128]
# max_depth_list =            [3,3,3,-1,-1,-1,16,16,16]
# num_boost_rounds_lgb_list = [400,400,400,300,300,300,400,400,400]
# max_bin_list =              [100,100,100,64,64,64,512,512,512]
# min_data_in_leaf_list =     [100,100,100,16,16,16,128,128,128]


# num_leaves_list =           [128,   128,    16,     16,     31,     31]
# max_depth_list =            [16,    16,     -1,     -1,     7,      7]
# num_boost_rounds_lgb_list = [400,   400,    300,    300,    500,    500]
# max_bin_list =              [512,   512,    64,     64,     100,    100]
# min_data_in_leaf_list =     [128,   128,    16,     16,     100,    100]
# option_list = [18,3,18,3,18,3]

# num_leaves_list =           [7,     7,      9,      9,      31,     31]
# max_depth_list =            [3,     3,      4,      4,      7,      7]
# num_boost_rounds_lgb_list = [900,   900,    800,    800,    600,    600]
# max_bin_list =              [100,   100,    100,    100,    100,    100]
# min_data_in_leaf_list =     [100,   100,    100,    100,    100,    100]
# option_list =               [18,    3,      18,     3,      18,     3]

num_leaves_list =           [9,         63,         31,         15]
max_depth_list =            [4,         8,          7,          6]
num_boost_rounds_lgb_list = [800,       500,        700,        800]
max_bin_list =              [100,       100,        100,        100]
min_data_in_leaf_list =     [100,       100,        128,        128]
option_list =               [18,        3,          18,         18]


# for option in option_list:
# for boosting_type in ['gbdt', 'dart']:
for boosting_type in ['gbdt']:    
    for k in range(len(num_boost_rounds_lgb_list)):
        option = option_list[k]
        num_boost_rounds_lgb = num_boost_rounds_lgb_list[k]
        num_leaves = num_leaves_list[k]
        max_depth = max_depth_list[k]  
        max_bin = max_bin_list[k]  
        min_data_in_leaf = min_data_in_leaf_list[k]

        print ('==============================================================')
        # print('num leaves:', num_leaves)
        # print('max depth:', max_depth)
        # print ('option:', option)
        # print ('num_boost_rounds_lgb:', num_boost_rounds_lgb)
        # print ('max_bin:', max_bin)

        modelfilename = 'final_{}_numleave{}_maxdepth{}_maxbin{}_mininleaf{}_round{}_option{}.txt'. \
                format(
                    boosting_type,
                    num_leaves,
                    max_depth,
                    max_bin,
                    min_data_in_leaf,
                    num_boost_rounds_lgb,
                    option
                )
        print (modelfilename)                
        if os.path.isfile(modelfilename):
            print('--------------------------------------')
            print('Already trained...')
        else:             
            TRAIN(boosting_type, num_leaves,max_depth, option, num_boost_rounds_lgb, max_bin, min_data_in_leaf)
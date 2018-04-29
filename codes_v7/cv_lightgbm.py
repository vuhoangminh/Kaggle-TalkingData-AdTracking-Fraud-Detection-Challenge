import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

debug=1
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

boosting_type = 'gbdt'
# boosting_type = 'dart'



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


# OPTION 4 - RFE - 7_3_50_9777 9_4_50_9779
PREDICTORS4 = [
    'app', 'os', 'channel', 'hour',
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

# OPTION 5 - nextclick - 7_3_100_9776
PREDICTORS5 = [
    'app', 'os', 'channel', 'hour',
    'device', 'mobile',
    'ip_os_device_app_nextclick',
    'ip_nextclick',                 # 36
    'ip_app_nextclick',
    'ip_device_os_nextclick',       # 43
    'ip_channel_nextclick',         # 18
    'ip_os_device_app_nextclick',
    'ip_os_device_channel_app_nextclick',   # 18
    'ip_os_device_channel_nextclick',
    'ip_day_hour_var_channel',
    'ip_day_hour_nunique_channel',
    'ip_app_nunique_channel'        # 53
    ] 

# OPTION 6 - 7_3_50_9774
PREDICTORS6 = [ 
    # core 8
    'app', 'os', 'channel', 'hour',
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

# OPTION 7 - 7_3_50_9774
PREDICTORS7 = [
    # core 8    
    'app', 'os', 'channel', 'hour',
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

# OPTION 8 - 7_3_50_9772
PREDICTORS8 = [
    # core 8
    'app', 'os', 'channel', 'hour',
    'mobile',
    'ip_os_device_app_nextclick',
    'ip_device_os_nunique_app',
    'ip_nunique_channel',

    # add 10
    'ip_mobile_app_day_std_hour',
    'ip_mobile_app_day_var_hour',
    'ip_mobile_channel_day_nunique_hour',
    'ip_mobile_app_day_nunique_hour',
    'ip_mobile_channel_day_std_hour',
    'ip_mobile_channel_day_var_hour',
    'ip_mobile_app_channel_day_std_hour',
    'ip_mobile_app_channel_day_var_hour',
    'ip_mobile_app_channel_day_nunique_hour',
    'ip_app_channel_var_day',
    'ip_app_channel_mean_hour'
    ]  

# OPTION 9 - 7_3_50_9482
PREDICTORS9 = [
    # core 8
    'app', 'os', 'channel', 'hour',
    'mobile',
    'ip_os_device_app_nextclick',
    'ip_device_os_nunique_app',
    'ip_nunique_channel',

    # add 10
    'app_confRate',
    'channel_app_confRate',
    'ip_app_confRate',
    'ip_channel_confRate',
    'channel_confRate',
    'mobile_channel_confRate',
    'ip_confRate',
    'device_confRate',
    'os_confRate'
    ] 

# OPTION 10 - CORE
PREDICTORS10 = [
    # core 10
    'app', 'os', 'channel', 'hour',
    'mobile', 'device',
    'ip_os_device_app_nextclick',
    'ip_device_os_nunique_app',
    'ip_nunique_channel',
    'ip_nunique_app'
    ]

# OPTION 11
PREDICTORS11 = [
    # core 10
    'app', 'os', 'channel', 'hour',
    'mobile', 'device',
    'ip_os_device_app_nextclick',
    'ip_device_os_nunique_app',
    'ip_nunique_channel',
    'ip_nunique_app',

    'ip_app_nextclick',
    'ip_channel_nextclick',
    'ip_device_os_nextclick',
    'ip_nextclick',
    'ip_app_os_nunique_channel',
    'ip_app_channel_var_day',
    'ip_mobile_app_day_std_hour'
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
    if option==1:
        predictors = PREDICTORS1
    if option==2:
        predictors = PREDICTORS2
    if option==3:
        predictors = PREDICTORS3
    if option==4:
        predictors = PREDICTORS4
    if option==5:
        predictors = PREDICTORS5                
    if option==6:
        predictors = PREDICTORS6
    if option==7:
        predictors = PREDICTORS7
    if option==8:
        predictors = PREDICTORS8
    if option==9:
        predictors = PREDICTORS9
    if option==10:
        predictors = PREDICTORS10
    if option==11:
        predictors = PREDICTORS11



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
            print_memory()
    t1 = time.time()
    total = t1-t0
    print('total reading time:', total)
    print(train_df.info())   
    return train_df


def DO(num_leaves,max_depth, option):
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

    subfilename = yearmonthdate_string + '_' + str(len(predictors)) + \
            'features_' + boosting_type + '_cv_' + str(int(100*frac)) + \
            'percent_full_%d_%d'%(num_leaves,max_depth) + '_OPTION' + str(option) + '.csv.gz'
    modelfilename = yearmonthdate_string + '_' + str(len(predictors)) + \
            'features_' + boosting_type + '_cv_' + str(int(100*frac)) + \
            'percent_full_%d_%d'%(num_leaves,max_depth) + '_OPTION' + str(option)

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
    start_time = time.time()

    params = {
        'learning_rate': 0.2,
        'num_leaves': num_leaves,  # 2^max_depth - 1
        'max_depth': max_depth,  # -1 means no limit
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
    print_memory()

    print('>> prepare dataset...')
    dtrain_lgb = lgb.Dataset(train_df_array, label=train_df_labels,
                        feature_name=predictors,
                        categorical_feature=categorical)
    del train_df_array, train_df_labels; gc.collect()                        
    print_memory()                        
    
    print('>> start cv...')
    cv_results  = lgb.cv(params, 
                        dtrain_lgb, 
                        categorical_feature = categorical,
                        num_boost_round=1000,                       
                        metrics='auc',
                        shuffle = True,
                        stratified=True, 
                        nfold=5, 
                        show_stdv=True,
                        early_stopping_rounds=30, 
                        verbose_eval=True)                     


    print('[{}]: model training time'.format(time.time() - start_time))
    print('Total memory in use after cv training: ', process.memory_info().rss/(2**30), ' GB\n')

    print (cv_results)
    num_boost_rounds_lgb = len(cv_results['auc-mean'])
    print('num_boost_rounds_lgb=' + str(num_boost_rounds_lgb))

    print ('train model...')
    model_lgb = lgb.train(
                        params, dtrain_lgb, 
                        num_boost_round=num_boost_rounds_lgb,
                        feature_name = predictors,
                        categorical_feature = categorical)
    del dtrain_lgb
    gc.collect()

    print('--------------------------------------------------------------------') 
    print('>> Save model...')
    # save model to file
    model_lgb.save_model(modelfilename+'.txt')

    print('--------------------------------------------------------------------') 
    print('reading test')
    test_df = read_processed_h5(TEST_HDF5,predictors+['click_id'])
    print(test_df.info()); print(test_df.head())
    print_memory()
    print("test size : ", len(test_df))
    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')

    print(">> Predicting...")
    sub['is_attributed'] = model_lgb.predict(test_df[predictors])
    # if not debug:
    print("writing...")
    sub.to_csv(subfilename,index=False,compression='gzip')
    print("done...")
    return sub

num_leaves_list = [11]
max_depth_list = [5]
option_list = [3]

for option in option_list:
    for i in range(len(num_leaves_list)):
        print ('==============================================================')
        num_leaves = num_leaves_list[i]
        max_depth = max_depth_list[i]
        print('num leaves:', num_leaves)
        print('max depth:', max_depth)
        if debug: print ('option:', option)
        predictors = get_predictors(option)
        subfilename = yearmonthdate_string + '_' + str(len(predictors)) + \
                'features_' + boosting_type + '_minh_hope_' + str(int(100*frac)) + \
                'percent_full_%d_%d'%(num_leaves,max_depth) + '_OPTION' + str(option) + '.csv.gz'
        if debug: print (subfilename)                
        if os.path.isfile(subfilename):
            print('--------------------------------------')
            print('Already trained...')
        else:             
            sub=DO(num_leaves,max_depth, option)
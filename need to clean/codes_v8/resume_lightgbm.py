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

boosting_type = 'gbdt'
# boosting_type = 'dart'


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

# OPTION 12 - 11 5 9782
PREDICTORS12 = [
    # core 9
    'app', 'os', 'channel', 'hour',
    'mobile',
    'ip_os_device_app_nextclick',
    'ip_device_os_nunique_app',
    'ip_nunique_channel',
    'ip_nunique_app',

# final round 
    'ip_app_count_channel',
    'ip_nunique_device',
    'ip_app_nunique_os',
    'ip_app_nextclick',
    'ip_device_os_cumcount_app',
    'ip_app_os_nunique_channel',
    'ip_app_os_var_hour',
    'ip_cumcount_os',
    'app_nunique_channel',
    'ip_day_hour_nunique_channel',
    'ip_day_hour_var_channel',
    'ip_app_channel_mean_hour',
    'ip_device_os_nextclick',
    'ip_nextclick',
    'ip_day_hour_count_channel',
    'ip_os_device_channel_app_nextclick',
    'ip_os_device_channel_nextclick',
    'mobile_app',
    'mobile_channel',
    'app_channel'
    ]


# OPTION 13 - no cat - 9730 ===> NO
PREDICTORS13 = [
    # core 9
    # 'app', 'os', 'channel', 'hour',
    # 'mobile',
    'ip_os_device_app_nextclick',
    'ip_device_os_nunique_app',
    'ip_nunique_channel',
    'ip_nunique_app',

    # final round
    'ip_app_count_channel',
    'ip_nunique_device',
    'ip_app_nunique_os',
    'ip_app_nextclick',
    'ip_device_os_cumcount_app',
    'ip_app_os_nunique_channel',
    'ip_app_os_var_hour',
    'ip_cumcount_os',
    'app_nunique_channel',
    'ip_day_hour_nunique_channel',
    'ip_day_hour_var_channel',
    'ip_app_channel_mean_hour',
    'ip_device_os_nextclick',
    'ip_nextclick',
    'ip_day_hour_count_channel',
    'ip_os_device_channel_app_nextclick',
    'ip_os_device_channel_nextclick',
    # 'mobile_app',
    # 'mobile_channel',
    # 'app_channel'
    ]

# OPTION 14 - cat and some - 9783
PREDICTORS14 = [
    # core 9
    'app', 'os', 'channel', 'hour',
    'mobile',
    'ip_os_device_app_nextclick',
    'ip_device_os_nunique_app',
    'ip_nunique_channel',
    'ip_nunique_app',

    # final round
    'ip_app_count_channel',
    'ip_nunique_device',
    'ip_cumcount_os',
    'app_nunique_channel',
    'ip_device_os_nextclick',
    'ip_nextclick',
    'ip_os_device_channel_app_nextclick',
    'mobile_app',
    'mobile_channel',
    'app_channel'
    ]    

# OPTION 15 - remove some cat to avoid overfitting?
PREDICTORS15 = [
    # core 9
    'app', 'os', 'hour', 'device', 
    'ip_os_device_app_nextclick',
    'ip_device_os_nunique_app',
    'ip_nunique_channel',
    'ip_nunique_app',
    
    # final round
    'ip_app_count_channel',
    'ip_nunique_device',
    'ip_cumcount_os',
    'app_nunique_channel',
    'ip_device_os_nextclick',
    'ip_nextclick',
    'ip_os_device_channel_app_nextclick',
    ]

PREDICTORS19 = [    
    # core 9
    'app', 'os', 'device', 'channel', 'hour',
    'ip_os_device_app_nextclick',
    'ip_device_os_nunique_app',
    'ip_nunique_channel',
    'ip_nunique_app', 
    # add 18
    'app_count_channel',
    'ip_device_os_nunique_channel',
    'ip_nextclick',
    'ip_channel_nextclick',
    'ip_device_os_nextclick',
    'ip_count_app',
    'ip_nunique_device',
    'ip_cumcount_os',
    'ip_os_device_channel_app_nextclick',
    'ip_app_os_count_channel',
    # add 3
    'ip_device_os_cumcount_app',
    'ip_day_nunique_hour',
    'ip_app_nunique_os',    # remove?
    'app_nunique_channel',  # remove?  
    'ip_app_os_var_hour',
    'ip_app_channel_var_day',
    'ip_app_channel_mean_hour',
    
    # add
    'ip_os_device_channel_nextclick',   # remove?
    'ip_os_device_app_hour_nextclick',
    'ip_os_device_channel_hour_nextclick',
    'device_nextclick',
    'device_channel_nextclick',     
    'app_device_channel_nextclick',
    'device_hour_nextclick',
    'ip_day_count_hour',
    'ip_app_count_os',      # remove?
    'ip_day_channel_var_hour',  # remove?    
    'os_device_app_channel_mean_hour',
    'os_device_app_channel_nunique_hour'
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
    if option==12:
        predictors = PREDICTORS12
    if option==13:
        predictors = PREDICTORS13
    if option==14:
        predictors = PREDICTORS14
    if option==15:
        predictors = PREDICTORS15
    if option==19:
        predictors = PREDICTORS19


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

def lgb_modelfit_nocv(params, train_df_array, train_df_labels, val_df_array, val_df_labels, predictors, target='target', objective='binary', metrics='auc',
                 feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': boosting_type,
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.05,
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
        'nthread': 8,
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
                     verbose_eval=1, 
                     feval=feval)

    print("\nModel Report")
    print("bst1.best_iteration: ", bst1.best_iteration)
    print(metrics+":", evals_results['valid'][metrics][bst1.best_iteration-1])

    return (bst1,bst1.best_iteration)

NEW_FEATURE = [    
    'channel_count_app',
    'ip_count_app',
    'ip_app_count_os',
    'ip_count_device',
    'app_count_channel',
    'ip_device_os_nunique_channel',
    'channel_nunique_app',
    'ip_day_count_hour',
    'ip_app_count_os',
    'os_device_app_channel_nunique_hour'
    ]

NEW_FEATURE2 = [    
    'ip_day_channel_var_hour',  # remove?    
    'os_device_app_channel_mean_hour'
    ]   

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
            if feature in NEW_FEATURE2:
                print('convert {} to float32'.format(feature))
                train_df[feature] = train_df[feature].fillna(0)
                train_df[feature] = train_df[feature].astype('uint32')     

            if 'nextclick' in feature:
                print('>> log-binning features...')
                train_df[feature]= np.log2(1 + train_df[feature].values).astype('uint32')


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
    print('categorical', categorical)
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
    print('predictors', predictors)
    print('categorical', categorical)
    print('target', target)
    print('option:', option)
    print('=======================================================================')
    train_df = read_processed_h5(TRAIN_HDF5, predictors+target)

    if frac<1:
        train_df = train_df.sample(frac=frac, random_state = SEED)
    print_memory('afer reading train')
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=SEED)
    print(train_df.head())
    print_memory('afer split')
    print("train size: ", len(train_df))
    print("valid size: ", len(val_df))

    gc.collect()

    subfilename = yearmonthdate_string + '_' + str(len(predictors)) + \
            'features_' + boosting_type + '_minh_hope_' + str(int(100*frac)) + \
            'percent_full_%d_%d'%(num_leaves,max_depth) + '_OPTION' + str(option) + '.csv.gz'
    modelfilename = yearmonthdate_string + '_' + str(len(predictors)) + \
            'features_' + boosting_type + '_minh_hope_' + str(int(100*frac)) + \
            'percent_full_%d_%d'%(num_leaves,max_depth) + '_OPTION' + str(option)

    print('submission file name:', subfilename)
    print('model file name:', modelfilename)
    print('fraction:', frac)

    start_time = time.time()

    params = {
        'learning_rate': 0.05,
        'num_leaves': num_leaves,  # 2^max_depth - 1
        'max_depth': max_depth,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.5,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight':200 # because training data is extremely unbalanced 
    }
    print (params)

    print('>> cleaning train...')
    train_df_array = train_df[predictors].values.astype(np.float32)
    train_df_labels = train_df[target].values.astype('int').flatten()
    del train_df; gc.collect()
    print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')
    print('>> cleaning val...')
    val_df_array = val_df[predictors].values.astype(np.float32)
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
                            early_stopping_rounds=20, 
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

num_leaves_list = [7,15,9]
max_depth_list = [3,7,4]
option_list = [19]

for option in option_list:
    for i in range(len(num_leaves_list)):
        print ('==============================================================')
        # i = len(num_leaves_list) - k - 1
        num_leaves = num_leaves_list[i]
        max_depth = max_depth_list[i]
        print('num leaves:', num_leaves)
        print('max depth:', max_depth)
        if debug: print ('option:', option)
        predictors = get_predictors(option)
        subfilename = yearmonthdate_string + '_' + str(len(predictors)) + \
                'features_' + boosting_type + '_minh_hope_' + str(int(100*frac)) + \
                'percent_full_%d_%d'%(num_leaves,max_depth) + '_OPTION' + str(option) + '.csv.gz'
        print (subfilename)                
        if os.path.isfile(subfilename):
            print('--------------------------------------')
            print('Already trained...')
        else:             
            sub=DO(num_leaves,max_depth, option)

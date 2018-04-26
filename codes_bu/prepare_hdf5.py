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
import random as rnd
import os
import featuretools as ft
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import configs
import math

process = psutil.Process(os.getpid())

# # init vars
# CATEGORY_LIST = configs.cat_list
# DATATYPE_LIST = configs.datatype_list
# NCHUNKS = configs.nchunks
# CAT_COMBINATION_FILENAME = configs.CAT_COMBINATION_FILENAME
# NEXTCLICK_FILENAME = configs.NEXTCLICK_FILENAME
# TIME_FILENAME = configs.TIME_FILENAME

CATEGORY_LIST = ['ip', 'app', 'device', 'os', 'channel', 
    'mobile', 'mobile_app', 'mobile_channel', 'app_channel'
    ]
# DATATYPE_LIST = {
#     'ip'                : 'uint32',
#     'app'               : 'uint16',
#     'device'            : 'uint16',
#     'os'                : 'uint16',
#     'channel'           : 'uint16',
#     'is_attributed'     : 'uint8',
#     'click_id'          : 'uint32',
#     'mobile'            : 'uint16',
#     'mobile_app'        : 'uint16',
#     'mobile_channel'    : 'uint16',
#     'app_channel'       : 'uint16',
#     'category'          : 'category',
#     'epochtime'         : 'int64',
#     'nextClick'         : 'int64',
#     'nextClick_shift'   : 'float64',
#     'min'               : 'uint8',
#     'day'               : 'uint8',
#     'hour'              : 'uint8',
#     'ip_mobile_day_count_hour'                  : 'uint32',
#     'ip_mobile_app_day_count_hour'              : 'uint32',
#     'ip_mobile_channel_day_count_hour'          : 'uint32',
#     'ip_app_channel_day_count_hour'             : 'uint32',
#     'ip_mobile_app_channel_day_count_hour'      : 'uint32',
#     'ip_mobile_day_var_hour'                    : 'float32',
#     'ip_mobile_app_day_var_hour'                : 'float32',
#     'ip_mobile_channel_day_var_hour'            : 'float32',
#     'ip_app_channel_day_var_hour'               : 'float32',
#     'ip_mobile_app_channel_day_var_hour'        : 'float32',
#     'ip_mobile_day_std_hour'                    : 'float32',
#     'ip_mobile_app_day_std_hour'                : 'float32',
#     'ip_mobile_channel_day_std_hour'            : 'float32',
#     'ip_app_channel_day_std_hour'               : 'float32',
#     'ip_mobile_app_channel_day_std_hour'        : 'float32',
#     'ip_mobile_day_cumcount_hour'               : 'uint32',
#     'ip_mobile_app_day_cumcount_hour'           : 'uint32',
#     'ip_mobile_channel_day_cumcount_hour'       : 'uint32',
#     'ip_app_channel_day_cumcount_hour'          : 'uint32',
#     'ip_mobile_app_channel_day_cumcount_hour'   : 'uint32',
#     'ip_mobile_day_nunique_hour'                : 'uint32',
#     'ip_mobile_app_day_nunique_hour'            : 'uint32',
#     'ip_mobile_channel_day_nunique_hour'        : 'uint32',
#     'ip_app_channel_day_nunique_hour'           : 'uint32',
#     'ip_mobile_app_channel_day_nunique_hour'    : 'uint32'
#     }

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

TRAIN_HDF5 = 'train_day9.h5'
TEST_HDF5 = 'test_day9.h5'

DATATYPE_LIST_STRING = {
    'mobile'            : 'category',
    'mobile_app'        : 'category',
    'mobile_channel'    : 'category',
    'app_channel'       : 'category',
    }

CAT_COMBINATION_FILENAME = 'day9_cat_combination.csv'
CAT_COMBINATION_NUMERIC_CATEGORY_FILENAME = 'day9_cat_combination_numeric_category.csv'
NEXTCLICK_FILENAME = 'day9_nextClick.csv'
TIME_FILENAME = 'day9_day_hour_min.csv'
IP_HOUR_RELATED_FILENAME = 'day9_ip_hour_related.csv'
TRAINSET_FILENAME = '../input/valid_day_9.csv'
NCHUNKS = 100000

debug=1
NROWS=10000000
# nrows=10

if not debug:
    print('=======================================================================')
    print('process on server...')
    print('=======================================================================')
else:
    print('=======================================================================')
    print('for testing only...')
    print('=======================================================================')

SIZE_TRAIN = 53016937
SIZE_TEST = 18790468
input = '../processed_day9/'

def print_memory(print_string=''):
    print('Total memory in use ' + print_string + ': ', process.memory_info().rss/(2**30), ' GB')

def get_keys_h5(f):
    return [key for key in f.keys()]

def convert_to_save_memory(train_df, usecols, dtype = DATATYPE_LIST):
    for feature, feature_type in DATATYPE_LIST.items(): 
        if feature in list(train_df) and feature in usecols:
            print('convert', feature, 'to', feature_type)
            train_df[feature]=train_df[feature].astype(feature_type)   
    print(train_df.info())                 
    return train_df

def read_train(usecols_train, filename):
    if debug:
        if 'click_time' in usecols_train:
            train_df = pd.read_csv(filename, 
                skiprows=range(1,10), nrows=NROWS, dtype=DATATYPE_LIST, 
                parse_dates=['click_time'], usecols=usecols_train)
        else:
            train_df = pd.read_csv(filename, 
                skiprows=range(1,10), nrows=NROWS, dtype=DATATYPE_LIST, 
                usecols=usecols_train)
    else:
        if 'click_time' in usecols_train:
            train_df = pd.read_csv(filename, parse_dates=['click_time'],
                header=0 , dtype=DATATYPE_LIST, usecols=usecols_train)           
        else:
            train_df = pd.read_csv(filename, header=0, 
                dtype=DATATYPE_LIST, usecols=usecols_train)    
    return train_df                     

def read_train_test(style='is_merged', \
    usecols_train=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'], \
    usecols_test=['ip','app','device','os', 'channel', 'click_time', 'click_id']):
    print('loading train data...')
    train_df = read_train(usecols_train, TRAINSET_FILENAME)
    print('len train:', len(train_df))            
    print_memory ('after reading train')

    print('loading test data...')
    test_df = read_train(usecols_test, '../input/test.csv')
    print('len test:', len(test_df))            
    print_memory ('after reading test')
    if style == 'is_merged':
        train_df = train_df.append(test_df)
        train_df = train_df.fillna(0)
        del test_df; gc.collect()
        return train_df 
    else:
        return train_df, test_df  

def prepare_dataset(which_dataset, train_df, filename, usecols, dtype = DATATYPE_LIST):
    if which_dataset == 'test':
        if debug:
            nrows = NROWS
            skiprows = NROWS
        else:
            nrows = SIZE_TEST
            skiprows = SIZE_TRAIN
        gp = pd.read_csv(filename, dtype=dtype, usecols=usecols, 
                nrows=nrows, skiprows=range(1,skiprows+1))

    if which_dataset == 'train':
        if debug:
            nrows = NROWS
        else:
            nrows = SIZE_TRAIN
        gp = pd.read_csv(filename, dtype=dtype, usecols=usecols, 
                nrows=nrows)
    
    print('merge with train_df')                
    train_df = pd.concat([train_df, gp], axis=1, join_axes=[train_df.index])
    if dtype != DATATYPE_LIST_STRING:
        train_df = train_df.fillna(0)
        print('convert to save memory...')
        train_df = convert_to_save_memory(train_df, usecols)
    del gp; gc.collect()
    return train_df

def get_filename(selcols, apply_type):
    feature_name = ''
    for i in range(len(selcols)-1):
        feature_name = feature_name + selcols[i] + '_'
    feature_name = feature_name + apply_type + '_' + selcols[len(selcols)-1]
    print('doing feature:', feature_name)
    filename = input + 'day9_' + feature_name + '.csv'
    return filename, feature_name


# APPLY_TYPE_LIST = ['count', 'cumcount', 'nunique', 'var', 'std'] # will run full if enough RAM
APPLY_TYPE_LIST = ['count', 'cumcount', 'var', 'nunique', 'std']
APPLY_TYPE_LIST_TRAIN = ['count', 'cumcount', 'nunique']

selcols0 = ['ip','mobile','day','hour']
selcols1 = ['ip','mobile_app','day','hour']
selcols2 = ['ip','mobile_channel','day','hour']
selcols3 = ['ip','app_channel','day','hour']
selcols4 = ['ip', 'mobile','app_channel','day','hour']    

selcols_list = [selcols0, selcols1, selcols2, selcols3, selcols4]

def get_info_key_hdf5(store_name, key):
    print('-----------------------')
    store = pd.HDFStore(store_name)
    print(store)
    df = store.select(key) 
    print('-----------------------')
    print(df.info())
    # print(df.head())

def add_dataset_to_hdf5(save_name, train_df, which_dataset):
    if which_dataset == 'test':
        store_name = TEST_HDF5             
    if which_dataset == 'train':
        store_name = TRAIN_HDF5 
    usecols = list(train_df)

    store = pd.HDFStore(store_name) 
    existing_key = store.keys()
    for feature in usecols:
        key = '/' + feature
        if key in existing_key:
            print ('feature already added...')
        else:                    
            print ('add key to hdf5...')                        
            temp = pd.DataFrame()
            temp[feature] = train_df[feature]
            temp.to_hdf(store_name, key=feature, mode='a')
            get_info_key_hdf5(store_name, key=feature)

def prepare_dataset_hdf5(which_dataset, train_df, filename, usecols, dtype = DATATYPE_LIST):
    if which_dataset == 'test':
        if debug:
            nrows = NROWS
            skiprows = NROWS
        else:
            nrows = SIZE_TEST
            skiprows = SIZE_TRAIN
        store_name = TEST_HDF5             

    if which_dataset == 'train':
        if debug:
            nrows = NROWS
        else:
            nrows = SIZE_TRAIN
        store_name = TRAIN_HDF5                        
    store = pd.HDFStore(store_name) 
    existing_key = store.keys()
    # print(existing_key)
    for feature in usecols:
        key = '/' + feature
        if key in existing_key:
            print ('feature already added...')
        else:
            if which_dataset == 'test':
                gp = pd.read_csv(filename, dtype=dtype, usecols=usecols, 
                        nrows=nrows, skiprows=range(1,skiprows+1))
            else:
                gp = pd.read_csv(filename, dtype=dtype, usecols=usecols, 
                        nrows=nrows)                        
            print ('add key to hdf5...')                        
            temp = pd.DataFrame()
            temp[feature] = gp[feature]
            temp.to_hdf(store_name, key=feature, mode='a')
            get_info_key_hdf5(store_name, key=feature)

def do_same(save_name, train_df, which_dataset):
    print(train_df.info())

    print('-------------------------------------------------------------------')
    print('load day9_cat_combination_numeric_category...')
    filename = input + 'day9_cat_combination_numeric_category.csv'
    prepare_dataset_hdf5(which_dataset, train_df, 
            filename, usecols=['mobile', 'mobile_app', 'mobile_channel', 'app_channel'])
    print_memory()    

    print('-------------------------------------------------------------------')
    print('load day9_day_hour_min...')
    filename = input + 'day9_day_hour_min.csv'
    prepare_dataset_hdf5(which_dataset, train_df, 
            filename, usecols=['day', 'hour', 'min'])
    print_memory()

    print('-------------------------------------------------------------------')
    print('load day9_nextClick...')
    filename = input + 'day9_nextClick.csv'
    prepare_dataset_hdf5(which_dataset, train_df, 
            filename, usecols=['nextClick', 'nextClick_shift'])
    print_memory()

    if which_dataset == 'train':
        apply_type_list = APPLY_TYPE_LIST
    else:
        apply_type_list = APPLY_TYPE_LIST                

    for apply_type in apply_type_list:
        for selcols in selcols_list:
            print('-------------------------------------------------------------------')
            print('merging...')
            print('select column:', selcols)
            print('apply type:', apply_type)
            filename, feature_name = get_filename(selcols, apply_type)
            prepare_dataset_hdf5(which_dataset, train_df, filename, 
                    usecols=[feature_name])
            print_memory()

    # print('saving...')
    # train_df.to_hdf(save_name, key=which_dataset, mode='w')   

def do_train():
    print('-------------------------------------------------------------------')
    print('do TRAIN')
    print('-------------------------------------------------------------------')
    print('load dataset...')
    train_df, test_df = read_train_test('is_not_merged')
    del test_df; gc.collect()
    print_memory()
    which_dataset = 'train'
    save_name='train_day9.h5'
    add_dataset_to_hdf5(save_name, train_df, which_dataset)
    do_same(save_name, train_df, which_dataset)
    

def do_test():
    print('-------------------------------------------------------------------')
    print('do TEST')
    print('-------------------------------------------------------------------')
    print('load dataset...')
    # train_df, test_df = read_train_test('is_not_merged', usecols_train=['is_attributed'],
    #             usecols_test=['click_id'])
    train_df, test_df = read_train_test('is_not_merged')                
    del train_df; gc.collect()
    print_memory()
    which_dataset = 'test'
    save_name='test_day9.h5'
    add_dataset_to_hdf5(save_name, test_df, which_dataset)
    do_same(save_name, test_df, which_dataset)

do_test()
do_train()
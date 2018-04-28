debug=0
print('debug', debug)

if not debug:
    print('=======================================================================')
    print('process on server...')
    print('=======================================================================')
else:
    print('=======================================================================')
    print('for testing only...')
    print('=======================================================================')

if debug==1:
    DATASET = 'day9'
else:
    DATASET = 'full'  

import pandas as pd
import h5py
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
import glob

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
    'hour'              : 'uint8'
    }

TRAIN_HDF5 = 'train_' + DATASET + '.h5'
TEST_HDF5 = 'test_' + DATASET + '.h5'
# TEST_HDF5 = 'test_day123.h5'
      


DATATYPE_LIST_STRING = {
    'mobile'            : 'category',
    'mobile_app'        : 'category',
    'mobile_channel'    : 'category',
    'app_channel'       : 'category',
    }

if debug==1:
    PATH = '../debug_processed_day9/'        
else:
    PATH = '../processed_full/'                
CAT_COMBINATION_FILENAME = PATH + DATASET + '_cat_combination.csv'
CAT_COMBINATION_NUMERIC_CATEGORY_FILENAME = PATH + DATASET + '_cat_combination_numeric_category.csv'
NEXTCLICK_FILENAME = PATH + DATASET + '_nextClick.csv'
TIME_FILENAME = PATH + DATASET + '_day_hour_min.csv'
IP_HOUR_RELATED_FILENAME = PATH + DATASET + '_ip_hour_related.csv'
if debug==1:
    TRAINSET_FILENAME = '../input/valid_day_9.csv'
else:
    TRAINSET_FILENAME = '../input/train.csv'        
NCHUNKS = 100000
if debug==1:
    NROWS=10000000
else: 
    NROWS =10000    
# nrows=10

if not debug:
    print('=======================================================================')
    print('process on server...')
    print('=======================================================================')
else:
    print('=======================================================================')
    print('for testing only...')
    print('=======================================================================')

SIZE_TRAIN = 184903890
SIZE_TEST = 18790469

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
    print('>> doing feature:', feature_name)
    filename = PATH + DATASET + '_' + feature_name + '.csv'
    return filename, feature_name

REMOVED_LIST = [
    'cat_combination.csv',
    'nextClick.csv',
    'day_hour_min.csv',
    'cat_combination_numeric_category.csv'
    ]

# DATATYPE_DICT = {
#     'count'     : 'uint32',
#     'nunique'   : 'uint32',
#     'cumcount'  : 'uint32',
#     'var'       : 'float32',
#     'std'       : 'float32',
#     'confRate'  : 'uint32',
#     'nextclick' : 'int64',
#     'mean'      : 'float32'
#     }

DATATYPE_DICT = {
    'count'     : 'float',
    'nunique'   : 'float',
    'cumcount'  : 'float',
    'var'       : 'float32',
    'std'       : 'float32',
    'confRate'  : 'uint32',
    'nextclick' : 'int',
    'mean'      : 'float32'
    }

def get_datatype(feature_name):
    datatype = 'UNKNOWN'
    for key, type in DATATYPE_DICT.items():
        if key in feature_name:
            datatype = type
            break
    return datatype           

def update_datatype_dict():
    datatype_list = DATATYPE_LIST
    files = glob.glob(PATH + "*.csv") 
    print (files)
    if debug:
        PATH_corrected = PATH.replace(DATASET + '/', DATASET + '\\') 
        removed_string = PATH_corrected + DATASET +'_'
    else:
        PATH_corrected = PATH
        removed_string = PATH_corrected + DATASET + '_'        
    print(removed_string)
    for file in files:
        feature_name = file.replace(removed_string,'')
        if feature_name not in REMOVED_LIST:
            feature_name = feature_name.split('.')[0]
            feature_type = get_datatype(feature_name)
            # print('feature {} has datatype {}'.format(feature_name, feature_type))
            datatype_list[feature_name] = feature_type
    return datatype_list

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

    # if not os.path.exists(store_name):
    #     print ('add dump')
    #     dump_df = pd.DataFrame({'col1':[0,2,3,2],'col2':[1,0,0,1]})
    #     dump_df.to_hdf(store_name, 'dump_later', mode='a')
    # # print(store_name)     
    # # store = pd.io.pytables.HDFStore(store_name)
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

def prepare_dataset_hdf5(which_dataset, filename, usecols, dtype = DATATYPE_LIST):
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

def do_same(save_name, which_dataset):
    # print(train_df.info())

    print('-------------------------------------------------------------------')
    print('load full_cat_combination_numeric_category...')
    filename = PATH + DATASET + '_cat_combination_numeric_category.csv'
    prepare_dataset_hdf5(which_dataset, 
            filename, usecols=['mobile', 'mobile_app', 'mobile_channel', 'app_channel'])
    print_memory()  

    print('-------------------------------------------------------------------')
    print('load full_cat_combination_numeric_category...')
    filename = PATH + DATASET + '_cat_combination_numeric_category.csv'
    prepare_dataset_hdf5(which_dataset,  
            filename, usecols=['mobile', 'mobile_app', 'mobile_channel', 'app_channel'])
    print_memory()    

    print('-------------------------------------------------------------------')
    print('load full_day_hour_min...')
    filename = PATH + DATASET + '_day_hour_min.csv'
    prepare_dataset_hdf5(which_dataset,  
            filename, usecols=['day', 'hour', 'min'])
    print_memory()

    # print('-------------------------------------------------------------------')
    # print('load full_nextClick...')
    # filename = PATH + 'full_nextClick.csv'
    # prepare_dataset_hdf5(which_dataset, train_df, 
    #         filename, usecols=['nextClick', 'nextClick_shift'])
    # print_memory()
       
    for feature_name, type in DATATYPE_LIST_UPDATED.items():
        is_added = False
        for key, key_datatype in DATATYPE_DICT.items():
            if key in feature_name:
                is_added = True
        if is_added:                
            print('-------------------------------------------------------------------')
            filename = PATH + DATASET +'_' + feature_name + '.csv'
            print ('>> doing: {}, type {}, and save to {}'.format(feature_name, type, filename))
            print('merging...')
            prepare_dataset_hdf5(which_dataset, filename, 
                    usecols=[feature_name])
            print_memory()

def do_train():
    print('-------------------------------------------------------------------')
    print('do TRAIN')
    print('-------------------------------------------------------------------')
    print('load dataset...')
    # train_df, test_df = read_train_test('is_not_merged')
    # del test_df; gc.collect()
    print_memory()
    which_dataset = 'train'
    save_name='train_' + DATASET + '.h5'
    # add_dataset_to_hdf5(save_name, train_df, which_dataset)
    do_same(save_name, which_dataset)
    
def do_test():
    print('-------------------------------------------------------------------')
    print('do TEST')
    print('-------------------------------------------------------------------')
    print('load dataset...')
    # train_df, test_df = read_train_test('is_not_merged')                
    # del train_df; gc.collect()
    print_memory()
    which_dataset = 'test'
    save_name='test_' + DATASET + '.h5'
    # add_dataset_to_hdf5(save_name, test_df, which_dataset)
    do_same(save_name, which_dataset)
    
DATATYPE_LIST_UPDATED = update_datatype_dict()

for key, type in DATATYPE_LIST_UPDATED.items():
    print (key, type)    


# print('============================================================================')
# print('FINAL SUMMARY')
# print('============================================================================')

do_test()
do_train()

print('============================================================================')
print('FINAL SUMMARY')
print('============================================================================')
print('-----------------------------------------------------')
print('reading train')
train_h5 = pd.HDFStore(TRAIN_HDF5)
print(train_h5)
train_df = train_h5.select('ip_mobile_channel_day_cumcount_hour') 
print(train_df.info()); print(train_df.head())
print_memory()

print('-----------------------------------------------------')
print('reading test')
train_h5 = pd.HDFStore(TEST_HDF5)
print(train_h5)
train_df = train_h5.select('ip_mobile_channel_day_cumcount_hour') 
print(train_df.info()); print(train_df.head())
print_memory()



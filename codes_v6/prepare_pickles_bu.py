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
    'ip_mobile_day_count_hour'              : 'uint32',
    'ip_mobileapp_day_count_hour'           : 'uint32',
    'ip_mobilechannel_day_count_hour'       : 'uint32',
    'ip_appchannel_day_count_hour'          : 'uint32',
    'ip_mobile_app_channel_day_count_hour'  : 'uint32',
    'ip_mobile_day_var_hour'                : 'float32',
    'ip_mobileapp_day_var_hour'             : 'float32',
    'ip_mobilechannel_day_var_hour'         : 'float32',
    'ip_appchannel_day_var_hour'            : 'float32',
    'ip_mobile_app_channel_day_var_hour'    : 'float32',
    'ip_mobile_day_std_hour'                : 'float32',
    'ip_mobileapp_day_std_hour'             : 'float32',
    'ip_mobilechannel_day_std_hour'         : 'float32',
    'ip_appchannel_day_std_hour'            : 'float32',
    'ip_mobile_app_channel_day_std_hour'    : 'float32',
    'ip_mobile_day_cumcount_hour'               : 'uint32',
    'ip_mobileapp_day_cumcount_hour'            : 'uint32',
    'ip_mobilechannel_day_cumcount_hour'        : 'uint32',
    'ip_appchannel_day_cumcount_hour'           : 'uint32',
    'ip_mobile_app_channel_day_cumcount_hour'   : 'uint32',
    'ip_mobile_day_nunique_hour'                : 'uint32',
    'ip_mobileapp_day_nunique_hour'             : 'uint32',
    'ip_mobilechannel_day_nunique_hour'         : 'uint32',
    'ip_appchannel_day_nunique_hour'            : 'uint32',
    'ip_mobile_app_channel_day_nunique_hour'    : 'uint32'
    }

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

debug=0
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

def convert_to_save_memory(train_df):
    for feature, type in DATATYPE_LIST.items(): 
        if feature in list(train_df):
            print('convert', feature)
            train_df[feature]=train_df[feature].astype(type)   
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


def prepare_dataset(which_dataset, train_df, filename, usecols):
    if which_dataset == 'test':
        if debug:
            nrows = NROWS
            skiprows = NROWS
        else:
            nrows = SIZE_TEST
            skiprows = SIZE_TRAIN
        gp = pd.read_csv(filename, dtype=DATATYPE_LIST, usecols=usecols, 
                nrows=nrows, skiprows=range(1,skiprows+1))

    if which_dataset == 'train':
        if debug:
            nrows = NROWS
        else:
            nrows = SIZE_TRAIN
        gp = pd.read_csv(filename, dtype=DATATYPE_LIST, usecols=usecols, 
                nrows=nrows)
    print('convert to save memory...')
    gp = convert_to_save_memory(gp)
    print('merge with train_df')                
    train_df = pd.concat([train_df, gp], axis=1, join_axes=[train_df.index])
    return train_df

def get_filename(selcols, apply_type):
    feature_name = ''
    for i in range(len(selcols)-1):
        feature_name = feature_name + selcols[i] + '_'
    feature_name = feature_name + apply_type + '_' + selcols[len(selcols)-1]
    print('doing feature:', feature_name)
    filename = input + 'day9_' + feature_name + '.csv'
    return filename, feature_name


apply_type_list = ['count', 'cumcount', 'nunique', 'var', 'std']

selcols0 = ['ip','mobile','day','hour']
selcols1 = ['ip','mobile_app','day','hour']
selcols2 = ['ip','mobile_channel','day','hour']
selcols3 = ['ip','app_channel','day','hour']
selcols4 = ['ip', 'mobile','app_channel','day','hour']    

selcols_list = [selcols0, selcols1, selcols2, selcols3, selcols4]



def do_same(save_name, train_df, which_dataset):
    print(train_df.info())

    # print('-------------------------------------------------------------------')
    # print('load day9_cat_combination...')
    # filename = input + 'day9_cat_combination.csv'
    # train_df_test = prepare_dataset(which_dataset, train_df, 
    #         filename, usecols=['mobile', 'mobile_app', 'mobile_channel', 'app_channel'])
    # print_memory()
    # print(train_df_test.info())
    # print(train_df_test.head())
    # del train_df_test; gc.collect()


    print('-------------------------------------------------------------------')
    print('load day9_cat_combination_numeric_category...')
    filename = input + 'day9_cat_combination_numeric_category.csv'
    train_df = prepare_dataset(which_dataset, train_df, 
            filename, usecols=['mobile', 'mobile_app', 'mobile_channel', 'app_channel'])
    print_memory()
    print(train_df.info())
    print(train_df.head())


    print('-------------------------------------------------------------------')
    print('load day9_day_hour_min...')
    filename = input + 'day9_day_hour_min.csv'
    train_df = prepare_dataset(which_dataset, train_df, 
            filename, usecols=['day', 'hour'])
    print_memory()
    print(train_df.info())
    print(train_df.head())


    print('-------------------------------------------------------------------')
    print('load day9_nextClick...')
    filename = input + 'day9_nextClick.csv'
    train_df = prepare_dataset(which_dataset, train_df, 
            filename, usecols=['nextClick', 'nextClick_shift'])
    print_memory()
    print(train_df.info())
    print(train_df.head())


    for apply_type in apply_type_list:
        for selcols in selcols_list:
            print('-------------------------------------------------------------------')
            print('merging...')
            print('select column:', selcols)
            print('apply type:', apply_type)
            filename, feature_name = get_filename(selcols, apply_type)
            train_df = prepare_dataset(which_dataset, train_df, filename, 
                    usecols=[feature_name])
            print_memory()
    print(train_df.info())
    print(train_df.head())

    
    train_df.to_pickle(save_name)

def do_train():
    print('-------------------------------------------------------------------')
    print('do TRAIN')
    print('-------------------------------------------------------------------')
    print('load dataset...')
    train_df, test_df = read_train_test('is_not_merged')
    del test_df; gc.collect()
    print_memory()
    which_dataset = 'train'
    save_name='train_day9'
    do_same(save_name, train_df, which_dataset)

def do_test():
    print('-------------------------------------------------------------------')
    print('do TEST')
    print('-------------------------------------------------------------------')
    print('load dataset...')
    train_df, test_df = read_train_test('is_not_merged')
    del train_df; gc.collect()
    print_memory()
    which_dataset = 'test'
    save_name='test_day9'
    do_same(save_name, test_df, which_dataset)

do_test()
do_train()

print (range(0))
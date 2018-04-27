import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

debug=2
print('debug', debug)

if not debug:
    print('=======================================================================')
    print('process on server...')
    print('=======================================================================')
else:
    print('=======================================================================')
    print('for testing only...')
    print('=======================================================================')

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
import datetime



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

TRAIN_HDF5 = 'train_day9.h5'
TEST_HDF5 = 'test_day9.h5'
# TEST_HDF5 = 'test_day123.h5'

DATATYPE_LIST_STRING = {
    'mobile'            : 'category',
    'mobile_app'        : 'category',
    'mobile_channel'    : 'category',
    'app_channel'       : 'category',
    }

if debug:
    PATH = '../debug_processed_day9/'        
else:
    PATH = '../processed_day9/'                
CAT_COMBINATION_FILENAME = PATH + 'day9_cat_combination.csv'
CAT_COMBINATION_NUMERIC_CATEGORY_FILENAME = PATH + 'day9_cat_combination_numeric_category.csv'
NEXTCLICK_FILENAME = PATH + 'day9_nextClick.csv'
TIME_FILENAME = PATH + 'day9_day_hour_min.csv'
IP_HOUR_RELATED_FILENAME = PATH + 'day9_ip_hour_related.csv'
TRAINSET_FILENAME = '../input/valid_day_9.csv'
NCHUNKS = 100000
if debug==1:
    NROWS=10000000
else: 
    NROWS =100    
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
SIZE_TEST = 18790469

def print_memory(print_string=''):
    print('Total memory in use ' + print_string + ': ', process.memory_info().rss/(2**30), ' GB')

def get_keys_h5(f):
    return [key for key in f.keys()]



SEED = 1988
frac = 0.7
process = psutil.Process(os.getpid())

if debug==2:
    start_point = 1 
    end_point = 1000
if debug==1:
    start_point = 1 
    end_point = 1000000
if debug==0:
    start_point = 1 
    end_point = 10000000


DATATYPE_DICT = {
    'count',     
    'nunique',   
    'cumcount',  
    'var'      ,
    'std'       ,
    'confRate' ,
    'nextclick' ,
    'nextClick',
    'nextClick_shift'
    }


def is_processed(feature):
    is_processed = True
    for key in DATATYPE_DICT:
        if key in feature:
            is_processed = False
    return is_processed            


def read_processed_h5(start_point, end_point, filename):
    with h5py.File(filename,'r') as hf:
        feature_list = list(hf.keys())
    train_df = pd.DataFrame()
    t0 = time.time()
    for feature in feature_list:
        if feature!='dump_later' and not is_processed(feature):
            print('>> adding', feature)
            train_df[feature] = pd.read_hdf(filename, key=feature,
                    start=start_point, stop=end_point)    
            print_memory()
    t1 = time.time()
    total = t1-t0
    print('total reading time:', total)
    return train_df


def draw_save_heatmap(start_point, end_point, filename):
    train_df = read_processed_h5(start_point, end_point, filename)
    print(train_df.info())
    sns.heatmap(train_df.corr(),annot=True,
            xticklabels=1, yticklabels=1, cmap='RdYlGn',linewidths=0.2)
    fig=plt.gcf()
    fig.set_size_inches(50,50)
    savename = yearmonthdate_string + '_' + str(start_point)  \
        + '_' + str(end_point) + '_' + filename + '_heatmap.png'
    plt.savefig(savename)
    # plt.show()

def read_processed_h5_full(start_point, end_point, filename):
    with h5py.File(filename,'r') as hf:
        feature_list = list(hf.keys())
    train_df = pd.DataFrame()
    t0 = time.time()
    for feature in feature_list:
        if feature!='dump_later':
            print('>> adding', feature)
            train_df[feature] = pd.read_hdf(filename, key=feature,
                    start=start_point, stop=end_point)    
            print_memory()
    t1 = time.time()
    total = t1-t0
    print('total reading time:', total)
    return train_df


def print_all_features(start_point, end_point, filename):
    train_df = read_processed_h5_full(start_point, end_point, filename)
    features = list(train_df)
    if debug: 
        for feature in  features: 
            print (feature)
    print(train_df)      
    temp_df = train_df[CATEGORY_LIST]
    print(temp_df)


def eda():
    end_point_list = [10000000]
    filename_list = [TRAIN_HDF5, TEST_HDF5]
    for filename in filename_list:
        for end_point in end_point_list:
            start_point = 1
            draw_save_heatmap(start_point, end_point, filename)

def find_predictors(filename, expected_num_feature):
    train_df = read_processed_h5(start_point, end_point, filename)
    predictors = []
    features = list(train_df)
    if debug: 
        for feature in  features: 
            print (feature)

    return predictors

# find_predictors(TRAIN_HDF5, 5)

print_all_features(0, 10, TRAIN_HDF5)


from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR


debug=2
print('debug', debug)

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

process = psutil.Process(os.getpid())

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

# DATATYPE_DICT = {
#     'count',     
#     'nunique',   
#     'cumcount',  
#     'var'      ,
#     'std'       ,
#     'confRate' ,
#     'nextclick' ,
#     'nextClick',
#     'nextClick_shift'
#     }

DATATYPE_DICT = {
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
        # if feature!='dump_later' and not is_processed(feature):
        if feature!='dump_later':
            print('>> adding', feature)
            train_df[feature] = pd.read_hdf(filename, key=feature,
                    start=start_point, stop=end_point)    
            print_memory()
    t1 = time.time()
    total = t1-t0
    print('total reading time:', total)
    return train_df


if debug==2:
    start_point = 0 
    end_point = 10
if debug==1:
    start_point = 0 
    end_point = 1000000
if debug==0:
    start_point = 0 
    end_point = 10000000


def find_predictors(filename, expected_num_feature):
    train_df = read_processed_h5(start_point, end_point, filename)
    predictors = []
    features = list(train_df)
    if debug: 
        for feature in  features: 
            print (feature)

    return predictors


# find_predictors(TRAIN_HDF5, 5)

train_df = read_processed_h5(start_point, end_point, TRAIN_HDF5)
print(train_df.info())



train_df = train_df.drop(['click_time'], axis=1)
print(train_df.info())
print(train_df.head())



y = train_df['is_attributed']
X = train_df.loc[:, train_df.columns != 'is_attributed']

# X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
estimator = SVR(kernel="linear")
selector = RFE(estimator, 15, step=1)
selector = selector.fit(X, y)

i=0
print('feature selected:')
for feature in train_df:
    if selector[i]:
        print(feature)
    i = i+1        
print(selector.support_ )
# print(selector.ranking_)

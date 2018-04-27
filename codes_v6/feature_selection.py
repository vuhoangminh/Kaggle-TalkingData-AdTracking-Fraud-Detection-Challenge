debug=1
print('debug', debug)
  

if debug==2:
    DATASET = 'full'  
else:    
    DATASET = 'day9'

print(DATASET)    

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


if debug==1:
    start_point = 0 
    end_point = 10
if debug==1:
    start_point = 0 
    end_point = 5000
if debug==0:
    start_point = 0 
    end_point = 1000


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
from sklearn.svm import SVC, SVR
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt


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

TRAIN_HDF5 = 'train_' + DATASET + '.h5'
TEST_HDF5 = 'test_' + DATASET + '.h5'

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
        if feature!='dump_later' and not is_processed(feature) or feature=='is_attributed' :
        # if feature!='dump_later':
            print('>> adding', feature)
            train_df[feature] = pd.read_hdf(filename, key=feature,
                    start=start_point, stop=end_point)    
            print_memory()
    t1 = time.time()
    total = t1-t0
    print('total reading time:', total)
    return train_df





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


def find_predictors(filename, expected_num_feature):
    train_df = read_processed_h5(start_point, end_point, filename)
    predictors = []
    features = list(train_df)
    if debug: 
        for feature in features and not is_processed(feature): 
            print (feature)
    return predictors


# find_predictors(TRAIN_HDF5, 5)

train_df = read_processed_h5(start_point, end_point, TRAIN_HDF5)
train_df = train_df.fillna(0)
print(train_df.info())


if 'click_time' in train_df:
    train_df = train_df.drop(['click_time'], axis=1)
print(train_df.info())
print(train_df.head())

print('>> prepare dataset...')
y = train_df['is_attributed']
X = train_df.loc[:, train_df.columns != 'is_attributed']
print_memory()

# t = time.time()
# do stuff
print('>> fit on {} samples and {} features...'.format(len(X), len(list(X))))
estimator = SVR(kernel="linear") 
print('done estimator')
selector = RFE(estimator, 20, step=1, verbose=1)
print('done selector')
selector = selector.fit(X, y)
print_memory()
# elapsed = time.time() - t
# print('processing time:', elapsed)


# i=0

print('========================================================================')
print('SUMMARY')
print('========================================================================')
print_memory()
features = list(X)
print(len(features), len(selector.ranking_))
print('feature selected:')
for i in range(len(features)):
    feature = features[i]
    if selector.support_[i]:
        print(feature, selector.ranking_[i])
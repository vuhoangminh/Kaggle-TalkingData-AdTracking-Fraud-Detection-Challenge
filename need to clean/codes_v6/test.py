import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import datetime







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



debug=0
print('debug', debug)
  

if debug==2:
    START_POINT = 0 
    END_POINT = 10
if debug==1:
    START_POINT = 0 
    END_POINT = 1000
if debug==0:
    START_POINT = 0 
    END_POINT = 3000


if debug==1:
    DATASET = 'day9'
else:    
    DATASET = 'full'

print(DATASET)   
TRAIN_HDF5 = 'train_' + DATASET + '.h5'
TEST_HDF5 = 'test_' + DATASET + '.h5'
if debug == 0:
    TRAIN_HDF5 = 'converted_' + TRAIN_HDF5
    TEST_HDF5 = 'converted_' + TEST_HDF5


DATATYPE_LIST_STRING = {
    'mobile'            : 'category',
    'mobile_app'        : 'category',
    'mobile_channel'    : 'category',
    'app_channel'       : 'category',
    }
      

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

def get_keys_h5(f):
    return [key for key in f.keys()]


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
        if feature!='dump_later':
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


# start_point = START_POINT
# end_point = END_POINT
# train_df = read_processed_h5(start_point, end_point, TRAIN_HDF5)
# hour = train_df['hour']
# train_df = train_df.fillna(0)
# print(train_df.info())
# print(train_df.head())
# print(hour.head())
# print(hour.dtype)

# with h5py.File(TRAIN_HDF5,  "a") as f:
#     del f['hour']
#     del f['day']
#     del f['min']

# with h5py.File(TEST_HDF5,  "a") as f:
#     del f['hour']
#     del f['day']
#     del f['min']

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


def read_and_write(which_dataset, filename):
    if which_dataset=='train':
        train_df = pd.read_csv('../input/train.csv', parse_dates=['click_time'],
            dtype=DATATYPE_LIST, usecols=['click_time'])   
    else:
        train_df = pd.read_csv('../input/test.csv', parse_dates=['click_time'],
            dtype=DATATYPE_LIST, usecols=['click_time'])   
    gp = pd.DataFrame()
    print('>> read min')
    gp['min'] = pd.to_datetime(train_df.click_time).dt.minute.astype('uint8')
    print('>> read hour')
    gp['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    print('>> read day')
    gp['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    print(len(train_df))

    print('>> write min')
    temp_min = pd.DataFrame()
    temp_min['min'] = gp['min']
    temp_min.to_hdf(filename, key='min', mode='a')

    print('>> write hour')
    temp_hour = pd.DataFrame()
    temp_hour['hour'] = gp['hour']
    temp_hour.to_hdf(filename, key='hour', mode='a')

    print('>> write day')
    temp_day = pd.DataFrame()
    temp_day['day'] = gp['day']
    temp_day.to_hdf(filename, key='day', mode='a')
    # return gp

# read_and_write('train', TRAIN_HDF5)
# read_and_write('test', TEST_HDF5)

train_df = pd.DataFrame()
train_df['hour'] = pd.read_hdf(TRAIN_HDF5, key='hour')
print(train_df)
print(train_df.describe())
print(train_df.dtypes)
print(train_df.isnull().sum())

test_df = pd.DataFrame()
test_df['hour'] = pd.read_hdf(TEST_HDF5, key='hour')
print(test_df)
print(test_df.describe())
print(test_df.dtypes)
print(test_df.isnull().sum())



# import glob
# PATH = '../processed_full/'
# files = glob.glob(PATH + "*.csv") 

# # for file in files:
# df = pd.DataFrame()
# df = pd.read_csv(PATH + 'full_day_hour_min.csv')
# print('------------------')
# print(PATH + 'full_day_hour_min.csv')
# print(len(df))
# print(df.info())

# with h5py.File(TRAIN_HDF5,'r') as hf:
#     feature_list = list(hf.keys())
# train_df = pd.DataFrame()
# t0 = time.time()


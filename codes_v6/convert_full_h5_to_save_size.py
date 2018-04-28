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


debug=0
print('debug', debug)
  

if debug==1:
    DATASET = 'day9'
else:    
    DATASET = 'full'

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


if debug==2:
    START_POINT = 0 
    END_POINT = 10
if debug==1:
    START_POINT = 0 
    END_POINT = 1000
if debug==0:
    START_POINT = 0 
    END_POINT = 100000000


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


process = psutil.Process(os.getpid())


TRAIN_HDF5 = 'train_' + DATASET + '.h5'
TEST_HDF5 = 'test_' + DATASET + '.h5'

TO_TRAIN_HDF5 = 'converted_' + TRAIN_HDF5
TO_TEST_HDF5 = 'converted_' + TEST_HDF5


def print_memory(print_string=''):
    print('Total memory in use ' + print_string + ': ', process.memory_info().rss/(2**30), ' GB')

def get_keys_h5(f):
    return [key for key in f.keys()]      

DATATYPE_DICT_CONVERT = [
    'count',
    'nunique',
    'cumcount'
]

def read_processed_h5_and_write(filename, tofilename):
    with h5py.File(filename,'r') as hf:
        feature_list = list(hf.keys())        
    with h5py.File(tofilename,'r') as hf_to:
        feature_list_to = list(hf_to.keys())            
    for feature in feature_list:
        if feature in feature_list_to:
            print('already added')
        else:            
            t0 = time.time()
            print('>> doing', feature)
            if feature!='dump_later' and feature != 'click_time':
                is_convert = False
                for key in DATATYPE_DICT_CONVERT:
                    if key in feature:
                        is_convert = True
                        print('need to convert', feature, 'to int to save memory')
                df_temp = pd.DataFrame()            
                print('reading...')
                df_temp[feature] = pd.read_hdf(filename, key=feature)
                if is_convert:
                    print('min anc max before:', df_temp[feature].min(), df_temp[feature].max())                        
                    df_temp = df_temp.fillna(0)
                    df_temp[feature] = df_temp[feature].astype('uint32')                        
                    print('min anc max after:', df_temp[feature].min(), df_temp[feature].max()) 

                print('saving')                
                df_temp.to_hdf(tofilename, key=feature, mode='a')    
                del df_temp; gc.collect()
                print_memory()
                t1 = time.time()
                total = t1-t0
                print('total reading time:', total)

print(TRAIN_HDF5, TO_TRAIN_HDF5)
print(TEST_HDF5, TO_TEST_HDF5)
print('start')
read_processed_h5_and_write(TRAIN_HDF5, TO_TRAIN_HDF5)
read_processed_h5_and_write(TEST_HDF5, TO_TEST_HDF5)
print('done')

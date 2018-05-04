debug = 1
frac = 0.01

# debug = 0
# frac = 0.5

OPTION = 18

import pandas as pd
import numpy as np

import keras
import tensorflow as tf
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4}) 
# config = tf.ConfigProto( device_count = {'GPU': 0, 'CPU': 4} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

# if debug:
#     os.environ['OMP_NUM_THREADS'] = '4'
# else:    
#     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Reshape
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras.layers import LSTM, LeakyReLU
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from sklearn.cross_validation import train_test_split
import h5py
import os, time
from keras.backend.tensorflow_backend import set_session
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import sparse
from sklearn.metrics import roc_auc_score
import gc
import psutil
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

SEED = 1988
process = psutil.Process(os.getpid())

if debug==1:
    DATASET = 'day9'
else:    
    DATASET = 'full'

TRAIN_HDF5 = 'train_' + DATASET + '.h5'
TEST_HDF5 = 'test_' + DATASET + '.h5'
if debug == 0:
    TRAIN_HDF5 = 'converted_' + TRAIN_HDF5
    TEST_HDF5 = 'converted_' + TEST_HDF5

TRAIN_HDF5 = '../codes_v6/' + TRAIN_HDF5    
TEST_HDF5 = '../codes_v6/' + TEST_HDF5

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
    'ip_device_os_nunique_app', # X8
    'ip_os_device_app_nextclick',
    'ip_day_hour_count_channel',
    'ip_app_count_channel',
    'ip_app_os_count_channel',
    ]    


PREDICTORS18 = [
    # core 9
    'app', 'os', 'device', 'channel', 'hour',
    'ip_os_device_app_nextclick',
    'ip_device_os_nunique_app',
    'ip_nunique_channel',
    'ip_nunique_app', 
    # add
    'ip_nunique_device',
    'ip_cumcount_os',
    'ip_device_os_nextclick',
    'ip_os_device_channel_app_nextclick',
    'ip_app_os_count_channel',
    'ip_count_app',
    'app_count_channel',
    'ip_device_os_nunique_channel',
    'ip_nextclick',
    'ip_channel_nextclick'
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


def get_predictors():
    if OPTION==3:
        predictors = PREDICTORS3
    if OPTION==18:
        predictors = PREDICTORS18     
    print('------------------------------------------------')
    print('predictors:')
    for feature in predictors:
        print (feature)
    print('number of features:', len(predictors))            
    return predictors 

def get_categorical(predictors):
    predictors = get_predictors()
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


NEW_FEATURE = [    
    'channel_count_app',
    'ip_count_app',
    'ip_app_count_os',
    'ip_count_device',
    'app_count_channel',
    'ip_device_os_nunique_channel',
    'channel_nunique_app'
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
            print_memory()
    train_df = train_df.fillna(0)            
    t1 = time.time()
    total = t1-t0
    print('total reading time:', total)
    print(train_df.info())   
    return train_df

#build model

# 2 layers
print('>> prepare predictors...')
predictors = get_predictors()
categorical = get_categorical(predictors)
target = TARGET

print('>> read train...')
print('frac:',frac)
train_df = read_processed_h5(TRAIN_HDF5, predictors+target)
if frac<1:
    train_df = train_df.sample(frac=frac, random_state = SEED)
print_memory()
train_label = train_df[target]
train_label = train_df[target].values.astype('int').flatten()

train_df = train_df.drop(target, axis=1)
train_cat = train_df[categorical].as_matrix()

print('>> read test...')
test_df = read_processed_h5(TEST_HDF5, predictors+['click_id'])
test_cat = test_df[categorical].as_matrix()
test_id = test_df['click_id']
test_df = test_df.drop('click_id', axis=1)
print_memory()

# print('>> stack...')
# traintest_cat = np.vstack((train_cat, test_cat))
# traintest_cat = pd.DataFrame(traintest_cat, columns=categorical)
# print(traintest_cat.shape)
# print_memory()

# print ('>> label encoder')
# from sklearn.preprocessing import LabelEncoder
# train_df[categorical].apply(LabelEncoder().fit_transform)
# test_df[categorical].apply(LabelEncoder().fit_transform)
# print('train:', train_df.head())
# print('test:', test_df.head())
# print_memory()

print('>> prepare dataset...')
train_df = train_df.as_matrix()
test_df = test_df.as_matrix()
train_list = train_df
test_list = test_df
del train_df, test_df; gc.collect()
if debug: print(train_list); print(test_list)
print_memory()

X_test = test_list
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))


import numpy as np
from numpy.testing import assert_allclose
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import ModelCheckpoint

model = load_model('model/weights_improvement_1percent_option18.h5')

print('>> predict...')
cv_pred = model.predict_proba(x=X_test, batch_size=32, verbose=1)[:, 0]

print('--------------------------------------------------------------------') 
sub = pd.DataFrame()
sub['click_id'] = test_id
subfilename = yearmonthdate_string + '_' + str(len(predictors)) + \
            'features_dl_cv_' + str(int(100*frac)) + \
            'percent_full.csv.gz'

print(">> Predicting...")
sub['is_attributed'] = cv_pred * 1.
print("writing...")
sub.to_csv(subfilename,index=False,compression='gzip')
print("done...")
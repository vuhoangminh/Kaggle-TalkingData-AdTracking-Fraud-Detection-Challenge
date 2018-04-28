debug = 2
frac = 1

import pandas as pd
import numpy as numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Reshape
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras.layers import LSTM, LeakyReLU
from keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn.cross_validation import train_test_split
import h5py
import os, time
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score
from keras.wrappers.scikit_learn import KerasRegressor

os.environ['OMP_NUM_THREADS'] = '4'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import gc

import psutil
import datetime

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

PREDICTORS = ['ip', 'app', 'device', 'os', 'channel', 'hour',
    'ip_nunique_channel',
    'ip_device_os_cumcount_app',
    'ip_day_nunique_hour',
    'ip_nunique_app',
    'ip_app_nunique_os',
    'ip_nunique_device',
    'app_nunique_channel',
    'ip_cumcount_os',
    'ip_device_os_nunique_app',
    'ip_os_device_app_nextclick',
    'ip_day_hour_count_channel',
    'ip_app_count_channel',
    'ip_app_os_count_channel'
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
    predictors = PREDICTORS
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
            else:
                train_df[feature] = pd.read_hdf(filename, key=feature)                                 
            print_memory()
    t1 = time.time()
    total = t1-t0
    print('total reading time:', total)
    print(train_df.info())   
    return train_df



#build model

# 2 layers
predictors = get_predictors()
target = TARGET
train_df = read_processed_h5(TRAIN_HDF5, predictors+target)
train_df = train_df.sample(frac=frac, random_state = SEED)
# train_df, val_df = train_test_split(train_df, test_size=0.33, random_state=SEED)


output_file_name='CNN_2_relu'

batch_size= 2048
epochs = 100
step_size = len(train_df)
nb_features = len(predictors)

# #split training validation
# training_size = len(train_df)
# training_datas = train_df[predictors]
# training_labels = train_df[target]
# validation_datas = val_df[predictors]
# validation_labels = val_df[target]


def baseline_model():
    model = Sequential()
    model.add(Dense(10, input_dim=19, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

X = train_df[predictors]
y = train_df[target]
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, 
        batch_size=100, verbose=2)
kfold = KFold(n_splits=10, random_state=SEED)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

estimator.fit(X, y)
prediction = estimator.predict(X)
# accuracy_score(y, prediction)
debug = 0
frac = 0.5

import pandas as pd
import numpy as np

import keras
import tensorflow as tf
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)


from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Reshape
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras.layers import LSTM, LeakyReLU
from keras.callbacks import CSVLogger, ModelCheckpoint
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
from sklearn.preprocessing import StandardScaler
from scipy import sparse
from sklearn.metrics import roc_auc_score
import gc
import psutil
import datetime

if debug:
    os.environ['OMP_NUM_THREADS'] = '4'
else:    
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


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
PREDICTORS = [
    'app', 'device', 'os', 'channel', 'hour',
    'ip_nunique_channel',   # X0
    'ip_device_os_cumcount_app',
    'ip_day_nunique_hour',
    'ip_nunique_app',
    'ip_app_nunique_os',
    'ip_nunique_device',
    'app_nunique_channel',
    # 'ip_cumcount_os', # X6
    'ip_device_os_nunique_app', # X8
    'ip_os_device_app_nextclick',
    'ip_day_hour_count_channel',
    'ip_app_count_channel',
    'ip_app_os_count_channel',
    # 'ip_day_channel_var_hour', # miss
    # 'ip_app_os_var_hour',
    # 'ip_app_channel_var_day',
    # 'ip_app_channel_mean_hour'
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
print('>> prepare predictors...')
predictors = get_predictors()
categorical = get_categorical(predictors)
target = TARGET

print('>> read train...')
train_df = read_processed_h5(TRAIN_HDF5, predictors+target)
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

print('>> stack...')
traintest_cat = np.vstack((train_cat, test_cat))
traintest_cat = pd.DataFrame(traintest_cat, columns=categorical)
print(traintest_cat.shape)
print_memory()

print ('>> label encoder')
from sklearn.preprocessing import LabelEncoder
train_df[categorical].apply(LabelEncoder().fit_transform)
test_df[categorical].apply(LabelEncoder().fit_transform)
print('train:', train_df.head())
print('test:', test_df.head())
print_memory()

print('>> prepare dataset...')
train_df = train_df.as_matrix()
test_df = test_df.as_matrix()
train_list = train_df
test_list = test_df
del train_df, test_df; gc.collect()
if debug: print(train_list); print(test_list)
print_memory()

print('>> scale standard')
scaler = StandardScaler()
scaler.fit(np.concatenate((train_list, test_list), axis=0))
train_list = scaler.transform(train_list)
test_list = scaler.transform(test_list)
print_memory()

X = train_list
del train_list; gc.collect
X_test = test_list
del test_list; gc.collect

gc.collect()
print(X.shape, X_test.shape)
np.set_printoptions(precision=3)
print(X)



print('>> init network')
output_file_name='CNN_2_relu'

batch_size= 2048
epochs = 100
# step_size = len(train_df)
nb_features = len(predictors)

def baseline_model():
    model = Sequential()

    model.add(Dense(512, input_dim=X.shape[1], init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.9))

    model.add(Dense(64, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.8))

    model.add(Dense(1, init='he_normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
    return (model)


# print(train_label)

NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=218)

cv_train = np.zeros(len(train_label))
cv_pred = np.zeros(len(test_id))
num_seeds = 2
print('=========================================================================')
print('>> start training')
for s in range(num_seeds):
    np.random.seed(s)
    print('--------------------------------------------------------')
    print('seed', s)
    print('--------------------------------------------------------')
    for (inTr, inTe) in kfold.split(X, train_label):
        print('>> split')
        xtr = X[inTr]
        ytr = train_label[inTr]
        xte = X[inTe]
        yte = train_label[inTe]

        print('>> fitting...')
        model = baseline_model()
        model.fit(xtr, ytr, epochs=21, batch_size=2048*64, verbose=1, validation_data=[xte, yte])
        print('>> valid...')
        cv_train[inTe] += model.predict_proba(x=xte, batch_size=2048, verbose=1)[:, 0]
        print('>> predict...')
        cv_pred += model.predict_proba(x=X_test, batch_size=2048, verbose=1)[:, 0]
    print(s)
    print(roc_auc_score(train_label, cv_train))


print('--------------------------------------------------------------------') 
sub = pd.DataFrame()
sub['click_id'] = test_id
subfilename = yearmonthdate_string + '_' + str(len(predictors)) + \
            'features_dl_cv_' + str(int(100*frac)) + \
            'percent_full.csv.gz'

print(">> Predicting...")
sub['is_attributed'] = cv_pred * 1./ (NFOLDS * num_seeds)
print("writing...")
sub.to_csv(subfilename,index=False,compression='gzip')
print("done...")
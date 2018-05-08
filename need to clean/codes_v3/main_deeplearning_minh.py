# good day, my friends
# in this kernel we try to continue development of our DL models
# thanks for people who share their works. i hope together we can create smth interest

# https://www.kaggle.com/baomengjiao/embedding-with-neural-network
# https://www.kaggle.com/gpradk/keras-starter-nn-with-embeddings
# https://www.kaggle.com/pranav84/lightgbm-fixing-unbalanced-data-auc-0-9787
# https://www.kaggle.com/rteja1113/lightgbm-with-count-features
# https://www.kaggle.com/knowledgegrappler/a-simple-nn-solution-with-keras-0-48611-pl
# https://www.kaggle.com/isaienkov/rnn-with-keras-ridge-sgdr-0-43553
# https://www.kaggle.com/valkling/mercari-rnn-2ridge-models-with-notes-0-42755/versions#base=2202774&new=2519287


#======================================================================================
# we continue our work started in previos kernel "Deep learning support.."
# + we will try to find a ways which can help us increase specialisation of neural network on our task
# + we will try to work with different architect decisions for neural networks
# if you need a details about what we try to create follow the comments

print ('Good luck')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['OMP_NUM_THREADS'] = '4'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import gc

import tensorflow as tf
import keras.backend as K
import keras
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
import keras.models 
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, SpatialDropout1D, Conv1D, GaussianDropout
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.optimizers import Adam
import psutil

process = psutil.Process(os.getpid())

debug=0
nrows=184903891-1
# nchunk=180000000
nchunk=10000
val_size=int(nchunk*0.1)

frac = 1

# frm=nrows-75000000
frm=10
if debug:
    frm=0
    nchunk=100000
    val_size=10000

to=frm+nchunk

print('========================================================================')
print ('preparation....')
print('========================================================================')
# subfilename = boosting_type + '_sub_it_' + str(frac) + '_%d_%d_31_5.csv.gz'%(frm,to)
subfilename = 'dl_' + str(int(100*frac)) + 'percent_%d_%d.csv.gz'%(frm,to)
print('submission file name:', subfilename)
print('fraction:', frac)

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32',
        }


print('doing nextClick')
predictors=[]

new_feature = 'nextClick'
filename='nextClick_%d_%d.csv'%(frm,to)

predictors.append(new_feature)
predictors.append(new_feature+'_shift')


target = 'is_attributed'
predictors.extend(['app','device','os', 'channel', 'hour', 'day', 
                'ip_tcount', 'ip_tchan_count', 'ip_app_count',
                'ip_app_os_count', 'ip_app_os_var',
                'ip_app_channel_var_day','ip_app_channel_mean_hour'])
categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']
naddfeat=9
for i in range(0,naddfeat):
    predictors.append('X'+str(i))
    
print('predictors',predictors)

save_name='test_%d_%d'%(frm,to)
test_df = pd.read_pickle(save_name)
print('Total memory in use after reading test: ', process.memory_info().rss/(2**30), ' GB\n')
save_name='val_%d_%d'%(frm,to)
val_df = pd.read_pickle(save_name)
if frac < 1:
    val_df = val_df.sample(frac=frac)
print('Total memory in use after reading val: ', process.memory_info().rss/(2**30), ' GB\n')
save_name='train_%d_%d'%(frm,to)
train_df = pd.read_pickle(save_name)
if frac < 1:
    train_df = train_df.sample(frac=frac)
print('Total memory in use after reading train: ', process.memory_info().rss/(2**30), ' GB\n')

print("train size: ", len(train_df))
print("valid size: ", len(val_df))
print("test size : ", len(test_df))

print(train_df.head())


# ========================================================================

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')
y_train = train_df['is_attributed'].values

# del test_df; gc.collect()

print('========================================================================')
print ('neural network....')
print('========================================================================')



def get_keras_data(dataset):
    X = {
        'app': np.array(dataset.app),
        'channel': np.array(dataset.channel),
        'device': np.array(dataset.device),
        'os': np.array(dataset.os),
        'hour': np.array(dataset.hour),
        'day': np.array(dataset.day),
        'nextClick': np.array(dataset.nextClick),
        'nextClick_shift': np.array(dataset.nextClick_shift),
        'ip_tcount': np.array(dataset.ip_tcount),
        'ip_tchan_count': np.array(dataset.ip_tchan_count),
        'ip_app_count': np.array(dataset.ip_app_count),
        'ip_app_os_count': np.array(dataset.ip_app_os_count),
        'ip_app_os_var': np.array(dataset.ip_app_os_var),
        'ip_app_channel_var_day': np.array(dataset.ip_app_channel_var_day),
        'ip_app_channel_mean_hour': np.array(dataset.ip_app_channel_mean_hour),
        'X0': np.array(dataset.X0),
        'X1': np.array(dataset.X1),
        'X2': np.array(dataset.X2),
        'X3': np.array(dataset.X3),
        'X4': np.array(dataset.X4),
        'X5': np.array(dataset.X5),
        'X6': np.array(dataset.X6),
        'X7': np.array(dataset.X7),
        'X8': np.array(dataset.X8)
    }
    return X
# train_df = get_keras_data(train_df)
# test_df = get_keras_data(test_df)

# Select the columns to use for prediction in the neural network
prediction_var = predictors
X = train_df[prediction_var].values
Y = train_df.is_attributed.values


from keras.optimizers import SGD
model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=24))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(X, Y,
          epochs=20,
          batch_size=128)


# batch_size = 65536
# epochs = 2
# exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
# steps = int(len(list(train_df)[0]) / batch_size) * epochs
# lr_init, lr_fin = 0.0013, 0.0001
# lr_decay = exp_decay(lr_init, lr_fin, steps)
# optimizer_adam = Adam(lr=lr_init, decay=lr_decay)
# optimizer_adam_nodecay = Adam(lr=lr_init)
# model.compile(loss='binary_crossentropy',optimizer=optimizer_adam_nodecay,metrics=['accuracy'])

# model.summary()

# class_weight = {0:.01,1:.70} # magic
# model.fit(train_df, y_train, batch_size=batch_size, epochs=epochs, class_weight=class_weight, shuffle=True, verbose=2)
# del train_df, y_train; gc.collect()
# model.save('dl_model.h5')

# print("predicting....")
# sub['is_attributed'] = model.predict(test_df, batch_size=batch_size, verbose=1)
# del test_df; gc.collect()
# print("writing....")
# sub.to_csv(subfilename,index=False)
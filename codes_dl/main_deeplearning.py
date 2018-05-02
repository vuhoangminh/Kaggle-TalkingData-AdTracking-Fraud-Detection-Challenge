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

def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score

# FROM https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/41108
def jacek_auc(y_true, y_pred):
   score, up_opt = tf.metrics.auc(y_true, y_pred)
   #score, up_opt = tf.contrib.metrics.streaming_auc(y_pred, y_true)    
   K.get_session().run(tf.local_variables_initializer())
   with tf.control_dependencies([up_opt]):
       score = tf.identity(score)
   return score

# FROM https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/41015
# AUC for a binary classifier
def discussion41015_auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)

#---------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N

#----------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P


import keras
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
import keras.models 

import keras
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, SpatialDropout1D, Conv1D
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.optimizers import Adam


path = '../input/'
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
print('load train....')
train_set_name = 'train_0'
train_df = pd.read_pickle(train_set_name)


print('load test....')
test_set_name = 'test_0'
test_df = pd.read_pickle(test_set_name)
len_train = len(train_df)
sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')
# train_df=train_df.append(test_df)
y_train = train_df['is_attributed'].values
print ("Size train:", len(train_df))
print ("Size test:", len(test_df))
# del test_df; gc.collect()

print ('neural network....')

max_app = np.max([train_df['app'].max(), test_df['app'].max()])+1
max_ch = np.max([train_df['channel'].max(), test_df['channel'].max()])+1
max_dev = np.max([train_df['device'].max(), test_df['device'].max()])+1
max_os = np.max([train_df['os'].max(), test_df['os'].max()])+1
max_h = np.max([train_df['hour'].max(), test_df['hour'].max()])+1
max_d = np.max([train_df['day'].max(), test_df['day'].max()])+1
max_wd = np.max([train_df['wday'].max(), test_df['wday'].max()])+1
max_qty = np.max([train_df['qty'].max(), test_df['qty'].max()])+1
max_c1 = np.max([train_df['ip_app_count'].max(), test_df['ip_app_count'].max()])+1
max_c2 = np.max([train_df['ip_app_os_count'].max(), test_df['ip_app_os_count'].max()])+1
def get_keras_data(dataset):
    X = {
        'app': np.array(dataset.app),
        'ch': np.array(dataset.channel),
        'dev': np.array(dataset.device),
        'os': np.array(dataset.os),
        'h': np.array(dataset.hour),
        'd': np.array(dataset.day),
        'wd': np.array(dataset.wday),
        'qty': np.array(dataset.qty),
        'c1': np.array(dataset.ip_app_count),
        'c2': np.array(dataset.ip_app_os_count)
    }
    return X
train_df = get_keras_data(train_df)

predictors =  ['app', 'channel', 'device', 'os', 'hour', 'day', 'wday', 'qty', 'ip_app_count', 'ip_app_os_count']

emb_n = 50
dense_n = 1000
embids = predictors
# get the max of each code type
embmaxs = dict((col, np.max([train_df[col].max(), test_df[col].max()])+1) for col in embids)
def get_keras_data(dataset):
    X = dict((col, np.array(dataset[col])) for col in embids)
    return X
# Build the inputs, embeddings and concatenate them all for each column
emb_inputs = dict((col, Input(shape=[1], name = col))  for col in embids)
emb_model  = dict((col, Embedding(embmaxs[col], emb_n)(emb_inputs[col])) for col in embids)
fe = concatenate([(emb_) for emb_ in emb_model.values()])
# Rest of the model
s_dout = SpatialDropout1D(0.2)(fe)
fl1 = Flatten()(s_dout)
conv = Conv1D(100, kernel_size=4, strides=1, padding='same')(s_dout)
fl2 = Flatten()(conv)
concat = concatenate([(fl1), (fl2)])
x = Dropout(0.2)(Dense(dense_n,activation='relu')(concat))
x = Dropout(0.2)(Dense(dense_n,activation='relu')(x))
outp = Dense(1,activation='sigmoid')(x)
model = Model(inputs=[inp for inp in emb_inputs.values()], outputs=outp)



batch_size = 2048*64
epochs = 100
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(list(train_df)[0]) / batch_size) * epochs
lr_init, lr_fin = 0.002, 0.0002
lr_decay = exp_decay(lr_init, lr_fin, steps)
optimizer_adam = Adam(lr=0.002, decay=lr_decay)
# model.compile(loss='binary_crossentropy',optimizer=optimizer_adam,metrics=['accuracy'])
model.compile(loss='binary_crossentropy',optimizer=optimizer_adam,metrics=[jacek_auc])

model.summary()

print("training....")
class_weight = {0:.01,1:.99} # magic

filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_jacek_auc', verbose=1, save_best_only=True, mode='max')

# check 5 epochs
early_stop = EarlyStopping(monitor='val_jacek_auc', patience=5, mode='max') 
callbacks_list = [checkpoint, early_stop]

model.fit(train_df, y_train, batch_size=batch_size, epochs=epochs, class_weight=class_weight, validation_split=0.3, shuffle=True, verbose=1, callbacks=callbacks_list)
del train_df, y_train; gc.collect()
model.save_weights('imbalanced_data.h5')

# test_df.drop(['click_id', 'click_time','ip','is_attributed'],1,inplace=True)
# test_df = get_keras_data(test_df)

print("predicting....")
sub['is_attributed'] = model.predict(test_df, batch_size=batch_size, verbose=1)
del test_df; gc.collect()
print("writing....")
sub.to_csv('imbalanced_data.csv',index=False)


# write now i don't use validation set
# since the data is imbalanced i can't understand how we can separate data the right way
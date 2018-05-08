"""
Adding improvements inspired from:
Ravi Teja's fe script: https://www.kaggle.com/rteja1113/lightgbm-with-count-features?scriptVersionId=2815638
"""

import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
import gc
import pickle
from sklearn.preprocessing import LabelEncoder

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


TRAINSAMPLE = 180000000
# TRAINSAMPLE = 1000
NROWS = 30000000
TRAINROWS = 90000000
# NROWS = 300
num_split = int(TRAINSAMPLE/NROWS)
print (num_split)


def load_write(iSplit):
    print('loading train data...')

    train_df = pd.read_csv(path+"train.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
    train_df = train_df.sample(frac=0.5)
    print ("len of train_df: ", len(train_df))

    # train_df_full = pd.read_csv(path+"train.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
    # print ("len of full: ", len(train_df_full))
    # train_df, train_df_delete = train_test_split(train_df_full, test_size=NROWS/TRAINSAMPLE)
    # del train_df_delete, train_df_full
    # gc.collect()
    # print ("len of sampling: ", len(train_df_full))


    print('loading test data...')
    test_df = pd.read_csv(path+"test.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    print ("len of test: ", len(test_df))
    gc.collect()

    len_train = len(train_df)
    train_df=train_df.append(test_df)
    del test_df
    gc.collect()

    # if iSplit>0:
    #     train_df = pd.read_csv(path+"train.csv", skiprows=range(1,skip_rows), nrows=NROWS, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
    # else:
    #     train_df = pd.read_csv(path+"train.csv", nrows=NROWS, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
            
    gc.collect()

    print('Extracting new features...')
    train_df['min'] = pd.to_datetime(train_df.click_time).dt.minute.astype('uint8')
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    train_df['wday']  = pd.to_datetime(train_df.click_time).dt.dayofweek.astype('uint8')
    print(train_df.head())

    print('grouping by ip alone....')
    gp = train_df[['ip','channel']].groupby(by=['ip'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ipcount'})
    train_df = train_df.merge(gp, on=['ip'], how='left')
    del gp; gc.collect()
    print('grouping by ip-day-hour combination....')
    gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'qty'})
    train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')
    del gp; gc.collect()
    print('group by ip-app combination....')
    gp = train_df[['ip','app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
    train_df = train_df.merge(gp, on=['ip','app'], how='left')
    del gp; gc.collect()
    print('group by ip-app-os combination....')
    gp = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
    train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
    del gp; gc.collect()
    print("vars and data type....")
    train_df['ipcount'] = train_df['qty'].astype('uint32')
    train_df['qty'] = train_df['qty'].astype('uint16')
    train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
    train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')
    print("label encoding....")

    train_df[['app','device','os', 'channel', 'hour', 'day', 'wday']].apply(LabelEncoder().fit_transform)
    # train_df[['app','device','os', 'channel', 'hour', 'day']].apply(LabelEncoder().fit_transform)
    print ('final part of preparation....')

    # train_df = train_df.drop(['click_time', 'wday', 'day'], axis=1)
    # train_df = train_df.drop(['click_time', 'day'], axis=1)
    train_df = train_df.drop(['click_time'], axis=1)
    
    print(train_df.head())
    print("train size: ", len(train_df))


    test_df = train_df[len_train:]
    print ("len of test: ", len(test_df))
    train_df = train_df[:len_train]
    print ("len of train: ", len(train_df))

    save_name = 'train_' + str(iSplit) 
    print("save to: ", save_name)
    train_df.to_pickle(save_name)
    del train_df
    gc.collect()

    save_name = 'test_' + str(iSplit) 
    print("save to: ", save_name)
    test_df.to_pickle(save_name)
    del test_df
    gc.collect()
    


for iSplit in range(3):
    print('Processing split', iSplit+1)
    skip_rows = iSplit*NROWS
    print (skip_rows)
    load_write(iSplit)    
"""
Adding improvements inspired from:
Ravi Teja's fe script: https://www.kaggle.com/rteja1113/lightgbm-with-count-features?scriptVersionId=2815638
"""

import pandas as pd
import time
import numpy as np
from numpy import random
from sklearn.cross_validation import train_test_split
import json
import lightgbm as lgb
import gc



path = 'D:/Users/RD/Documents/GitHub/Kaggle-datasets/Kaggle-TalkingData-AdTracking-Fraud-Detection-Challenge/input/'

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

def eda():
    # test
    test_set_name = 'test'
    test_df_full = pd.read_pickle(test_set_name)
    min_click_time = test_df_full['click_time'].min()
    max_click_time = test_df_full['click_time'].max()
    print (min_click_time)
    print (max_click_time)

    # train
    for iSplit in range(6):
        print (iSplit)   
        train_set_name = 'train_' + str(iSplit)
        print (train_set_name)
        if train_set_name == 'train_0':
            train_df_full = pd.read_pickle(train_set_name)
            min_click_time = train_df_full['click_time'].min()
            max_click_time = train_df_full['click_time'].max()
            print (min_click_time)
            print (max_click_time)
            del train_df_full
            gc.collect()

        else:
            train_df_full = pd.read_pickle(train_set_name)
            if min_click_time > train_df_full['click_time'].min():
                min_click_time = train_df_full['click_time'].min()
            if max_click_time < train_df_full['click_time'].max():
                max_click_time = train_df_full['click_time'].max()
            print (min_click_time)
            print (max_click_time)
            del train_df_full
            gc.collect()                


    # print(train_df_full['click_time'].min())
    # print(train_df_full['click_time'].max())
        



eda()

print ('done')

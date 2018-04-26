"""
A non-blending lightGBM model that incorporates portions and ideas from various public kernels
This kernel gives LB: 0.977 when the parameter 'debug' below is set to 0 but this implementation requires a machine with ~32 GB of memory
"""

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

process = psutil.Process(os.getpid())
# boosting_type = 'rf'
boosting_type = 'gbdt'
# boosting_type = 'dart'
frac = 1
# num_leaves = 31
# max_depth = 6
num_leaves = 38
max_depth = 21
noday = True

frm=1000
to=1001000

# frm=10
# to=180000010

debug=0

def minVal(x):
    return pd.Series(index=['min','idx'],data=[x.min(),x.idxmin()])

def maxVal(x):    
    return pd.Series(index=['max','idx'],data=[x.max(),x.idxmax()])

# save_name='train_%d_%d'%(frm,to)
save_name='train_%d_%d'%(frm,to)
train_df = pd.read_pickle(save_name)
train_df = train_df.sample(frac=frac)
print('Total memory in use after reading train: ', process.memory_info().rss/(2**30), ' GB\n')
print("train size: ", len(train_df))
train_df = train_df.fillna(0)
print(train_df.info())

# x = train_df.apply(minVal)
# print(x)
# x = train_df.apply(maxVal)
# print(x)

print('app unique:',len(train_df.app.unique()))
print('channel unique:',len(train_df.channel.unique()))
print('device unique:',len(train_df.device.unique()))
print('os unique:',len(train_df.os.unique()))
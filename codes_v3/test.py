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

# frm=1000
# to=1001000

frm=10
to=180000010

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


print('drop click_id...')
train_df = train_df.drop(['click_id'],axis=1)
print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')

# print('drop category, epochtime...')
# train_df = train_df.drop(['category'],axis=1)
# train_df = train_df.drop(['epochtime'],axis=1)
# print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')

print('convert to save memory...')
train_df['is_attributed'] = train_df['is_attributed'].astype('uint8')
print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')

print('convert X0-X8 to save memory...')
for i in range(0,9):
    x = 'X' + str(i)
    print('convert', x)
    train_df[x] = train_df[x].astype('uint32')
print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')    


print('convert to save memory...')
# train_df['nextClick_shift'] = train_df['nextClick_shift'].astype('int32')
train_df['ip_tchan_count'] = train_df['ip_tchan_count'].astype('uint32')
train_df['ip_app_os_var'] = train_df['ip_app_os_var'].astype('uint32')
train_df['ip_app_channel_var_day'] = train_df['ip_app_channel_var_day'].astype('uint32')
train_df['ip_app_channel_mean_hour'] = train_df['ip_app_channel_mean_hour'].astype('uint32')
print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')

# x = train_df.apply(minVal)
# print(x)

# x = train_df.apply(maxVal)
# print(x)

print(train_df.info())

print('saving...')
save_name='train_%d_%d_reduced'%(frm,to)
train_df.to_pickle(save_name)

print("val size: ", len(train_df))




# print(train_df.loc[train_df['nextClick_shift'].idxmax()])

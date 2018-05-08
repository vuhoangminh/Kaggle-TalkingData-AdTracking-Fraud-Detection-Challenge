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

debug=0
frm=0
if debug:
    frm=0
    nchunk=100000
    val_size=10000
to=1

def DO(frm,to,fileno):
    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32',
            }

    print('loading train data...')
    train_df = pd.read_csv("../input/train_day_7_8.csv.gz", parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
    print('len of train:', len(train_df))
    print('Total memory in use after reading train: ', process.memory_info().rss/(2**30), ' GB\n')    

    print('loading val data...')
    val_df = pd.read_csv("../input/valid_day_9.csv.gz", parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
    print('len of val:', len(val_df))
    print('Total memory in use after reading val: ', process.memory_info().rss/(2**30), ' GB\n')             

    print('loading test data...')
    if debug:
        test_df = pd.read_csv("../input/test.csv", nrows=10000, parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    else:
        # test_df = pd.read_csv("../input/test.csv", nrows=10000, parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
        test_df = pd.read_csv("../input/test.csv.gz", parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    print('len of val:', len(test_df))        
    print('Total memory in use after reading test: ', process.memory_info().rss/(2**30), ' GB\n')         

    train_df=train_df.append(val_df)
    del val_df; gc.collect()
    train_df=train_df.append(test_df)
    del test_df; gc.collect()
    
    print('Extracting new features...')
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    print('Total memory in use after extracting hour, day: ', process.memory_info().rss/(2**30), ' GB\n')  
    gc.collect()
    
    print('grouping by ip-day-hour combination...')
    gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_tcount'})
    print("saving...")
    filename = 'ip_day_hour_%d_%d.csv'%(frm,to)
    gp.to_csv(filename,index=False)
    del gp
    gc.collect()
  
    print('grouping by ip-app combination...')
    gp = train_df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
    print("saving...")
    filename = 'ip_app_%d_%d.csv'%(frm,to)
    gp.to_csv(filename,index=False)
    del gp
    gc.collect()

    print('grouping by ip-app-os combination...')
    gp = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
    print("saving...")
    filename = 'ip_app_os_%d_%d.csv'%(frm,to)
    gp.to_csv(filename,index=False)
    del gp
    gc.collect()

    # Adding features with var and mean hour (inspired from nuhsikander's script)
    print('grouping by ip_day_chl_var_hour')
    gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','channel'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_tchan_count'})
    print("saving...")
    filename = 'ip_day_chl_var_hour_%d_%d.csv'%(frm,to)
    gp.to_csv(filename,index=False)
    del gp
    gc.collect()

    print('grouping by ip_app_os_var_hour')
    gp = train_df[['ip','app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_app_os_var'})
    print("saving...")
    filename = 'ip_app_os_var_hour_%d_%d.csv'%(frm,to)
    gp.to_csv(filename,index=False)
    del gp
    gc.collect()

    print('grouping by ip_app_channel_var_day')
    gp = train_df[['ip','app', 'channel', 'day']].groupby(by=['ip', 'app', 'channel'])[['day']].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_var_day'})
    print("saving...")
    filename = 'ip_app_channel_var_day_%d_%d.csv'%(frm,to)
    gp.to_csv(filename,index=False)
    del gp
    gc.collect()

    print('grouping by ip_app_chl_mean_hour')
    gp = train_df[['ip','app', 'channel','hour']].groupby(by=['ip', 'app', 'channel'])[['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'})
    print("saving...")
    filename = 'ip_app_chl_mean_hour_%d_%d.csv'%(frm,to)
    gp.to_csv(filename,index=False)
    del gp
    gc.collect()
    return 0


sub=DO(frm,to,0)
print ('done')
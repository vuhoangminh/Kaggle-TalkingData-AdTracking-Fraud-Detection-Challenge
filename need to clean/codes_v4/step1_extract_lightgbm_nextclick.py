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
nrows=184903891-1
# nchunk=180000000
nchunk=10000
val_size=int(nchunk*0.1)

# frm=nrows-75000000
frm=0
if debug:
    frm=0
    nchunk=100000
    val_size=10000
to=1


if debug:
    print('*** debug parameter set: this is a test run for debugging purposes ***')

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

    print('doing nextClick')
    predictors=[]
    
    new_feature = 'nextClick'
    filename='nextClick_%d_%d.csv'%(frm,to)

    if os.path.exists(filename):
        print('loading from save file')
        QQ=pd.read_csv(filename).values
    else:
        D=2**26
        train_df['category'] = (train_df['ip'].astype(str) + "_" + train_df['app'].astype(str) + "_" + train_df['device'].astype(str) \
            + "_" + train_df['os'].astype(str)).apply(hash) % D
        click_buffer= np.full(D, 3000000000, dtype=np.uint32)

        train_df['epochtime']= train_df['click_time'].astype(np.int64) // 10 ** 9
        next_clicks= []
        for category, t in zip(reversed(train_df['category'].values), reversed(train_df['epochtime'].values)):
            next_clicks.append(click_buffer[category]-t)
            click_buffer[category]= t
        del(click_buffer)
        QQ= list(reversed(next_clicks))

        if not debug:
            print('saving')
            pd.DataFrame(QQ).to_csv(filename,index=False)

    return 0            

sub=DO(frm,to,0)
print ('done')
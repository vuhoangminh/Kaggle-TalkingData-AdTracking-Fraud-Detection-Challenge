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

    len_train = len(train_df)
    len_val = len(val_df)
    len_test = len(test_df)

    train_df=train_df.append(val_df)
    del val_df; gc.collect()
    train_df=train_df.append(test_df)
    del test_df; gc.collect()

    # if which_set == 'train':
    #     nrows = len_train
    #     train_df = train_df.iloc[:len_train]
    # if which_set == 'valid':        
    #     nrows = len_val
    # if which_set == 'test':
    #     nrows = len_test
    
    print('Extracting new features...')
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    print('Total memory in use after extracting hour, day: ', process.memory_info().rss/(2**30), ' GB\n')
    gc.collect()

    print('grouping by ip-day-hour combination...')
    filename = 'ip_day_hour_%d_%d.csv'%(frm,to)
    gp = pd.read_csv(filename)
    train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')
    print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')
    del gp; gc.collect()

    print('grouping by ip-app combination...')
    filename = 'ip_app_%d_%d.csv'%(frm,to)
    gp = pd.read_csv(filename)
    train_df = train_df.merge(gp, on=['ip','app'], how='left')
    print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')
    del gp; gc.collect()

    print('grouping by ip-app-os combination...')
    filename = 'ip_app_os_%d_%d.csv'%(frm,to)
    gp = pd.read_csv(filename)
    train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
    print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')
    del gp; gc.collect()

    # Adding features with var and mean hour (inspired from nuhsikander's script)
    print('grouping by ip_day_chl_var_hour')
    filename = 'ip_day_chl_var_hour_%d_%d.csv'%(frm,to)
    gp = pd.read_csv(filename)
    train_df = train_df.merge(gp, on=['ip','day','channel'], how='left')
    print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')
    del gp; gc.collect()

    print('grouping by ip_app_os_var_hour')
    filename = 'ip_app_os_var_hour_%d_%d.csv'%(frm,to)
    gp = pd.read_csv(filename)
    train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
    print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')
    del gp; gc.collect()

    print('grouping by ip_app_channel_var_day')
    filename = 'ip_app_channel_var_day_%d_%d.csv'%(frm,to)
    gp = pd.read_csv(filename)
    train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
    print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')
    del gp; gc.collect()

    print('grouping by ip_app_chl_mean_hour')
    filename = 'ip_app_chl_mean_hour_%d_%d.csv'%(frm,to)
    gp = pd.read_csv(filename)
    print("merging...")
    train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
    print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')
    del gp; gc.collect()
    
    
    
    naddfeat=9
    for i in range(0,naddfeat):
        if i==0: selcols=['ip', 'channel']; QQ=4;
        if i==1: selcols=['ip', 'device', 'os', 'app']; QQ=5;
        if i==2: selcols=['ip', 'day', 'hour']; QQ=4;
        if i==3: selcols=['ip', 'app']; QQ=4;
        if i==4: selcols=['ip', 'app', 'os']; QQ=4;
        if i==5: selcols=['ip', 'device']; QQ=4;
        if i==6: selcols=['app', 'channel']; QQ=4;
        if i==7: selcols=['ip', 'os']; QQ=5;
        if i==8: selcols=['ip', 'device', 'os', 'app']; QQ=4;
        print('selcols',selcols,'QQ',QQ)
        
        filename='X%d_%d_%d.csv'%(i,frm,to)
        
        if os.path.exists(filename):
            if QQ==5: 
                gp=pd.read_csv(filename,header=None)
                train_df['X'+str(i)]=gp
            else: 
                gp=pd.read_csv(filename)
                train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
        else:
            if QQ==0:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].count().reset_index().\
                    rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
                train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
            if QQ==1:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].mean().reset_index().\
                    rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
                train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
            if QQ==2:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].var().reset_index().\
                    rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
                train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
            if QQ==3:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].skew().reset_index().\
                    rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
                train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
            if QQ==4:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].nunique().reset_index().\
                    rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
                train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
            if QQ==5:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].cumcount()
                train_df['X'+str(i)]=gp.values
            
            if not debug:
                 gp.to_csv(filename,index=False)
            
        del gp
        gc.collect()    
    print('Total memory in use after QQ: ', process.memory_info().rss/(2**30), ' GB\n')         

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

    train_df[new_feature] = QQ
    predictors.append(new_feature)

    train_df[new_feature+'_shift'] = pd.DataFrame(QQ).shift(+1).values
    predictors.append(new_feature+'_shift')
    print('Total memory in use after nextClick: ', process.memory_info().rss/(2**30), ' GB\n')       
    del QQ; gc.collect()

    print("vars and data type: ")
    train_df.info()
    train_df['ip_tcount'] = train_df['ip_tcount'].astype('uint16')
    gc.collect()
    print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')       
    train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
    gc.collect()
    print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')       
    train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')
    gc.collect()
    print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')

    predictors.extend(['app','device','os', 'channel', 'hour', 'day', 
                  'ip_tcount', 'ip_tchan_count', 'ip_app_count',
                  'ip_app_os_count', 'ip_app_os_var',
                  'ip_app_channel_var_day','ip_app_channel_mean_hour'])
    for i in range(0,naddfeat):
        predictors.append('X'+str(i))
        
    print('predictors',predictors)

    test_df = train_df[len_train+len_val:]
    val_df = train_df[len_train:len_train+len_val]
    train_df = train_df[:len_train]

    print("train size: ", len(train_df))
    print("valid size: ", len(val_df))
    print("test size : ", len(test_df))

    print('saving...')
    save_name='test_%d_%d'%(frm,to)
    test_df.to_pickle(save_name)
    save_name='val_%d_%d'%(frm,to)
    val_df.to_pickle(save_name)
    save_name='train_%d_%d'%(frm,to)
    train_df.to_pickle(save_name)

    return 0

which_set = 'train'
sub=DO(frm,to,0)
print ('done')
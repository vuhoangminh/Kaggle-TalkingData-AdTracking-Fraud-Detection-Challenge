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
import random as rnd
import os
import featuretools as ft
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

process = psutil.Process(os.getpid())

debug=1
nrows=1000000
if debug:
    nchunk=100000
else:
    nchunk=180000000

val_size=int(nrows*0.1)
print('val size:',val_size)

frac = 0.1
frm=0
to=1


if debug:
    print('*** debug parameter set: this is a test run for debugging purposes ***')

def feature_engineering(train_df):
    print('=======================================================================')
    print('start feature engineering...')
    print('=======================================================================')
    
    es = ft.EntitySet(id="clicks")

    es = es.entity_from_dataframe(entity_id="clicks",
                                dataframe=train_df,
                                index="click_id",
                                variable_types={
                                    'ip': ft.variable_types.Categorical,
                                    'app': ft.variable_types.Categorical,
                                    'device': ft.variable_types.Categorical,
                                    'os': ft.variable_types.Categorical,
                                    'channel': ft.variable_types.Categorical,
                                    'is_attributed': ft.variable_types.Boolean
                                })

    print('normalize entity...')
    es = es.normalize_entity(base_entity_id="clicks", 
                            new_entity_id="info_ip",
                            index = "ip")

    es = es.normalize_entity(base_entity_id="clicks", 
                            new_entity_id="info_app",
                            index = "app")

    es = es.normalize_entity(base_entity_id="clicks", 
                            new_entity_id="info_device",
                            index = "device")

    es = es.normalize_entity(base_entity_id="clicks", 
                            new_entity_id="info_os",
                            index = "os")

    es = es.normalize_entity(base_entity_id="clicks", 
                            new_entity_id="info_channel",
                            index = "channel")

    print('----------------------------------------')
    print('Before automate')
    print(train_df.head())
    print(train_df.tail())

    # ## Regenerate new features related to IP
    print('Regenerate new features related to IP...')
    filename='IP_%d_%d.csv'%(frm,to)
    if os.path.exists(filename):
        print ('Done already')
    else:
        feature_matrix_ip, feature_defs_ip=ft.dfs(entityset=es, target_entity="info_ip")
        feature_matrix_ip['ip']=feature_matrix_ip.index
        feature_matrix_ip.rename(columns={"MODE(clicks.channel)": "mode_channel_ip",
                                                    "MODE(clicks.device)": "mode_device_ip", 
                                                    "MODE(clicks.os)": "mode_os_ip",
                                                    "MODE(clicks.app)": "mode_app_ip",
                                                    "NUM_UNIQUE(clicks.device)": "numunique_device_ip"}, inplace=True)
        print('doing mode_channel_ip...')                                                
        train_df = train_df.merge(feature_matrix_ip[['ip', 'mode_channel_ip']])
        print('doing mode_device_ip...')
        train_df = train_df.merge(feature_matrix_ip[['ip', 'mode_device_ip']])
        print('doing mode_os_ip...')
        train_df = train_df.merge(feature_matrix_ip[['ip', 'mode_os_ip']])
        print('doing mode_app_ip...')
        train_df = train_df.merge(feature_matrix_ip[['ip', 'mode_app_ip']])
        print('doing numunique_device_ip...')
        train_df = train_df.merge(feature_matrix_ip[['ip', 'numunique_device_ip']])
        
        feature_matrix_ip.to_csv(filename,index=False)

        del feature_matrix_ip, feature_defs_ip
        gc.collect()     
    print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')

    # ## Regenerate new features related to APP
    print('Regenerate new features related to APP...')
    # if debug: 
    #     print('feature matrix app:',feature_matrix_app.describe())
    # print(feature_defs_app)
    filename='APP_%d_%d.csv'%(frm,to)
    if os.path.exists(filename):
        print ('Done already')
    else:
        feature_matrix_app, feature_defs_app=ft.dfs(entityset=es, target_entity="info_app")
        feature_matrix_app['app']=feature_matrix_app.index
        feature_matrix_app.rename(columns={"MODE(clicks.channel)": "mode_channel_app", 
                                        "MODE(clicks.device)": "mode_device_app", 
                                        "MODE(clicks.os)": "mode_os_app",
                                        "MODE(clicks.ip)": "mode_ip_app"}, inplace=True)
        print('doing mode_channel_app...')                                        
        train_df = train_df.merge(feature_matrix_app[['app', 'mode_channel_app']])
        print('doing mode_device_app...')
        train_df = train_df.merge(feature_matrix_app[['app', 'mode_device_app']])
        print('doing mode_os_app...')
        train_df = train_df.merge(feature_matrix_app[['app', 'mode_os_app']])
        print('doing mode_ip_app...')
        train_df = train_df.merge(feature_matrix_app[['app', 'mode_ip_app']])
        feature_matrix_app.to_csv(filename,index=False)

        del feature_matrix_app, feature_defs_app
        gc.collect()      
    print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')

    # ## Regenerate new features related to DEVICE
    print('Regenerate new features related to DEVICE...')
    # if debug: 
    #     print('feature matrix devie:',feature_matrix_device.describe())
    # print(feature_defs_device)
    filename='DEVICE_%d_%d.csv'%(frm,to)
    if os.path.exists(filename):
        print ('Done already')
    else:
        feature_matrix_device, feature_defs_device=ft.dfs(entityset=es, target_entity="info_device")
        feature_matrix_device['device']=feature_matrix_device.index
        feature_matrix_device.rename(columns={"MODE(clicks.channel)": "mode_channel_device", 
                                        "MODE(clicks.app)": "mode_app_device", 
                                        "MODE(clicks.os)": "mode_os_device",
                                        "MODE(clicks.ip)": "mode_ip_device"}, inplace=True)
        print('doing mode_channel_device...')
        train_df = train_df.merge(feature_matrix_device[['device', 'mode_channel_device']])
        print('doing mode_app_device...')
        train_df = train_df.merge(feature_matrix_device[['device', 'mode_app_device']])
        print('doing mode_os_device...')
        train_df = train_df.merge(feature_matrix_device[['device', 'mode_os_device']])
        print('doing mode_ip_device...')
        train_df = train_df.merge(feature_matrix_device[['device', 'mode_ip_device']])
        feature_matrix_device.to_csv(filename,index=False)    
        del feature_matrix_device, feature_defs_device
        gc.collect()
    print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')    

    # ## Regenerate new features related to OS
    print('Regenerate new features related to OS...')
    # if debug: 
    #     print('feature matrix os:',feature_matrix_os.describe())
    # print(feature_defs_os)
    filename='OS_%d_%d.csv'%(frm,to)
    if os.path.exists(filename):
        print ('Done already')
    else:
        feature_matrix_os, feature_defs_os=ft.dfs(entityset=es, target_entity="info_os")
        feature_matrix_os['os']=feature_matrix_os.index
        feature_matrix_os.rename(columns={"MODE(clicks.channel)": "mode_channel_os", 
                                        "MODE(clicks.app)": "mode_app_os", 
                                        "MODE(clicks.device)": "mode_device_os",
                                        "MODE(clicks.ip)": "mode_ip_os",
                                        "NUM_UNIQUE(clicks.device)": "numunique_device_os"}, inplace=True)
        print('doing mode_channel_os...')                                    
        train_df = train_df.merge(feature_matrix_os[['os', 'mode_channel_os']])
        print('doing mode_app_os...')
        train_df = train_df.merge(feature_matrix_os[['os', 'mode_app_os']])
        print('doing mode_device_os...')
        train_df = train_df.merge(feature_matrix_os[['os', 'mode_device_os']])
        print('doing mode_ip_os...')
        train_df = train_df.merge(feature_matrix_os[['os', 'mode_ip_os']])
        print('doing numunique_device_os...')
        train_df = train_df.merge(feature_matrix_os[['os', 'numunique_device_os']])
        feature_matrix_os.to_csv(filename,index=False)
        del feature_matrix_os, feature_defs_os
        gc.collect()
    print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')    


    # ## Regenerate new features related to CHANNEL
    print('Regenerate new features related to CHANNEL...')
    # if debug: 
    #     print('feature matrix channel:',feature_matrix_channel.describe())
    # print(feature_defs_channel)
    filename='CHANNEL_%d_%d.csv'%(frm,to)
    if os.path.exists(filename):
        print ('Done already')
    else:
        feature_matrix_channel, feature_defs_channel=ft.dfs(entityset=es, target_entity="info_channel")
        feature_matrix_channel['channel']=feature_matrix_channel.index
        feature_matrix_channel.rename(columns={"MODE(clicks.os)": "mode_os_channel", 
                                        "MODE(clicks.app)": "mode_app_channel", 
                                        "MODE(clicks.device)": "mode_device_channel",
                                        "MODE(clicks.ip)": "mode_ip_channel",
                                        "NUM_UNIQUE(clicks.device)": "numunique_device_channel", 
                                        "NUM_UNIQUE(clicks.app)": "numunique_app_channel"}, inplace=True)
        print('doing mode_os_channel...')
        train_df = train_df.merge(feature_matrix_channel[['channel', 'mode_os_channel']])
        print('doing mode_app_channel...')
        train_df = train_df.merge(feature_matrix_channel[['channel', 'mode_app_channel']])
        print('doing mode_device_channel...')
        train_df = train_df.merge(feature_matrix_channel[['channel', 'mode_device_channel']])
        print('doing mode_ip_channel...')
        train_df = train_df.merge(feature_matrix_channel[['channel', 'mode_ip_channel']])
        print('doing numunique_device_channel...')
        train_df = train_df.merge(feature_matrix_channel[['channel', 'numunique_device_channel']])
        print('doing numunique_app_channel...')
        train_df = train_df.merge(feature_matrix_channel[['channel', 'numunique_app_channel']])
        feature_matrix_channel.to_csv(filename,index=False)
        del feature_matrix_channel, feature_defs_channel
        gc.collect()
    print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n') 
    # if debug:
    print('----------------------------------------')
    print('After automate')
    train_df = train_df.sort_values('click_id')
    print(train_df.head())
    print(train_df.tail())
    
    return train_df

def manual_feature_engineering(train_df, len_train, val_size):
    print('=======================================================================')
    print('start manual feature engineering...')
    print('=======================================================================')
    print('Extracting new features...')
    print(train_df.info())
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    print('Total memory in use after extracting hour, day: ', process.memory_info().rss/(2**30), ' GB\n')
    
    gc.collect()
    
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
            
            # if not debug:
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

        # if not debug:
        print('saving')
        pd.DataFrame(QQ).to_csv(filename,index=False)

    train_df[new_feature] = QQ
    predictors.append(new_feature)

    train_df[new_feature+'_shift'] = pd.DataFrame(QQ).shift(+1).values
    predictors.append(new_feature+'_shift')
    
    del QQ
    gc.collect()

    print('grouping by ip-day-hour combination...')
    gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_tcount'})
    train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')
    del gp
    gc.collect()

    print('grouping by ip-app combination...')
    gp = train_df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
    train_df = train_df.merge(gp, on=['ip','app'], how='left')
    del gp
    gc.collect()

    print('grouping by ip-app-os combination...')
    gp = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
    train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
    del gp
    gc.collect()

    # Adding features with var and mean hour (inspired from nuhsikander's script)
    print('grouping by ip_day_chl_var_hour')
    gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','channel'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_tchan_count'})
    train_df = train_df.merge(gp, on=['ip','day','channel'], how='left')
    del gp
    gc.collect()

    print('grouping by ip_app_os_var_hour')
    gp = train_df[['ip','app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_app_os_var'})
    train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
    del gp
    gc.collect()

    print('grouping by ip_app_channel_var_day')
    gp = train_df[['ip','app', 'channel', 'day']].groupby(by=['ip', 'app', 'channel'])[['day']].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_var_day'})
    train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
    del gp
    gc.collect()

    print('grouping by ip_app_chl_mean_hour')
    gp = train_df[['ip','app', 'channel','hour']].groupby(by=['ip', 'app', 'channel'])[['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'})
    print("merging...")
    train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
    del gp
    gc.collect()

    print("vars and data type: ")
    train_df.info()
    train_df['ip_tcount'] = train_df['ip_tcount'].astype('uint16')
    train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
    train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')


    for i in range(0,naddfeat):
        predictors.append('X'+str(i))
        
    print('predictors',predictors)
    print('fill nan with 0...')
    train_df = train_df.fillna(0)

    if debug:
        sns.heatmap(train_df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
        fig=plt.gcf()
        fig.set_size_inches(50,50)
        plt.show()

    print('----------------------------------------')
    print('After maunal')
    train_df = train_df.sort_values('click_id')
    print(train_df.head())
    print(train_df.tail())


    train_df = train_df.sort_values('click_id')
    test_df = train_df[:(len(train_df)-len_train)]
    val_df = train_df[(len(train_df)-len_train):(len(train_df)-len_train+val_size)]
    train_df = train_df[(len(train_df)-len_train+val_size):]
    # test_df = train_df[len_train:]
    # val_df = train_df[(len_train-val_size):len_train]
    # train_df = train_df[:(len_train-val_size)]

    if debug:
        print('index after split train and test...')
        print('len train:', len(train_df))            
        print('len val:', len(val_df))            
        print('len test:', len(test_df))  
        # print(train_df.info())
        # print(train_df.head())
        # print(train_df.tail()) 

    print('reset index...')
    train_df.reset_index(inplace=True, drop=True)
    val_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)
    
    print('----------------------------------------')
    print('After extracting test')
    print(test_df.head())
    print(test_df.tail())



    print('saving train, val, test...')
    save_name='test_%d_%d'%(frm,to)
    test_df.to_pickle(save_name)
    save_name='val_%d_%d'%(frm,to)
    val_df.to_pickle(save_name)
    save_name='train_%d_%d'%(frm,to)
    train_df.to_pickle(save_name)

    print('len train:', len(train_df))            
    print('len val:', len(val_df))            
    print('len test:', len(test_df))  

    del test_df, train_df, val_df
    gc.collect()


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

    print('loading train data...',frm,to)
    if debug:
        train_df = pd.read_csv("../input/train.csv", parse_dates=['click_time'], 
            skiprows=range(1,10), nrows=nrows, dtype=dtypes, 
            usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
    else:        
        train_df = pd.read_csv("../input/valid_day_9.csv", parse_dates=['click_time'], 
            dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])   
    print('len train:', len(train_df))            
    print('Total memory in use after reading train: ', process.memory_info().rss/(2**30), ' GB\n')         

    print('loading test data...')
    if debug:
        test_df = pd.read_csv("../input/test.csv", nrows=nrows, 
            parse_dates=['click_time'], dtype=dtypes, 
            usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    else:
        test_df = pd.read_csv("../input/test.csv", parse_dates=['click_time'], 
            dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    print('len test:', len(test_df))            
    print('Total memory in use after reading test: ', process.memory_info().rss/(2**30), ' GB\n')         

    len_train = len(train_df)
    train_df['click_id']=range(len(test_df), len(train_df)+len(test_df))
    train_df=train_df.append(test_df)
    train_df.reset_index(inplace=True, drop=True)
    # if debug:
    print('index after merge train and test...')
    train_df = train_df.sort_values('click_id')
    print(train_df.head())
    print(train_df.tail())

    del test_df
    gc.collect()        

    if debug:
        print('after drop...')
        print(train_df.head())
        print(train_df.tail())

    # print(train_df.head())
    print('fill nan with 0...')
    train_df = train_df.fillna(0)
    # train_df = feature_engineering(train_df)
    train_df['is_attributed'] = train_df['is_attributed'].astype('uint8')
    print('Total memory in use:', process.memory_info().rss/(2**30), ' GB\n')   
    manual_feature_engineering(train_df,len_train,val_size)
    print('Total memory in use:', process.memory_info().rss/(2**30), ' GB\n')        

    return 0


sub=DO(frm,to,0)


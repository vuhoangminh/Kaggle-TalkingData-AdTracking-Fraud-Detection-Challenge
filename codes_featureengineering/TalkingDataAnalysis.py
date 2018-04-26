# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import os
import featuretools as ft
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import gc
import psutil
import datetime

now = datetime.datetime.now()
print(now.year, now.month, now.day)

process = psutil.Process(os.getpid())

debug=0

train_df = pd.read_csv('../input/train.csv', nrows=1000)
print('Total memory in use after reading train: ', process.memory_info().rss/(2**30), ' GB\n')

train_df['click_id']=range(1, len(train_df) +1)
print(train_df.info())

# ## Create entities

def feature_engineering(train_df):
    at_df = pd.DataFrame()
    at_df['attributed_time'] = train_df['attributed_time']
    train_df = train_df.drop(["attributed_time"], axis=1)
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

    if debug:                             
        print('es:',es)

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

    if debug:                         
        print('info ip:',es["info_ip"])
        print('info app:',es["info_app"])
        print('info device:',es["info_device"])
        print('info os:',es["info_os"])
        print('info channel:',es["info_channel"])


    # ## Regenerate new features related to IP
    print('Regenerate new features related to IP...')
    feature_matrix_ip, feature_defs_ip=ft.dfs(entityset=es, target_entity="info_ip")
    if debug:   
        print('feature matrix ip:',feature_matrix_ip.describe())
    # print('feature defs ip',feature_defs_ip)

    feature_matrix_ip['ip']=feature_matrix_ip.index
    feature_matrix_ip.rename(columns={"MODE(clicks.channel)": "mode_channel_ip",
                                                "MODE(clicks.device)": "mode_device_ip", 
                                                "MODE(clicks.os)": "mode_os_ip",
                                                "MODE(clicks.app)": "mode_app_ip",
                                                "NUM_UNIQUE(clicks.device)": "numunique_device_ip"}, inplace=True)
    train_df = train_df.merge(feature_matrix_ip[['ip', 'mode_channel_ip']])
    train_df = train_df.merge(feature_matrix_ip[['ip', 'mode_device_ip']])
    train_df = train_df.merge(feature_matrix_ip[['ip', 'mode_os_ip']])
    train_df = train_df.merge(feature_matrix_ip[['ip', 'mode_app_ip']])
    train_df = train_df.merge(feature_matrix_ip[['ip', 'numunique_device_ip']])

    del feature_matrix_ip, feature_defs_ip
    gc.collect()
    print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')

    # ## Regenerate new features related to APP
    print('Regenerate new features related to APP...')
    feature_matrix_app, feature_defs_app=ft.dfs(entityset=es, target_entity="info_app")
    if debug: 
        print('feature matrix app:',feature_matrix_app.describe())
    # print(feature_defs_app)

    feature_matrix_app['app']=feature_matrix_app.index
    feature_matrix_app.rename(columns={"MODE(clicks.channel)": "mode_channel_app", 
                                    "MODE(clicks.device)": "mode_device_app", 
                                    "MODE(clicks.os)": "mode_os_app",
                                    "MODE(clicks.ip)": "mode_ip_app"}, inplace=True)
    train_df = train_df.merge(feature_matrix_app[['app', 'mode_channel_app']])
    train_df = train_df.merge(feature_matrix_app[['app', 'mode_device_app']])
    train_df = train_df.merge(feature_matrix_app[['app', 'mode_os_app']])
    train_df = train_df.merge(feature_matrix_app[['app', 'mode_ip_app']])

    del feature_matrix_app, feature_defs_app
    gc.collect()
    print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')

    # ## Regenerate new features related to DEVICE
    print('Regenerate new features related to DEVICE...')
    feature_matrix_device, feature_defs_device=ft.dfs(entityset=es, target_entity="info_device")
    if debug: 
        print('feature matrix devie:',feature_matrix_device.describe())
    # print(feature_defs_device)

    feature_matrix_device['device']=feature_matrix_device.index
    feature_matrix_device.rename(columns={"MODE(clicks.channel)": "mode_channel_device", 
                                    "MODE(clicks.app)": "mode_app_device", 
                                    "MODE(clicks.os)": "mode_os_device",
                                    "MODE(clicks.ip)": "mode_ip_device"}, inplace=True)
    train_df = train_df.merge(feature_matrix_device[['device', 'mode_channel_device']])
    train_df = train_df.merge(feature_matrix_device[['device', 'mode_app_device']])
    train_df = train_df.merge(feature_matrix_device[['device', 'mode_os_device']])
    train_df = train_df.merge(feature_matrix_device[['device', 'mode_ip_device']])

    del feature_matrix_device, feature_defs_device
    gc.collect()
    print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')    


    # ## Regenerate new features related to OS
    print('Regenerate new features related to OS...')
    feature_matrix_os, feature_defs_os=ft.dfs(entityset=es, target_entity="info_os")
    if debug: 
        print('feature matrix os:',feature_matrix_os.describe())
    # print(feature_defs_os)

    feature_matrix_os['os']=feature_matrix_os.index
    feature_matrix_os.rename(columns={"MODE(clicks.channel)": "mode_channel_os", 
                                    "MODE(clicks.app)": "mode_app_os", 
                                    "MODE(clicks.device)": "mode_device_os",
                                    "MODE(clicks.ip)": "mode_ip_os",
                                    "NUM_UNIQUE(clicks.device)": "numunique_device_os"}, inplace=True)
    train_df = train_df.merge(feature_matrix_os[['os', 'mode_channel_os']])
    train_df = train_df.merge(feature_matrix_os[['os', 'mode_app_os']])
    train_df = train_df.merge(feature_matrix_os[['os', 'mode_device_os']])
    train_df = train_df.merge(feature_matrix_os[['os', 'mode_ip_os']])
    train_df = train_df.merge(feature_matrix_os[['os', 'numunique_device_os']])

    del feature_matrix_os, feature_defs_os
    gc.collect()
    print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')    


    # ## Regenerate new features related to CHANNEL
    print('Regenerate new features related to CHANNEL...')
    feature_matrix_channel, feature_defs_channel=ft.dfs(entityset=es, target_entity="info_channel")
    if debug: 
        print('feature matrix channel:',feature_matrix_channel.describe())
    # print(feature_defs_channel)

    feature_matrix_channel['channel']=feature_matrix_channel.index
    feature_matrix_channel.rename(columns={"MODE(clicks.os)": "mode_os_channel", 
                                    "MODE(clicks.app)": "mode_app_channel", 
                                    "MODE(clicks.device)": "mode_device_channel",
                                    "MODE(clicks.ip)": "mode_ip_channel",
                                    "NUM_UNIQUE(clicks.device)": "numunique_device_channel", 
                                    "NUM_UNIQUE(clicks.app)": "numunique_app_channel"}, inplace=True)
    train_df = train_df.merge(feature_matrix_channel[['channel', 'mode_os_channel']])
    train_df = train_df.merge(feature_matrix_channel[['channel', 'mode_app_channel']])
    train_df = train_df.merge(feature_matrix_channel[['channel', 'mode_device_channel']])
    train_df = train_df.merge(feature_matrix_channel[['channel', 'mode_ip_channel']])
    train_df = train_df.merge(feature_matrix_channel[['channel', 'numunique_device_channel']])
    train_df = train_df.merge(feature_matrix_channel[['channel', 'numunique_app_channel']])

    del feature_matrix_channel, feature_defs_channel
    gc.collect()
    print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB\n')    

    # train_df.concat([train_df, at_df], axis=1, join_axes=[train_df.index])
    train_df['attributed_time'] = at_df['attributed_time']
    return train_df

train_df = feature_engineering(train_df)    
print (train_df.head())
print(train_df.info())
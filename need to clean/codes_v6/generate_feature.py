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
import configs
import math

process = psutil.Process(os.getpid())

# # init vars
# CATEGORY_LIST = configs.cat_list
# DATATYPE_LIST = configs.datatype_list
# NCHUNKS = configs.nchunks
# CAT_COMBINATION_FILENAME = configs.CAT_COMBINATION_FILENAME
# NEXTCLICK_FILENAME = configs.NEXTCLICK_FILENAME
# TIME_FILENAME = configs.TIME_FILENAME

CATEGORY_LIST = ['ip', 'app', 'device', 'os', 'channel', 
    'mobile', 'mobile_app', 'mobile_channel', 'app_channel'
    ]
DATATYPE_LIST = {
    'ip'                : 'uint32',
    'app'               : 'uint16',
    'device'            : 'uint16',
    'os'                : 'uint16',
    'channel'           : 'uint16',
    'is_attributed'     : 'uint8',
    'click_id'          : 'uint32',
    'mobile'            : 'uint16',
    'mobile_app'        : 'uint16',
    'mobile_channel'    : 'uint16',
    'app_channel'       : 'uint16',
    'category'          : 'category',
    'epochtime'         : 'int64',
    'nextClick'         : 'int64',
    'nextClick_shift'   : 'float64',
    'min'               : 'uint8',
    'day'               : 'uint8',
    'hour'              : 'uint8',
    'ip_mobile_day_count_hour'              : 'uint32',
    'ip_mobileapp_day_count_hour'           : 'uint32',
    'ip_mobilechannel_day_count_hour'       : 'uint32',
    'ip_appchannel_day_count_hour'          : 'uint32',
    'ip_mobile_app_channel_day_count_hour'  : 'uint32',
    'ip_mobile_day_var_hour'                : 'float32',
    'ip_mobileapp_day_var_hour'             : 'float32',
    'ip_mobilechannel_day_var_hour'         : 'float32',
    'ip_appchannel_day_var_hour'            : 'float32',
    'ip_mobile_app_channel_day_var_hour'    : 'float32',
    'ip_mobile_day_std_hour'                : 'float32',
    'ip_mobileapp_day_std_hour'             : 'float32',
    'ip_mobilechannel_day_std_hour'         : 'float32',
    'ip_appchannel_day_std_hour'            : 'float32',
    'ip_mobile_app_channel_day_std_hour'    : 'float32',
    'ip_mobile_day_cumcount_hour'               : 'uint32',
    'ip_mobileapp_day_cumcount_hour'            : 'uint32',
    'ip_mobilechannel_day_cumcount_hour'        : 'uint32',
    'ip_appchannel_day_cumcount_hour'           : 'uint32',
    'ip_mobile_app_channel_day_cumcount_hour'   : 'uint32',
    'ip_mobile_day_nunique_hour'                : 'uint32',
    'ip_mobileapp_day_nunique_hour'             : 'uint32',
    'ip_mobilechannel_day_nunique_hour'         : 'uint32',
    'ip_appchannel_day_nunique_hour'            : 'uint32',
    'ip_mobile_app_channel_day_nunique_hour'    : 'uint32'
    }
PATH = '../processed_day9/'        
CAT_COMBINATION_FILENAME = PATH + 'day9_cat_combination.csv'
CAT_COMBINATION_NUMERIC_CATEGORY_FILENAME = PATH + 'day9_cat_combination_numeric_category.csv'
NEXTCLICK_FILENAME = PATH + 'day9_nextClick.csv'
TIME_FILENAME = PATH + 'day9_day_hour_min.csv'
IP_HOUR_RELATED_FILENAME = PATH + 'day9_ip_hour_related.csv'
TRAINSET_FILENAME = '../input/valid_day_9.csv'
NCHUNKS = 100000

debug=1
nrows=10000000
# nrows=10

if not debug:
    print('=======================================================================')
    print('process on server...')
    print('=======================================================================')
else:
    print('=======================================================================')
    print('for testing only...')
    print('=======================================================================')

def convert_to_right_type():
    usecols_train=['ip','app','device','os', 'channel', 'click_time', 'is_attributed']
    usecols_test=['ip','app','device','os', 'channel', 'click_time', 'click_id']
    print('loading train data...')
    if debug:
        train_df = pd.read_csv(TRAINSET_FILENAME, nrows=nrows, parse_dates=['click_time'], 
                dtype=DATATYPE_LIST, usecols=usecols_train)
    else:        
        train_df = pd.read_csv(TRAINSET_FILENAME, 
                dtype=DATATYPE_LIST, usecols=usecols_train)  
    print(train_df.info())      
    train_df.to_csv('valid_day_9.csv', index=False)  
    del train_df; gc.collect
    print('loading test data...')
    if debug:
        test_df = pd.read_csv("../input/test.csv", nrows=nrows, parse_dates=['click_time'], 
                dtype=DATATYPE_LIST, usecols=usecols_test)
    else:        
        test_df = pd.read_csv("../input/test.csv", 
                dtype=DATATYPE_LIST, usecols=usecols_test)  
    print(test_df.info())    
    test_df.to_csv('test.csv', index=False)                     

def convert_categorical_to_number(train_df):
    train_df['ip'] = pd.Categorical(train_df.ip).codes
    train_df['app'] = pd.Categorical(train_df.app).codes
    train_df['device'] = pd.Categorical(train_df.device).codes
    train_df['os'] = pd.Categorical(train_df.os).codes
    train_df['channel'] = pd.Categorical(train_df.channel).codes
    train_df['mobile'] = pd.Categorical(train_df.mobile).codes
    train_df['mobile_app'] = pd.Categorical(train_df.mobile_app).codes
    train_df['mobile_channel'] = pd.Categorical(train_df.mobile_channel).codes
    train_df['app_channel'] = pd.Categorical(train_df.app_channel).codes
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
    print('After manual')
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
    train_df['app'] = pd.Categorical.from_array(train_df.app).codes

def convert_to_save_memory(train_df):
    for feature, type in DATATYPE_LIST.items(): 
        if feature in list(train_df):
            train_df[feature]=train_df[feature].astype(type)        
    return train_df

def convert_cat_combination_to_number():
    print('reading CAT_COMBINATION_FILENAME')
    train_df = pd.read_csv(CAT_COMBINATION_FILENAME)
    print_memory()
    print(train_df.describe())
    print(train_df.info())
    print('convert mobile...')
    train_df['mobile'] = pd.Categorical(train_df.mobile).codes
    train_df['mobile'] = train_df['mobile'].astype('uint16')
    print('convert mobile_app...')
    train_df['mobile_app'] = pd.Categorical(train_df.mobile_app).codes
    train_df['mobile_app'] = train_df['mobile_app'].astype('uint16')
    print('convert mobile_channel...')
    train_df['mobile_channel'] = pd.Categorical(train_df.mobile_channel).codes
    train_df['mobile_channel'] = train_df['mobile_channel'].astype('uint16')
    print('convert app_channel...')
    train_df['app_channel'] = pd.Categorical(train_df.app_channel).codes
    train_df['app_channel'] = train_df['app_channel'].astype('uint16')
    print(train_df.info())
    print(train_df.describe()) 
    print_memory()
    print('saving...')
    train_df.to_csv(CAT_COMBINATION_NUMERIC_CATEGORY_FILENAME, index=False)

def print_memory(print_string=''):
    print('Total memory in use ' + print_string + ': ', process.memory_info().rss/(2**30), ' GB')

def read_train(usecols_train, filename):
    if debug:
        if 'click_time' in usecols_train:
            train_df = pd.read_csv(filename, 
                skiprows=range(1,10), nrows=nrows, dtype=DATATYPE_LIST, parse_dates=['click_time'],
                usecols=usecols_train)
        else:
            train_df = pd.read_csv(filename, 
                skiprows=range(1,10), nrows=nrows, dtype=DATATYPE_LIST, 
                usecols=usecols_train)
    else:
        if 'click_time' in usecols_train:
            train_df = pd.read_csv(filename, parse_dates=['click_time'],
                dtype=DATATYPE_LIST, usecols=usecols_train)           
        else:
            train_df = pd.read_csv(filename, 
                dtype=DATATYPE_LIST, usecols=usecols_train)    
    return train_df                     

def read_train_test(style='is_merged', \
    usecols_train=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'], \
    usecols_test=['ip','app','device','os', 'channel', 'click_time', 'click_id']):
    print('loading train data...')
    train_df = read_train(usecols_train, TRAINSET_FILENAME)
    print('len train:', len(train_df))            
    print_memory ('after reading train')

    print('loading test data...')
    test_df = read_train(usecols_test, '../input/test.csv')
    print('len test:', len(test_df))            
    print_memory ('after reading test')
    if style == 'is_merged':
        train_df = train_df.append(test_df)
        train_df = train_df.fillna(0)
        del test_df; gc.collect()
        return train_df 
    else:
        return train_df, test_df   

def describe_one_column(train_df, feature):  
    print('---------------------------------')            
    print(feature, ':')    
    print(train_df[feature].describe())  
    if feature in CATEGORY_LIST: 
        train_df[feature] = train_df[feature].astype('category')
        print('---------------------------------')            
        print(feature, ':')    
        print(train_df[feature].describe()) 

def describe_one_dataset(train_df):     
    for feature in list(train_df):
        describe_one_column(train_df, feature)

def describe_both_dataset():
    train_df, test_df = read_train_test('is_not_merged')
    if debug: print (list(train_df)); print(train_df.info())
    print('--------------------------------------------------------')        
    print('describe train...')
    describe_one_dataset(train_df)
    print('--------------------------------------------------------')        
    print('describe test...')
    describe_one_dataset(test_df)  

def create_cat_combination(df):
    range_split = range(0,len(df),NCHUNKS)
    num_split = len(range(0,len(df),NCHUNKS))
    print('process', num_split, 'splits...')
    # gp_full = pd.DataFrame()
    for i in range(num_split):
        print('------------------------------------------')
        print('process split', i)
        if i+1<num_split:
            train_df = df[range_split[i]:range_split[i+1]]
            if debug==2: print(train_df)
        else:
            train_df = df[range_split[i]:]  
            if debug==2: print(train_df)
        gp = pd.DataFrame()  
        print('convert to string...')                      
        gp_device = train_df['device'].astype(str)
        gp_os = train_df['os'].astype(str)
        gp_app = train_df['app'].astype(str)
        gp_channel = train_df['channel'].astype(str)
        gp = pd.DataFrame()
        print('doing mobile...')
        gp['mobile'] = gp_device + '_' + gp_os
        print('doing mobile_app...')
        gp['mobile_app'] = gp_device + '_' + gp_os + '_' + gp_app
        print('doing mobile_channel...')
        gp['mobile_channel'] = gp_device + '_' + gp_os + '_' + gp_channel
        print('doing app_channel...')
        gp['app_channel'] = gp_app + '_' + gp_channel
        print('save cat combination...')
        if i==0:
            gp.to_csv(CAT_COMBINATION_FILENAME,index=False)
        else:
            print('concat cat combination...')
            with open(CAT_COMBINATION_FILENAME, 'a') as f:
                gp.to_csv(f, header=False, index=False)
        
        del gp_app, gp_channel, gp_device, gp_os, gp; gc.collect()
        print_memory()


def create_next_click(train_df):
    print('doing nextClick...')
    filename=NEXTCLICK_FILENAME
    gp = pd.DataFrame()
    D=2**26
    gp['category'] = (train_df['ip'].astype(str) + "_" + train_df['app'].astype(str) + "_" + train_df['device'].astype(str) \
        + "_" + train_df['os'].astype(str)).apply(hash) % D
    click_buffer= np.full(D, 3000000000, dtype=np.uint32)

    print('doing epochtime...')
    gp['epochtime']= train_df['click_time'].astype(np.int64) // 10 ** 9
    next_clicks= []
    print('doing nextClick...')
    i = 0
    for category, t in zip(reversed(gp['category'].values), reversed(gp['epochtime'].values)):
        if i%10000 == 0: print ('process', i)
        next_clicks.append(click_buffer[category]-t)
        click_buffer[category]= t
        i = i+1
    del(click_buffer); gc.collect
    gp['nextClick'] = list(reversed(next_clicks))
    # if not debug:
    print('saving...')
    gp['nextClick_shift'] = gp['nextClick'].shift(+1).values
    gp = gp.fillna(0)
    pd.DataFrame(gp).to_csv(filename,index=False)
    del gp; gc.collect()

def create_day_hour_min(train_df):
    print('Extracting new features...')
    gp = pd.DataFrame()
    gp['min'] = pd.to_datetime(train_df.click_time).dt.minute.astype('uint8')
    gp['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    gp['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    gp.to_csv(TIME_FILENAME,index=False)

def generate_groupby_by_type_and_columns(train_df, selcols, apply_type):      
    feature_name = ''
    for i in range(len(selcols)-1):
        feature_name = feature_name + selcols[i] + '_'
    feature_name = feature_name + apply_type + '_' + selcols[len(selcols)-1]
    print('doing feature:', feature_name)
    filename = 'day9_' + feature_name + '.csv'
    if os.path.exists(filename):
        print ('done already...')
    else:
        if apply_type == 'count':
            col_temp = train_df[selcols]. \
                groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].count(). \
                reset_index().rename(index=str, columns={selcols[len(selcols)-1]: feature_name})
        if apply_type == 'var':
            col_temp = train_df[selcols]. \
                groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].var(). \
                reset_index().rename(index=str, columns={selcols[len(selcols)-1]: feature_name})
        if apply_type == 'std':
            col_temp = train_df[selcols]. \
                groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].std(). \
                reset_index().rename(index=str, columns={selcols[len(selcols)-1]: feature_name})   
        if apply_type == 'cumcount':
            col_temp = train_df[selcols]. \
                groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].cumcount()
        if apply_type == 'nunique':
            col_temp = train_df[selcols]. \
                groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].nunique(). \
                reset_index().rename(index=str, columns={selcols[len(selcols)-1]: feature_name})                       

        col_extracted = pd.DataFrame()
        if apply_type != 'cumcount':
            train_df = train_df.merge(col_temp, on=selcols[0:len(selcols)-1], how='left')
            del col_temp; gc.collect()
            col_extracted[feature_name] = train_df[feature_name]
        else:
            col_extracted[feature_name] = col_temp.values
            del col_temp; gc.collect()

        col_extracted.to_csv(filename, index=False)
        del col_extracted; gc.collect()
        if debug==2: print(train_df.head()); print(col_temp.head())
    return feature_name        
    # return train_df, feature_name               

def create_ip_hour_related_features(train_df):
    gp = pd.DataFrame()
    new_feature_list=[]
    # get count
    apply_type = 'count'
    selcols = ['ip','mobile','day','hour']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)
    
    selcols = ['ip','mobile_app','day','hour']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)
    
    selcols = ['ip','mobile_channel','day','hour']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)

    selcols = ['ip','app_channel','day','hour']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)

    selcols = ['ip', 'mobile','app_channel','day','hour']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)    

    # get var
    apply_type = 'var'
    selcols = ['ip','mobile','day','hour']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)
    
    selcols = ['ip','mobile_app','day','hour']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)
    
    selcols = ['ip','mobile_channel','day','hour']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)

    selcols = ['ip','app_channel','day','hour']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)

    selcols = ['ip', 'mobile','app_channel','day','hour']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)     

    # get std
    apply_type = 'std'
    selcols = ['ip','mobile','day','hour']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)
    
    selcols = ['ip','mobile_app','day','hour']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)
    
    selcols = ['ip','mobile_channel','day','hour']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)

    selcols = ['ip','app_channel','day','hour']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)

    selcols = ['ip', 'mobile','app_channel','day','hour']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name) 
      
    # get cumcount
    apply_type = 'cumcount'
    selcols = ['ip','mobile','day','hour']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)
    
    selcols = ['ip','mobile_app','day','hour']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)
    
    selcols = ['ip','mobile_channel','day','hour']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)

    selcols = ['ip','app_channel','day','hour']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)

    selcols = ['ip', 'mobile','app_channel','day','hour']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)   

    # get nunique
    apply_type = 'nunique'
    selcols = ['ip','mobile','day','hour']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)
    
    selcols = ['ip','mobile_app','day','hour']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)
    
    selcols = ['ip','mobile_channel','day','hour']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)

    selcols = ['ip','app_channel','day','hour']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)

    selcols = ['ip', 'mobile','app_channel','day','hour']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)  

    return train_df

def create_ip_hour_related_features_big(train_df):
    gp = pd.DataFrame()
    # get count
    print('grouping by ip-mobile-day-hour combination...')
    col_temp = train_df[['ip','mobile','day','hour']]. \
        groupby(by=['ip','mobile','day'])[['hour']].count(). \
        reset_index().rename(index=str, columns={'hour': 'ip_mobile_day_count_hour'})
    gp['ip_mobile_day_count_hour'] = col_temp['ip_mobile_day_count_hour']        
    if debug == 1: 
        train_df = train_df.merge(col_temp, on=['ip','mobile','day'], how='left')
        print (train_df.info()); print(train_df.head())
    del col_temp; gc.collect()    
    print_memory('after grouping by ip-mobile-day-hour combination')    

    print('grouping by ip-mobileapp-day-hour combination...')
    col_temp = train_df[['ip','mobile_app','day','hour']]. \
        groupby(by=['ip','mobile_app','day'])[['hour']].count(). \
        reset_index().rename(index=str, columns={'hour': 'ip_mobileapp_day_count_hour'})
    gp['ip_mobileapp_day_count_hour'] = col_temp['ip_mobileapp_day_count_hour']        
    if debug == 1: 
        train_df = train_df.merge(col_temp, on=['ip','mobile_app','day'], how='left')
        print (train_df.info()); print(train_df.head())
    del col_temp; gc.collect()    
    print_memory('after grouping by ip-mobileapp-day-hour combination') 

    print('grouping by ip-mobilechannel-day-hour combination...')
    col_temp = train_df[['ip','mobile_channel','day','hour']]. \
        groupby(by=['ip','mobile_channel','day'])[['hour']].count(). \
        reset_index().rename(index=str, columns={'hour': 'ip_mobilechannel_day_count_hour'})
    gp['ip_mobilechannel_day_count_hour'] = col_temp['ip_mobilechannel_day_count_hour']        
    if debug == 1: 
        train_df = train_df.merge(col_temp, on=['ip','mobile_channel','day'], how='left')
        print (train_df.info()); print(train_df.head())
    del col_temp; gc.collect()    
    print_memory('after grouping by ip-mobilechannel-day-hour combination') 

    print('grouping by ip-appchannel-day-hour combination...')
    col_temp = train_df[['ip','app_channel','day','hour']]. \
        groupby(by=['ip','app_channel','day'])[['hour']].count(). \
        reset_index().rename(index=str, columns={'hour': 'ip_appchannel_day_count_hour'})
    gp['ip_appchannel_day_count_hour'] = col_temp['ip_appchannel_day_count_hour']        
    if debug == 1: 
        train_df = train_df.merge(col_temp, on=['ip','app_channel','day'], how='left')
        print (train_df.info()); print(train_df.head())
    del col_temp; gc.collect()    
    print_memory('after grouping by ip-appchannel-day-hour combination') 

    print('grouping by ip-mobile-appchannel-day-hour combination...')
    col_temp = train_df[['ip', 'mobile','app_channel','day','hour']]. \
        groupby(by=['ip','mobile','app_channel','day'])[['hour']].count(). \
        reset_index().rename(index=str, columns={'hour': 'ip_mobile_app_channel_day_count_hour'})
    gp['ip_mobile_app_channel_day_count_hour'] = col_temp['ip_mobile_app_channel_day_count_hour']        
    if debug == 1: 
        train_df = train_df.merge(col_temp, on=['ip','mobile','app_channel','day'], how='left')
        print (train_df.info()); print(train_df.head())
    del col_temp; gc.collect()    
    print_memory('after grouping by ip-mobile-appchannel-day-hour combination') 

    # get var
    gp = pd.DataFrame()
    print('grouping by ip-mobile-day-hour combination...')
    col_temp = train_df[['ip','mobile','day','hour']]. \
        groupby(by=['ip','mobile','day'])[['hour']].var(). \
        reset_index().rename(index=str, columns={'hour': 'ip_mobile_day_var_hour'})
    gp['ip_mobile_day_var_hour'] = col_temp['ip_mobile_day_var_hour']        
    if debug == 1: 
        train_df = train_df.merge(col_temp, on=['ip','mobile','day'], how='left')
        print (train_df.info()); print(train_df.head())
    del col_temp; gc.collect()    
    print_memory('after grouping by ip-mobile-day-hour combination')    

    print('grouping by ip-mobileapp-day-hour combination...')
    col_temp = train_df[['ip','mobile_app','day','hour']]. \
        groupby(by=['ip','mobile_app','day'])[['hour']].var(). \
        reset_index().rename(index=str, columns={'hour': 'ip_mobileapp_day_var_hour'})
    gp['ip_mobileapp_day_var_hour'] = col_temp['ip_mobileapp_day_var_hour']        
    if debug == 1: 
        train_df = train_df.merge(col_temp, on=['ip','mobile_app','day'], how='left')
        print (train_df.info()); print(train_df.head())
    del col_temp; gc.collect()    
    print_memory('after grouping by ip-mobileapp-day-hour combination') 

    print('grouping by ip-mobilechannel-day-hour combination...')
    col_temp = train_df[['ip','mobile_channel','day','hour']]. \
        groupby(by=['ip','mobile_channel','day'])[['hour']].var(). \
        reset_index().rename(index=str, columns={'hour': 'ip_mobilechannel_day_var_hour'})
    gp['ip_mobilechannel_day_var_hour'] = col_temp['ip_mobilechannel_day_var_hour']        
    if debug == 1: 
        train_df = train_df.merge(col_temp, on=['ip','mobile_channel','day'], how='left')
        print (train_df.info()); print(train_df.head())
    del col_temp; gc.collect()    
    print_memory('after grouping by ip-mobilechannel-day-hour combination') 

    print('grouping by ip-appchannel-day-hour combination...')
    col_temp = train_df[['ip','app_channel','day','hour']]. \
        groupby(by=['ip','app_channel','day'])[['hour']].var(). \
        reset_index().rename(index=str, columns={'hour': 'ip_appchannel_day_var_hour'})
    gp['ip_appchannel_day_var_hour'] = col_temp['ip_appchannel_day_var_hour']        
    if debug == 1: 
        train_df = train_df.merge(col_temp, on=['ip','app_channel','day'], how='left')
        print (train_df.info()); print(train_df.head())
    del col_temp; gc.collect()    
    print_memory('after grouping by ip-appchannel-day-hour combination') 

    print('grouping by ip-mobile-appchannel-day-hour combination...')
    col_temp = train_df[['ip', 'mobile','app_channel','day','hour']]. \
        groupby(by=['ip','mobile','app_channel','day'])[['hour']].var(). \
        reset_index().rename(index=str, columns={'hour': 'ip_mobile_app_channel_day_var_hour'})
    gp['ip_mobile_app_channel_day_var_hour'] = col_temp['ip_mobile_app_channel_day_var_hour']        
    if debug == 1: 
        train_df = train_df.merge(col_temp, on=['ip','mobile','app_channel','day'], how='left')
        print (train_df.info()); print(train_df.head())
    del col_temp; gc.collect()    
    print_memory('after grouping by ip-mobile-appchannel-day-hour combination') 


    # get std
    gp = pd.DataFrame()
    print('grouping by ip-mobile-day-hour combination...')
    col_temp = train_df[['ip','mobile','day','hour']]. \
        groupby(by=['ip','mobile','day'])[['hour']].var(). \
        reset_index().rename(index=str, columns={'hour': 'ip_mobile_day_std_hour'})
    gp['ip_mobile_day_std_hour'] = col_temp['ip_mobile_day_std_hour']        
    if debug == 1: 
        train_df = train_df.merge(col_temp, on=['ip','mobile','day'], how='left')
        print (train_df.info()); print(train_df.head())
    del col_temp; gc.collect()    
    print_memory('after grouping by ip-mobile-day-hour combination')    

    print('grouping by ip-mobileapp-day-hour combination...')
    col_temp = train_df[['ip','mobile_app','day','hour']]. \
        groupby(by=['ip','mobile_app','day'])[['hour']].var(). \
        reset_index().rename(index=str, columns={'hour': 'ip_mobileapp_day_std_hour'})
    gp['ip_mobileapp_day_std_hour'] = col_temp['ip_mobileapp_day_std_hour']        
    if debug == 1: 
        train_df = train_df.merge(col_temp, on=['ip','mobile_app','day'], how='left')
        print (train_df.info()); print(train_df.head())
    del col_temp; gc.collect()    
    print_memory('after grouping by ip-mobileapp-day-hour combination') 

    print('grouping by ip-mobilechannel-day-hour combination...')
    col_temp = train_df[['ip','mobile_channel','day','hour']]. \
        groupby(by=['ip','mobile_channel','day'])[['hour']].var(). \
        reset_index().rename(index=str, columns={'hour': 'ip_mobilechannel_day_std_hour'})
    gp['ip_mobilechannel_day_std_hour'] = col_temp['ip_mobilechannel_day_std_hour']        
    if debug == 1: 
        train_df = train_df.merge(col_temp, on=['ip','mobile_channel','day'], how='left')
        print (train_df.info()); print(train_df.head())
    del col_temp; gc.collect()    
    print_memory('after grouping by ip-mobilechannel-day-hour combination') 

    print('grouping by ip-appchannel-day-hour combination...')
    col_temp = train_df[['ip','app_channel','day','hour']]. \
        groupby(by=['ip','app_channel','day'])[['hour']].var(). \
        reset_index().rename(index=str, columns={'hour': 'ip_appchannel_day_std_hour'})
    gp['ip_appchannel_day_std_hour'] = col_temp['ip_appchannel_day_std_hour']        
    if debug == 1: 
        train_df = train_df.merge(col_temp, on=['ip','app_channel','day'], how='left')
        print (train_df.info()); print(train_df.head())
    del col_temp; gc.collect()    
    print_memory('after grouping by ip-appchannel-day-hour combination') 

    print('grouping by ip-mobile-appchannel-day-hour combination...')
    col_temp = train_df[['ip', 'mobile','app_channel','day','hour']]. \
        groupby(by=['ip','mobile','app_channel','day'])[['hour']].var(). \
        reset_index().rename(index=str, columns={'hour': 'ip_mobile_app_channel_day_std_hour'})
    gp['ip_mobile_app_channel_day_std_hour'] = col_temp['ip_mobile_app_channel_day_std_hour']        
    if debug == 1: 
        train_df = train_df.merge(col_temp, on=['ip','mobile','app_channel','day'], how='left')
        print (train_df.info()); print(train_df.head())
    del col_temp; gc.collect()    
    print_memory('after grouping by ip-mobile-appchannel-day-hour combination') 


    print(gp.head()); print(gp.info()); print(gp.describe())
    gp.to_csv(IP_HOUR_RELATED_FILENAME, index=False)

def create_great_features(train_df):
    print('=======================================================================')
    print('start creating great features...')
    print('=======================================================================')

    # crate mobile, mobile_app, mobile_channel, app_channel
    print('-------------------Step 1----------------------------')
    print('create cat combination...')
    cat_combination_filename = CAT_COMBINATION_FILENAME
    if os.path.exists(cat_combination_filename):
        print('already created, load from', cat_combination_filename)
    else: 
        create_cat_combination(train_df)
    if debug==2:        
        gp = pd.read_csv(cat_combination_filename, dtype=DATATYPE_LIST)
        train_df = pd.concat([train_df, gp], axis=1, join_axes=[train_df.index])
        del gp; gc.collect()
    print_memory('after get cat combination')
    # train_df = convert_categorical_to_number(train_df)
    # print_memory('after map cat to number')

    print('-------------------Step 2----------------------------')
    print('create next click...')
    nextclick_filename = NEXTCLICK_FILENAME
    if os.path.exists(nextclick_filename):
        print('already created, load from', nextclick_filename)
    else:
        create_next_click(train_df)
    if debug==2:        
        gp = pd.read_csv(nextclick_filename, dtype=DATATYPE_LIST)
        train_df = pd.concat([train_df, gp], axis=1, join_axes=[train_df.index])
        del gp; gc.collect()     
    print_memory('after get nextclick')        

    print('-------------------Step 3----------------------------')
    print('create day hour min...') 
    time_filename = TIME_FILENAME 
    if os.path.exists(time_filename):
        print('already created, load from', time_filename)
    else:
        create_day_hour_min(train_df)
    if debug==2:        
        gp = pd.read_csv(time_filename, dtype=DATATYPE_LIST)
        train_df = pd.concat([train_df, gp], axis=1, join_axes=[train_df.index])
        del gp; gc.collect()  
    print_memory('after get day-hour-min')

    print('-------------------Step 4----------------------------')
    print('convert cat combination to number...') 
    gc.collect()
    if not os.path.exists(CAT_COMBINATION_NUMERIC_CATEGORY_FILENAME):
        convert_cat_combination_to_number()
    print('create ip-hour related features...') 
    if os.path.exists(IP_HOUR_RELATED_FILENAME):
        print('already created ip-hour')
    else:
        if os.path.exists(CAT_COMBINATION_NUMERIC_CATEGORY_FILENAME):
            print('load cat combination file...', CAT_COMBINATION_NUMERIC_CATEGORY_FILENAME)
            gp = pd.read_csv(CAT_COMBINATION_NUMERIC_CATEGORY_FILENAME,
                    usecols=['mobile', 'mobile_app', 'mobile_channel', 'app_channel'], dtype=DATATYPE_LIST)
            train_df = pd.concat([train_df, gp], axis=1, join_axes=[train_df.index])
            del gp; gc.collect()
            print_memory('after reading new cat')
            gp = pd.read_csv(TIME_FILENAME, 
                    usecols=['hour', 'day'],dtype=DATATYPE_LIST)
            train_df = pd.concat([train_df, gp], axis=1, join_axes=[train_df.index])
            del gp; gc.collect()
            print_memory('after reading time')
            create_ip_hour_related_features(train_df)
        else: 
            print('please check, there is no file name', CAT_COMBINATION_NUMERIC_CATEGORY_FILENAME)
    print_memory('after get ip related features')

    print('=======================================================================')
    print('end creating great features...')
    print('=======================================================================')

    return train_df

def main():
    train_df = read_train_test('is_merged')
    train_df = convert_to_save_memory(train_df)
    if debug: print(train_df.info())
    print_memory('after convert to save memory')
    train_df = create_great_features(train_df)
    if not debug: print(train_df.info())
    print_memory('after create great features')       
    train_df = convert_to_save_memory(train_df)
    print('Total memory in use: ', process.memory_info().rss/(2**30), ' GB')        
    if debug==2: print(train_df.info()); print(train_df.head())

def main2():
    # train_df = read_train_test('is_merged', 
    #         usecols_train = ['ip', 'app', 'channel', 'device', 'os'], 
    #         usecols_test = ['ip', 'app', 'channel', 'device', 'os'])
    train_df = read_train_test('is_merged', 
            usecols_train = ['ip'], usecols_test = ['ip'])      
    # if debug: 
    print(train_df.info()); print(train_df.head())
    print_memory('after reading data')
    print('reading new cat...')
    gp = pd.read_csv(CAT_COMBINATION_NUMERIC_CATEGORY_FILENAME, 
            dtype=DATATYPE_LIST, )
    train_df = pd.concat([train_df, gp], axis=1, join_axes=[train_df.index])
    del gp; gc.collect()
    print_memory('after reading new cat')
    print('reading time...')
    gp = pd.read_csv(TIME_FILENAME, usecols=['hour', 'day'],dtype=DATATYPE_LIST)
    train_df = pd.concat([train_df, gp], axis=1, join_axes=[train_df.index])
    del gp; gc.collect()
    print_memory('after reading time')
    print(train_df.info()); print(train_df.head())
    create_ip_hour_related_features(train_df)

# convert_cat_combination_to_number()
# main()

main2()
debug=0
print('debug', debug)

import argparse
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
import glob

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
    'hour'              : 'uint8'
    }

DATATYPE_LIST_STRING = {
    'mobile'            : 'category',
    'mobile_app'        : 'category',
    'mobile_channel'    : 'category',
    'app_channel'       : 'category',
    }

REMOVED_LIST = [
    'cat_combination.csv',
    'nextClick.csv',
    'day_hour_min.csv',
    'cat_combination_numeric_category.csv'
    ]

DATATYPE_DICT = {
    'count'     : 'uint32',
    'nunique'   : 'uint32',
    'cumcount'  : 'uint32',
    'var'       : 'float32',
    'std'       : 'float32',
    'confRate'  : 'float32',
    'nextclick' : 'int64',
    'mean'      : 'float32'
    }

if debug==1:
    PATH = '../debug_processed_day9/'        
else:
    PATH = '../processed_full/'                
CAT_COMBINATION_FILENAME = PATH + 'full_cat_combination.csv'
CAT_COMBINATION_NUMERIC_CATEGORY_FILENAME = PATH + 'full_cat_combination_numeric_category.csv'
NEXTCLICK_FILENAME = PATH + 'full_nextClick.csv'
TIME_FILENAME = PATH + 'full_day_hour_min.csv'
IP_HOUR_RELATED_FILENAME = PATH + 'full_ip_hour_related.csv'
TRAINSET_FILENAME = '../input/train.csv'

if debug==2:
    NCHUNKS = 1000
else:    
    NCHUNKS = 100000

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

def convert_to_save_memory(train_df):
    for feature, type in DATATYPE_LIST_UPDATED.items(): 
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
    if debug==2:
        if 'click_time' in usecols_train:
            train_df = pd.read_csv(filename, 
                nrows=10000, dtype=DATATYPE_LIST_UPDATED, parse_dates=['click_time'],
                usecols=usecols_train)
        else:
            train_df = pd.read_csv(filename, 
                nrows=10000, dtype=DATATYPE_LIST_UPDATED, 
                usecols=usecols_train)  
    if debug==1:
        if 'click_time' in usecols_train:
            train_df = pd.read_csv(filename, 
                skiprows=range(1,10), nrows=nrows, dtype=DATATYPE_LIST_UPDATED, parse_dates=['click_time'],
                usecols=usecols_train)
        else:
            train_df = pd.read_csv(filename, 
                skiprows=range(1,10), nrows=nrows, dtype=DATATYPE_LIST_UPDATED, 
                usecols=usecols_train)                
    if debug==0:
        if 'click_time' in usecols_train:
            train_df = pd.read_csv(filename, parse_dates=['click_time'],
                dtype=DATATYPE_LIST_UPDATED, usecols=usecols_train)           
        else:
            train_df = pd.read_csv(filename, 
                dtype=DATATYPE_LIST_UPDATED, usecols=usecols_train)    
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

    print('Extracting new features...')
    gp = pd.DataFrame()
    gp['min'] = pd.to_datetime(train_df.click_time).dt.minute.astype('uint8')
    gp['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    gp['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    gp.to_csv(TIME_FILENAME,index=False)

def print_info(train_df):
    print(train_df.info()) 
    print(train_df.head())

def generate_groupby_by_type_and_columns(train_df, selcols, apply_type):      
    feature_name = ''
    for i in range(len(selcols)-1):
        feature_name = feature_name + selcols[i] + '_'
    feature_name = feature_name + apply_type + '_' + selcols[len(selcols)-1]
    print('>> doing feature:', feature_name)
    filename = PATH + 'full_' + feature_name + '.csv'
    if debug: print_info(train_df)
    if os.path.exists(filename) and debug!=2:
        print ('done already...')
    else:
        if debug: print(selcols); print (len(selcols)-1)
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
        if apply_type == 'mean':
            col_temp = train_df[selcols]. \
                groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].mean(). \
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
        # if debug==2: print(train_df.head()); print(col_temp.head())
        del col_extracted; gc.collect()
    return feature_name        
    # return train_df, feature_name               

def create_kernel_features(train_df):
    new_feature_list=[]
    # get count
    apply_type = 'count'
    selcols = ['ip', 'day', 'mobile']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)

    apply_type = 'count'
    selcols = ['ip', 'day', 'mobile_app']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)

    apply_type = 'count'
    selcols = ['ip', 'day', 'mobile_channel']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)


    # get count
    apply_type = 'count'
    selcols = ['ip', 'day', 'hour' ,'mobile']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)

    apply_type = 'count'
    selcols = ['ip', 'day', 'hour', 'mobile_app']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)

    apply_type = 'count'
    selcols = ['ip', 'day', 'hour', 'mobile_channel']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)

    # get more count
    apply_type = 'count'
    selcols = ['ip', 'day', 'hour']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)

    apply_type = 'count'
    selcols = ['ip', 'day', 'app']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)

    apply_type = 'count'
    selcols = ['ip', 'day','channel']
    feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
    print_memory()
    new_feature_list.append(feature_name)
    return train_df

ATTRIBUTION_CATEGORIES = [        
    # V1 Features #
    ###############
    ['ip'], ['app'], ['device'], ['os'], ['channel'],
    
    # V2 Features #
    ###############
    ['ip', 'channel'],
    ['ip', 'app'],
    ['mobile', 'channel'],
    ['mobile', 'app'],
    ['channel', 'app']
]

def generate_confidence(train_df, cols):
    # Find frequency of is_attributed for each unique value in column
    feature_name = '_'.join(cols)+'_confRate'    
    filename = PATH + 'full_' + feature_name + '.csv'
    
    if os.path.exists(filename) and debug!=2:
        print  ('done already...', filename)
    else:
        # Perform the groupby
        group_object = train_df.groupby(cols)
        
        # Group sizes    
        group_sizes = group_object.size()
        log_group = np.log(200000000) # 10000000 views -> 60% confidence, 100 views -> 40% confidence 
        print(">> Calculating confidence-weighted rate for: {}.\n   Saving to: {}. Group Max /Mean / Median / Min: {} / {} / {} / {}".format(
            cols, filename, 
            group_sizes.max(), 
            np.round(group_sizes.mean(), 2),
            np.round(group_sizes.median(), 2),
            group_sizes.min()
        ))
        
        # Aggregation function
        def rate_calculation(x):
            """Calculate the attributed rate. Scale by confidence"""
            rate = x.sum() / float(x.count())
            conf = np.min([1, np.log(x.count()) / log_group]) 
            return np.rint(rate * conf * 2**32)
            # return np.rint(rate*(2**32))
        
        # Perform the save
        col_extracted = pd.DataFrame()     
        print('merge...') 
        train_df = train_df.merge(
            group_object['is_attributed']. \
                apply(rate_calculation). \
                reset_index(). \
                rename( 
                    index=str,
                    columns={'is_attributed': feature_name}
                )[cols + [feature_name]],
            on=cols, how='left'
        )
        col_extracted[feature_name] = train_df[feature_name]
        print('saving...')
        col_extracted.to_csv(filename, index=False)
        del col_extracted            
        print_memory()

def create_kernel_confidence(train_df):
    for cols in ATTRIBUTION_CATEGORIES:
        generate_confidence(train_df, cols)

GROUP_BY_NEXT_CLICKS = [    
    # V1
    ['ip'],
    ['ip', 'app'],
    ['ip', 'channel'],
    # ['ip', 'device', 'os'],
    
    # V3
    # ['ip', 'os', 'device', 'app'],
    # ['ip', 'os', 'device', 'channel'],
    ['ip', 'os', 'device', 'channel', 'app']
]

def generate_click_anttip(train_df, cols, which_click):   
    # Name of new feature        
    if which_click == 'next':        
        feature_name = '_'.join(cols)+'_nextclick'   
    else:
        feature_name = '_'.join(cols)+'_prevclick'                  
    filename = PATH + 'full_' + feature_name + '.csv'
    print('-----------------------------------------------------')
    print('>> doing feature:', feature_name, 'save to', filename)
    if os.path.exists(filename) and debug!=2:
        print ('done already...', filename)
    else:
        D=2**26
        print('find category...')
        train_df['category'] = ''
        for col in cols:
            train_df['category'] = train_df['category'] + '_' + train_df[col].map(str)
        print_memory()            

        if debug: print (train_df.head())
        train_df['category'] = train_df['category'].apply(hash) % D
        print (train_df.head())
        click_buffer= np.full(D, 3000000000, dtype=np.uint32)
        print_memory()

        print('find epochtime...')
        train_df['epochtime']= train_df['click_time'].astype(np.int64) // 10 ** 9
        col_extracted = pd.DataFrame()
        i = 0
        if which_click == 'next':
            next_clicks= []
            for category, t in zip(reversed(train_df['category'].values), reversed(train_df['epochtime'].values)):
                if i%100000 == 0: print ('process', i); print_memory()
                i = i+1                    
                next_clicks.append(click_buffer[category]-t)
                click_buffer[category]= t
            del(click_buffer); gc.collect()
            col_extracted[feature_name] = list(reversed(next_clicks))
            del next_clicks; gc.collect()
        else:
            prev_clicks= []
            for category, time in zip(train_df['category'].values, train_df['epochtime'].values):
                if i%100000 == 0: print ('process', i); print_memory()
                i = i+1  
                prev_clicks.append(time-click_buffer[category])
                click_buffer[category]= time
            del(click_buffer); gc.collect()
            col_extracted[feature_name] = list(prev_clicks)
            del prev_clicks; gc.collect()
        if debug: print(col_extracted.describe()); print(col_extracted.head()); print(col_extracted.info())

        print('saving...')
        col_extracted.to_csv(filename, index=False)
        del col_extracted            
        print_memory()

def create_kernel_click_anttip(train_df, which_click):
    for cols in GROUP_BY_NEXT_CLICKS:
        generate_click_anttip(train_df, cols, which_click)

MINH_LIST_NUNIQUE =[
    ['ip','mobile','day','hour'],
    ['ip','mobile_app','day','hour'],
    ['ip','mobile_channel','day','hour'],
    ['ip','app_channel','day','hour'],
    ['ip', 'mobile','app_channel','day','hour']
]

MINH_LIST_CUMCOUNT =[
    ['ip','mobile','day','hour'],
    ['ip','mobile_app','day','hour'],
    ['ip','mobile_channel','day','hour'],
    ['ip','app_channel','day','hour'],
    ['ip', 'mobile','app_channel','day','hour']
]

MINH_LIST_STD =[
    ['ip','mobile','day','hour'],
    ['ip','mobile_app','day','hour'],
    ['ip','mobile_channel','day','hour'],
    ['ip','app_channel','day','hour'],
    ['ip', 'mobile','app_channel','day','hour']
]

MINH_LIST_VAR =[
    ['ip','mobile','day','hour'],
    ['ip','mobile_app','day','hour'],
    ['ip','mobile_channel','day','hour'],
    ['ip','app_channel','day','hour'],
    ['ip', 'mobile','app_channel','day','hour']
]

MINH_LIST_COUNT =[
    ['ip','mobile','day','hour'],
    ['ip','mobile_app','day','hour'],
    ['ip','mobile_channel','day','hour'],
    ['ip','app_channel','day','hour'],
    ['ip', 'mobile','app_channel','day','hour']
]

REPLICATE_LIST_NUNIQUE = [
    ['ip', 'channel'],
    ['ip', 'day', 'hour'],
    ['ip', 'app'],
    ['ip', 'app', 'os'],
    ['ip', 'device'],
    ['app', 'channel'],
    ['ip', 'device', 'os', 'app'],
    ['ip','day','hour','channel'],
    ['ip', 'app', 'channel'],
    ['ip','app', 'os', 'channel'],
]

REPLICATE_LIST_COUNT = [
    ['ip','day','hour','channel'],
    ['ip', 'app', 'channel'],
    ['ip','app', 'os', 'channel']
]

REPLICATE_LIST_CUMCOUNT = [
    ['ip', 'device', 'os', 'app'],
    ['ip', 'os']
]

REPLICATE_LIST_VAR = [
    ['ip','day','hour','channel'],
    ['ip', 'os'],
    ['ip','app', 'os', 'hour'],
    ['ip','app', 'channel', 'day'],
]

REPLICATE_LIST_MEAN = [
    ['ip','app', 'channel','hour']
]

def replicate_result(train_df):
    new_feature_list=[]
    # get nunique
    apply_type = 'nunique'
    for selcols in REPLICATE_LIST_NUNIQUE:
        feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
        print_memory()
        new_feature_list.append(feature_name)
    # get nunique
    apply_type = 'cumcount'
    for selcols in REPLICATE_LIST_CUMCOUNT:
        feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
        print_memory()
        new_feature_list.append(feature_name)  
    # get var        
    apply_type = 'var'
    for selcols in REPLICATE_LIST_VAR:
        feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
        print_memory()
        new_feature_list.append(feature_name)          
    # get mean              
    apply_type = 'mean'
    for selcols in REPLICATE_LIST_MEAN:
        feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
        print_memory()
        new_feature_list.append(feature_name)  
    # get count
    apply_type = 'count'
    for selcols in REPLICATE_LIST_COUNT:
        feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
        print_memory()
        new_feature_list.append(feature_name)  

def create_day_hour_min(train_df):
    print('>> Extracting new features...')
    gp = pd.DataFrame()
    gp['min'] = pd.to_datetime(train_df.click_time).dt.minute.astype('uint8')
    gp['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    gp['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    gp.to_csv(TIME_FILENAME,index=False)

def create_time_call(train_df):
    print('>> create day hour min...') 
    time_filename = TIME_FILENAME 
    if os.path.exists(time_filename):
        print('already created, load from', time_filename)
    else:
        create_day_hour_min(train_df)
    if debug==2:        
        gp = pd.read_csv(time_filename, dtype=DATATYPE_LIST_UPDATED)
        train_df = pd.concat([train_df, gp], axis=1, join_axes=[train_df.index])
        del gp; gc.collect()  
    print_memory('after get day-hour-min')

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

def create_cat_combination_call(train_df):
    print('>> create cat combination...')
    cat_combination_filename = CAT_COMBINATION_FILENAME
    if os.path.exists(cat_combination_filename):
        print('already created, load from', cat_combination_filename)
    else: 
        create_cat_combination(train_df)
    if debug==2:        
        gp = pd.read_csv(cat_combination_filename, dtype=DATATYPE_LIST_STRING)
        train_df = pd.concat([train_df, gp], axis=1, join_axes=[train_df.index])
        del gp; gc.collect()
    print_memory('after get cat combination')

def generate_minh_features(train_df):
    new_feature_list=[]
    # get nunique
    apply_type = 'nunique'
    for selcols in MINH_LIST_NUNIQUE:
        feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
        print_memory()
        new_feature_list.append(feature_name)
    # get nunique
    apply_type = 'cumcount'
    for selcols in MINH_LIST_CUMCOUNT:
        feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
        print_memory()
        new_feature_list.append(feature_name)  
    # get var        
    apply_type = 'var'
    for selcols in MINH_LIST_VAR:
        feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
        print_memory()
        new_feature_list.append(feature_name)          
    # get count
    apply_type = 'count'
    for selcols in MINH_LIST_COUNT:
        feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
        print_memory()
        new_feature_list.append(feature_name)  
    # get count
    apply_type = 'std'
    for selcols in MINH_LIST_STD:
        feature_name= generate_groupby_by_type_and_columns(train_df, selcols, apply_type)
        print_memory()
        new_feature_list.append(feature_name)  

def create_cat_combination_to_number_call(train_df):
    print('>> convert cat combination to number...') 
    gc.collect()
    if not os.path.exists(CAT_COMBINATION_NUMERIC_CATEGORY_FILENAME):
        convert_cat_combination_to_number()

def get_datatype(feature_name):
    datatype = 'UNKNOWN'
    for key, type in DATATYPE_DICT.items():
        if key in feature_name:
            datatype = type
            break
    return datatype           

def update_datatype_dict():
    datatype_list = DATATYPE_LIST
    files = glob.glob(PATH + "*.csv") 
    print (files)
    if debug:
        PATH_corrected = PATH.replace('full/', 'full\\') 
        removed_string = PATH_corrected + 'full_'
    else:
        PATH_corrected = PATH
        removed_string = PATH_corrected + 'full_'        
    print(removed_string)
    for file in files:
        feature_name = file.replace(removed_string,'')
        if feature_name not in REMOVED_LIST:
            feature_name = feature_name.split('.')[0]
            feature_type = get_datatype(feature_name)
            # print('feature {} has datatype {}'.format(feature_name, feature_type))
            datatype_list[feature_name] = feature_type
    return datatype_list

# DATATYPE_LIST_UPDATED = update_datatype_dict()
DATATYPE_LIST_UPDATED = DATATYPE_LIST

def extend_df_with_time_cat(train_df):
    print('>> load cat combination file...', CAT_COMBINATION_NUMERIC_CATEGORY_FILENAME)
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

    if debug: print(train_df.info())
    return train_df

def extend_df_with_time(train_df):
    gp = pd.read_csv(TIME_FILENAME, 
            usecols=['hour', 'day'],dtype=DATATYPE_LIST)
    train_df = pd.concat([train_df, gp], axis=1, join_axes=[train_df.index])
    del gp; gc.collect()
    print_memory('after reading time')

    if debug: print(train_df.info())
    return train_df

def extend_df_with_cat(train_df):
    print('>> load cat combination file...', CAT_COMBINATION_NUMERIC_CATEGORY_FILENAME)
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
    if debug: print(train_df.info())
    return train_df

def main():
    train_df = read_train_test('is_merged')
    if debug: print(train_df.info())
    print('>> reading time...')    
    print('------------------------------------------------------') 
    create_time_call(train_df)
    print('------------------------------------------------------')
    create_cat_combination_call(train_df)
    print('------------------------------------------------------')
    create_cat_combination_to_number_call(train_df)
    print('------------------------------------------------------')
    print('>> extend df with time and new cat...')
    # train_df = extend_df_with_time_cat(train_df)
    train_df = extend_df_with_time(train_df)
    print('------------------------------------------------------')
    print('>> create kernel features...')
    create_kernel_features(train_df)
    print('------------------------------------------------------')
    print('>> create kernel confidence...')
    create_kernel_confidence(train_df)
    print('------------------------------------------------------')
    print('>> create replicate result...')
    replicate_result(train_df)
    print('------------------------------------------------------')
    print('>> create generate minh features...')
    generate_minh_features(train_df)    
    print('------------------------------------------------------')
    print('>> create kernel click anttip...')
    create_kernel_click_anttip(train_df, 'next')
main()
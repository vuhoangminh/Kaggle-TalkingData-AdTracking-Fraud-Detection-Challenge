debug=1
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
    'ip_mobile_app_channel_day_nunique_hour'    : 'uint32',
    'ip_mobile_app_channel_day_nunique_hour'    : 'uint32',
    'ip_confRate'                               : 'float32',
    'app_confRate'                              : 'float32',
    'device_confRate'                           : 'float32',
    'os_confRate'                               : 'float32',
    'channel_confRate'                          : 'float32',
    'ip_channel_confRate'                       : 'float32',
    'ip_app_confRate'                           : 'float32',
    'mobile_channel_confRate'                   : 'float32',
    'mobile_app_confRate'                       : 'float32',
    'channel_app_confRate'                      : 'float32',
    }


if debug:
    PATH = '../debug_processed_day9/'        
else:
    PATH = '../processed_day9/'                
CAT_COMBINATION_FILENAME = PATH + 'day9_cat_combination.csv'
CAT_COMBINATION_NUMERIC_CATEGORY_FILENAME = PATH + 'day9_cat_combination_numeric_category.csv'
NEXTCLICK_FILENAME = PATH + 'day9_nextClick.csv'
TIME_FILENAME = PATH + 'day9_day_hour_min.csv'
IP_HOUR_RELATED_FILENAME = PATH + 'day9_ip_hour_related.csv'
TRAINSET_FILENAME = '../input/valid_day_9.csv'
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
    print('>> doing feature:', feature_name)
    filename = PATH + 'day9_' + feature_name + '.csv'
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
    filename = PATH + 'day9_' + feature_name + '.csv'
    
    if os.path.exists(filename):
        print  ('done already...', filename)
    else:
        # Perform the groupby
        group_object = train_df.groupby(cols)
        
        # Group sizes    
        group_sizes = group_object.size()
        log_group = np.log(100000) # 1000 views -> 60% confidence, 100 views -> 40% confidence 
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
            return rate * conf
        
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
    ['ip', 'device', 'os'],
    
    # V3
    ['ip', 'os', 'device', 'app'],
    ['ip', 'os', 'device', 'channel'],
    ['ip', 'os', 'device', 'channel', 'app']
]

def generate_click(train_df, spec, which_click):   
    # Name of new feature
        # Name of new feature
    if which_click == 'next':        
        feature_name = '{}_nextClick'.format('_'.join(spec['groupby']))    
    else:
        feature_name = '{}_prevClick'.format('_'.join(spec['groupby']))                    
    filename = PATH + 'day9_' + feature_name + '.csv'
    
    if os.path.exists(filename):
        print ('done already...', filename)
    else:
        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']
        
        # Run calculation
        print(">> Grouping by: {}.\n Saving to next click in {}". \
            format(
                spec['groupby'],
                filename
            ))
        col_extracted = pd.DataFrame()
        if which_click == 'next':     
            col_extracted[feature_name] = train_df[all_features]. \
                    groupby(spec['groupby']).click_time. \
                    transform(lambda x: x.diff().shift(-1)).dt.seconds
        else:
            col_extracted[feature_name] = train_df[all_features]. \
                    groupby(spec['groupby']).click_time. \
                    transform(lambda x: x.diff().shift(1)).dt.seconds

        print('saving...')
        col_extracted.to_csv(filename, index=False)
        del col_extracted            
        print_memory()

def create_kernel_click(train_df, which_click):
    for spec in GROUP_BY_NEXT_CLICKS:
        generate_click(train_df, spec, which_click)

def generate_click_anttip(train_df, spec, which_click):   
    # Name of new feature
    feature_name = '_'.join(cols)+'_confRate'            
    if which_click == 'next':        
        feature_name = '_'.join(cols)+'_confRate'   
    else:
        feature_name = '_'.join(cols)+'_confRate'                  
    filename = PATH + 'day9_' + feature_name + '.csv'
    
    if os.path.exists(filename):
        print ('done already...', filename)
    else:
        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']
        
        # Run calculation
        print(">> Grouping by: {}.\n Saving to next click in {}". \
            format(
                spec['groupby'],
                filename
            ))
        col_extracted = pd.DataFrame()
        if which_click == 'next':     
            col_extracted[feature_name] = train_df[all_features]. \
                    groupby(spec['groupby']).click_time. \
                    transform(lambda x: x.diff().shift(-1)).dt.seconds
        else:
            col_extracted[feature_name] = train_df[all_features]. \
                    groupby(spec['groupby']).click_time. \
                    transform(lambda x: x.diff().shift(1)).dt.seconds

        print('saving...')
        col_extracted.to_csv(filename, index=False)
        del col_extracted            
        print_memory()


def create_kernel_click_anttip(train_df, which_click, selcols):
    D= 2**26
    df['category'] = (df['ip'].astype(str) + "_" + df['app'].astype(str) + "_" + df['device'].astype(str) \
                        + "_" + df['os'].astype(str)).apply(hash) % D
    click_buffer= np.full(D, 3000000000, dtype=np.uint32)
    df['epochtime']= df['click_time'].astype(np.int64) // 10 ** 9
    next_clicks= []
    for category, time in zip(reversed(df['category'].values), reversed(df['epochtime'].values)):
        next_clicks.append(click_buffer[category]-time)
        click_buffer[category]= time
    del(click_buffer)
    df['next_click']= list(reversed(next_clicks))

def main():
    train_df = read_train_test('is_merged')
    if debug: print(train_df.info())
    print('load cat combination file...', CAT_COMBINATION_NUMERIC_CATEGORY_FILENAME)
    gp = pd.read_csv(CAT_COMBINATION_NUMERIC_CATEGORY_FILENAME,
            usecols=['mobile', 'mobile_app', 'mobile_channel', 'app_channel'], 
            dtype=DATATYPE_LIST)
    train_df = pd.concat([train_df, gp], axis=1, join_axes=[train_df.index])
    del gp; gc.collect()
    print_memory('after reading new cat')
    gp = pd.read_csv(TIME_FILENAME, 
            usecols=['hour', 'day'],dtype=DATATYPE_LIST)
    train_df = pd.concat([train_df, gp], axis=1, join_axes=[train_df.index])
    del gp; gc.collect()
    print_memory('after reading time')
    # train_df = create_great_features(train_df)
    create_kernel_features(train_df)
    create_kernel_click(train_df, 'prev')
    create_kernel_click(train_df, 'next')
    
    
    # create_kernel_confidence(train_df)

    if debug: print(train_df.info())
    print_memory('after create great features')       

main()
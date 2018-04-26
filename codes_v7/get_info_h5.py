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
    'ip_mobile_day_count_hour'                  : 'uint32',
    'ip_mobile_app_day_count_hour'              : 'uint32',
    'ip_mobile_channel_day_count_hour'          : 'uint32',
    'ip_app_channel_day_count_hour'             : 'uint32',
    'ip_mobile_app_channel_day_count_hour'      : 'uint32',
    'ip_mobile_day_var_hour'                    : 'float16',
    'ip_mobile_app_day_var_hour'                : 'float16',
    'ip_mobile_channel_day_var_hour'            : 'float16',
    'ip_app_channel_day_var_hour'               : 'float16',
    'ip_mobile_app_channel_day_var_hour'        : 'float16',
    'ip_mobile_day_std_hour'                    : 'float16',
    'ip_mobile_app_day_std_hour'                : 'float16',
    'ip_mobile_channel_day_std_hour'            : 'float16',
    'ip_app_channel_day_std_hour'               : 'float16',
    'ip_mobile_app_channel_day_std_hour'        : 'float16',
    'ip_mobile_day_cumcount_hour'               : 'uint32',
    'ip_mobile_app_day_cumcount_hour'           : 'uint32',
    'ip_mobile_channel_day_cumcount_hour'       : 'uint32',
    'ip_app_channel_day_cumcount_hour'          : 'uint32',
    'ip_mobile_app_channel_day_cumcount_hour'   : 'uint32',
    'ip_mobile_day_nunique_hour'                : 'uint32',
    'ip_mobile_app_day_nunique_hour'            : 'uint32',
    'ip_mobile_channel_day_nunique_hour'        : 'uint32',
    'ip_app_channel_day_nunique_hour'           : 'uint32',
    'ip_mobile_app_channel_day_nunique_hour'    : 'uint32'
    }


process = psutil.Process(os.getpid())
def print_memory(print_string=''):
    print('Total memory in use ' + print_string + ': ', process.memory_info().rss/(2**30), ' GB')

print('-----------------------------------------------------')
print('reading train')
train_h5 = pd.HDFStore('train_day9.h5')
print(train_h5)
train_df = train_h5.select('train') 
print(train_df.info()); print(train_df.head())
print_memory()

print('-----------------------------------------------------')
print('reading test')
test_h5 = pd.HDFStore('test_day9.h5')
print(test_h5)
test_df = test_h5.select('test') 
print(test_df.info()); print(test_df.head())
print_memory()




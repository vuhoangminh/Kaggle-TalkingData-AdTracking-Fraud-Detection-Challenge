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
import datetime


print('prepare predictors...')
predictors=[]
new_feature = 'nextClick'
predictors.append(new_feature)
predictors.append(new_feature + '_shift')
predictors.extend(['app','device','os', 'channel', 'hour', 'day', 
            'ip_tcount', 'ip_tchan_count', 'ip_app_count',
            'ip_app_os_count', 'ip_app_os_var',
            'ip_app_channel_var_day','ip_app_channel_mean_hour',
            'mode_channel_ip', 'mode_device_ip', 'mode_os_ip', 'mode_app_ip',
            'numunique_device_ip','mode_channel_app','mode_device_app','mode_os_app',
            'mode_ip_app','mode_channel_device','mode_app_device','mode_os_device',
            'mode_ip_device','mode_channel_os','mode_app_os','mode_device_os',
            'mode_ip_os','numunique_device_os','mode_os_channel','mode_app_channel',
            'mode_device_channel','mode_ip_channel','numunique_device_channel',
            'numunique_app_channel','category','epochtime'])

categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']
naddfeat=9
for i in range(0,naddfeat):
    predictors.append('X'+str(i))

print('--------------------------------------')
print('predictors',predictors)
print('number of features:', len(predictors))

target = 'is_attributed'

predictors_removed = ['mode_os_app','mode_os_device','mode_app_os','mode_device_os',
            'numunique_app_channel','X3','X5','epochtime','ip_app_count',
            'ip_app_channel_mean_hour']

print('--------------------------------------')
print('highly correlated features:',predictors_removed)
print('number of removed features:', len(predictors_removed))

predictors_kept = predictors
for item in predictors_removed:
    while predictors_kept.count(item) > 0:
        predictors_kept.remove(item)

# predictors_kept = predictors-predictors_removed

print('--------------------------------------')
print('highly correlated features:',predictors_kept)
print('number of removed features:', len(predictors_kept))
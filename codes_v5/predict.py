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

now = datetime.datetime.now()
if now.month<10:
    month_string = '0'+str(now.month)
else:
    month_string = str(now.month)
if now.day<10:
    day_string = '0'+str(now.day)
else:
    day_string = str(now.day)
yearmonthdate_string = str(now.year) + month_string + day_string

frm=10
to=180000010
process = psutil.Process(os.getpid())

boosting_type = 'gbdt'
# boosting_type = 'dart'
frac = 0.5

debug = 0
predictors_removed = ['mode_channel_ip', 'mode_device_ip', 'mode_os_ip', 'mode_app_ip',
            'numunique_device_ip','mode_channel_app','mode_device_app','mode_os_app',
            'mode_ip_app','mode_channel_device','mode_app_device','mode_os_device',
            'mode_ip_device','mode_channel_os','mode_app_os','mode_device_os',
            'mode_ip_os','numunique_device_os','mode_os_channel','mode_app_channel',
            'mode_device_channel','mode_ip_channel','numunique_device_channel',
            'numunique_app_channel','epochtime', 'category',
            'X3','ip_tcount','day', 'ip', 'click_id']  

def get_predictors():
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
    naddfeat=9
    for i in range(0,naddfeat):
        predictors.append('X'+str(i))
    if debug:
        print('--------------------------------------')
        print('predictors',predictors)
        print('number of features:', len(predictors))
                
    if debug:
        print('--------------------------------------')
        print('highly correlated features:',predictors_removed)
        print('number of removed features:', len(predictors_removed))

    predictors_kept = predictors
    for item in predictors_removed:
        while predictors_kept.count(item) > 0:
            predictors_kept.remove(item)
    if debug:
        print('--------------------------------------')
        print('kept features:',predictors_kept)
        print('number of kept features:', len(predictors_kept))   
    return predictors_kept 


def predict(modelname,subfilename,num_iteration):
    # load model to predict
    print('Load model to predict')
    bst = lgb.Booster(model_file=modelname)

    # can only predict with the best iteration (or the saving iteration)
    print('prepare predictors...')
    predictors = get_predictors()
    print('predictors:',predictors)
  
    save_name='test_%d_%d'%(frm,to)
    test_df = pd.read_pickle(save_name)
    print('Total memory in use after reading test: ', process.memory_info().rss/(2**30), ' GB\n')
    print("test size : ", len(test_df))
    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')

    print("Predicting...")
    sub['is_attributed'] = bst.predict(test_df[predictors],num_iteration=num_iteration)
    # if not debug:
    print("writing...")
    sub.to_csv(subfilename,index=False,compression='gzip')
    print("done...")


# num_leaves_list = [7,9,11,13,15,31,31,9]
# max_depth_list = [3,4,5,6,7,5,6,5]
num_leaves_list = [7,9,11,13,15]
max_depth_list = [3,4,5,6,7]
num_iteration_list = [327,225,203,154,110]

predictors = get_predictors()
print(predictors)


for i in range(len(num_leaves_list)):
    print ('==============================================================')
    # num_leaves = num_leaves_list[len(num_leaves_list)-1-i]
    # max_depth = max_depth_list[len(num_leaves_list)-1-i]
    num_leaves = num_leaves_list[i]
    max_depth = max_depth_list[i]
    num_iteration = num_iteration_list[i]
    print('num leaves:', num_leaves)
    print('max depth:', max_depth)
    print('iteration:', num_iteration)
    predictors = get_predictors()
    subfilename = yearmonthdate_string + '_' + str(len(predictors)) + \
            'features_' + boosting_type + '_sub_it_' + str(int(100*frac)) + \
            'percent_3days_%d_%d.csv.gz'%(num_leaves,max_depth)
    modelfilename = yearmonthdate_string + '_' + str(len(predictors)) + \
            'features_' + boosting_type + '_sub_it_' + str(int(100*frac)) + \
            'percent_3days_%d_%d'%(num_leaves,max_depth) + '.txt'           
    predict(modelfilename,subfilename,num_iteration)   
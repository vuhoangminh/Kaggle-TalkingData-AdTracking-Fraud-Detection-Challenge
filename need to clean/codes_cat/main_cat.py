from catboost import CatBoostClassifier, Pool, cv, CatBoostRegressor
from sklearn.metrics import accuracy_score
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

input = '../codes_v3/'
frm = 1000
to = 10001000
# frm = 1000
# to = 1001000
frac = 0.5
debug = False

if debug:
    NROUNDS = 1000
else:
    NROUNDS = 6500  # Warning: needs to run overnight on multiple cores

print('read test...')
save_name = 'test_%d_%d'%(frm,to)
test_df = pd.read_pickle(save_name)
print('Total memory in use after reading test: ', process.memory_info().rss/(2**30), ' GB\n')
print('read val...')
save_name = 'val_%d_%d'%(frm,to)
val_df = pd.read_pickle(save_name)
val_df = val_df.sample(frac=frac)
print('Total memory in use after reading val: ', process.memory_info().rss/(2**30), ' GB\n')
print('read train...')
save_name = 'train_%d_%d'%(frm,to)
train_df = pd.read_pickle(save_name)
# train_df = val_df
train_df = train_df.sample(frac=frac)
print('Total memory in use after reading train: ', process.memory_info().rss/(2**30), ' GB\n')



dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32',
        }


print('doing nextClick')
predictors=[]

new_feature = 'nextClick'
filename='nextClick_%d_%d.csv'%(frm,to)


predictors.append(new_feature)
predictors.append(new_feature+'_shift')


target = 'is_attributed'
predictors.extend(['app','device','os', 'channel', 'hour', 'day', 
                'ip_tcount', 'ip_tchan_count', 'ip_app_count',
                'ip_app_os_count', 'ip_app_os_var',
                'ip_app_channel_var_day','ip_app_channel_mean_hour'])
categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']
categorical_features_indices = [2,3,4,5,6,7]
naddfeat=9
for i in range(0,naddfeat):
    predictors.append('X'+str(i))
    
print('predictors',predictors)

# Initialize params
params = {
    # 'iterations':NROUNDS, 
    # 'learning_rate':0.1, 
    # 'depth':10, 
    # 'loss_function':'RMSE',
    # 'eval_metric':'AUC',
    # 'use_best_model': True,
    # 'depth':10, 
    'scale_pos_weight':200
    }

model = CatBoostRegressor(
    # params,
    iterations=NROUNDS, 
    learning_rate=0.1,
    use_best_model = True,
    depth=10, 
    l2_leaf_reg=14,
    bagging_temperature=8,
    # loss_function='MAE',
    loss_function='RMSE',
    eval_metric='AUC',
    random_seed=i, 
    od_type='Iter', 
    od_wait=30
)

# model = CatBoostRegressor(iterations=1000, learning_rate=0.1, 
#     eval_metric='AUC', loss_function='CrossEntropy',
#     class_weights=[200, 1], scale_pos_weight=200,
#     od_type='Iter', od_wait=30)

X_train = train_df[predictors]
y_train = train_df[target]
X_validation = val_df[predictors]
y_validation = val_df[target]
model.fit(
    X_train, y_train,
    cat_features=categorical_features_indices,
    eval_set=(X_validation, y_validation),
    logging_level='Verbose',  # you can uncomment this for text output
    # sample_weight = [200,1],
    plot=False
)
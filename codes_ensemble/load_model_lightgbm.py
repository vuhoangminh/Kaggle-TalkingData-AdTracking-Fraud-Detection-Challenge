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
# boosting_type = 'rf'
boosting_type = 'gbdt'
# boosting_type = 'dart'
frac = 0.5
num_leaves = 31
max_depth = 6
noday = True

frm=10
to=180000010

debug=0
if debug:
    print('*** debug parameter set: this is a test run for debugging purposes ***')

savename = '../codes_cat/test_1000_1001000' 
test_df = pd.read_pickle(savename)
test_df = test_df.loc[:, test_df.columns != 'click_time']
# load model to predict
print('Load model to predict')
bst = lgb.Booster(model_file='gbdt_removeday_sub_it_50percent_10_180000010_31_6.txt')
# can only predict with the best iteration (or the saving iteration)
sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')
sub['is_attributed'] = bst.predict(test_df)

print("writing...")
subfilename = 'sub.csv'
sub.to_csv(subfilename,index=False)
print("done...")
# eval with loaded model
# print('The rmse of loaded model\'s prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
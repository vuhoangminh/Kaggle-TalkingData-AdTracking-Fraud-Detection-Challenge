"""
Adding improvements inspired from:
"""

import pandas as pd
import time
import numpy as np
from numpy import random
from sklearn.cross_validation import train_test_split
import json
import lightgbm as lgb
import gc


prediction0 = 'lightgbm_v19_0.csv'
prediction1 = 'lightgbm_v19_1.csv'
prediction2 = 'lightgbm_v19_2.csv'

print ('read 0...')
df0 = pd.read_csv(prediction0)
print ('read 1...')
df1 = pd.read_csv(prediction1)
print ('read 2...')
df2 = pd.read_csv(prediction2)

p = pd.Panel({n: df for n, df in enumerate([df0, df1, df2])})

# m = pd.DataFrame()
# m['click_id'] = df0['click_id'].astype('int')
m = p.mean(axis=0)
m['click_id'] = m['click_id'].astype('int')

print (df0.head())
print (df1.head())
print (df2.head())
print (m.head())



save_name = 'lightgbm_v19_average.csv'
print("save to: ", save_name)
print("writing...")
m.to_csv(save_name, index=False)
print("done...")
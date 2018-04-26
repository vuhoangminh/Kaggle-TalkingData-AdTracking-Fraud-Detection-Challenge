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
import datetime
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

SEED = 1988
process = psutil.Process(os.getpid())

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

boosting_type = 'gbdt'
# boosting_type = 'dart'
frac = 0.7

frm=10
to=180000010

print('reading test')
test_h5 = pd.HDFStore('test_day9.h5')
print(test_h5)
test_df = test_h5.select('test') 

sns.heatmap(test_df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(50,50)
plt.show()
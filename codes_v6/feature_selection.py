import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
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


debug=0
print('debug', debug)
  

if debug==1:
    DATASET = 'day9'
else:    
    DATASET = 'full'

print(DATASET)    

if debug==0:
    print('=======================================================================')
    print('process on server...')
    print('=======================================================================')
if debug==1:
    print('=======================================================================')
    print('for testing only...')
    print('=======================================================================')
if debug==2:
    print('=======================================================================')
    print('for LIGHT TEST only...')
    print('=======================================================================')


if debug==2:
    START_POINT = 0 
    END_POINT = 10
if debug==1:
    START_POINT = 0 
    END_POINT = 1000
if debug==0:
    START_POINT = 0 
    END_POINT = 100000000


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
import h5py
from sklearn.svm import SVC, SVR
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt


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

process = psutil.Process(os.getpid())

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

TRAIN_HDF5 = 'train_' + DATASET + '.h5'
TEST_HDF5 = 'test_' + DATASET + '.h5'

DATATYPE_LIST_STRING = {
    'mobile'            : 'category',
    'mobile_app'        : 'category',
    'mobile_channel'    : 'category',
    'app_channel'       : 'category',
    }

if debug==1:
    PATH = '../debug_processed_day9/'        
else:
    PATH = '../processed_full/'                
CAT_COMBINATION_FILENAME = PATH + DATASET + '_cat_combination.csv'
CAT_COMBINATION_NUMERIC_CATEGORY_FILENAME = PATH + DATASET + '_cat_combination_numeric_category.csv'
NEXTCLICK_FILENAME = PATH + DATASET + '_nextClick.csv'
TIME_FILENAME = PATH + DATASET + '_day_hour_min.csv'
IP_HOUR_RELATED_FILENAME = PATH + DATASET + '_ip_hour_related.csv'
if debug==1:
    TRAINSET_FILENAME = '../input/valid_day_9.csv'
else:
    TRAINSET_FILENAME = '../input/train.csv'        

if not debug:
    print('=======================================================================')
    print('process on server...')
    print('=======================================================================')
else:
    print('=======================================================================')
    print('for testing only...')
    print('=======================================================================')

SIZE_TRAIN = 53016937
SIZE_TEST = 18790469

def print_memory(print_string=''):
    print('Total memory in use ' + print_string + ': ', process.memory_info().rss/(2**30), ' GB')

def get_keys_h5(f):
    return [key for key in f.keys()]

# DATATYPE_DICT = {
#     'count',     
#     'nunique',   
#     'cumcount',  
#     'var'      ,
#     'std'       ,
#     'confRate' ,
#     'nextclick' ,
#     'nextClick',
#     'nextClick_shift'
#     }

DATATYPE_DICT = {
    }

def is_processed(feature):
    is_processed = True
    for key in DATATYPE_DICT:
        if key in feature:
            is_processed = False
    return is_processed            

DATATYPE_DICT_CONVERT = [
    'count',
    'nunique',
    'cumcount'
]
def read_processed_h5(start_point, end_point, filename):
    with h5py.File(filename,'r') as hf:
        feature_list = list(hf.keys())
    train_df = pd.DataFrame()
    t0 = time.time()
    for feature in feature_list:
        if feature!='dump_later' and not is_processed(feature) or feature=='is_attributed' :
        # if feature!='dump_later':
            print('>> adding', feature)
            is_convert = False
            for key in DATATYPE_DICT_CONVERT:
                if key in feature:
                    is_convert = True
                    print('need to convert', feature, 'to int to save memory')
            if is_convert:
                df_temp = pd.DataFrame()
                df_temp[feature] = pd.read_hdf(filename, key=feature,
                        start=start_point, stop=end_point)
                print('min anc max before:', df_temp[feature].min(), df_temp[feature].max())                        
                df_temp = df_temp.fillna(0)
                df_temp[feature] = df_temp[feature].astype('uint32')                        
                if debug: print(df_temp.dtypes); print(df_temp.describe())
                print('min anc max before:', df_temp[feature].min(), df_temp[feature].max())
                if debug: print(df_temp.dtypes); print(df_temp.describe())
                train_df[feature] = df_temp[feature]   
                del df_temp; gc.collect
            else:
                train_df[feature] = pd.read_hdf(filename, key=feature,
                        start=start_point, stop=end_point)   
            print_memory()
    t1 = time.time()
    total = t1-t0
    print('total reading time:', total)
    return train_df





DATATYPE_DICT = {
    'count',     
    'nunique',   
    'cumcount',  
    'var'      ,
    'std'       ,
    'confRate' ,
    'nextclick' ,
    'nextClick',
    'nextClick_shift'
    }


def find_predictors(filename, expected_num_feature):
    train_df = read_processed_h5(start_point, end_point, filename)
    predictors = []
    features = list(train_df)
    if debug: 
        for feature in features and not is_processed(feature): 
            print (feature)
    return predictors


# find_predictors(TRAIN_HDF5, 5)
start_point = START_POINT
end_point = END_POINT
train_df = read_processed_h5(start_point, end_point, TRAIN_HDF5)
train_df = train_df.fillna(0)
print(train_df.info())


if 'click_time' in train_df:
    train_df = train_df.drop(['click_time'], axis=1)
print(train_df.info())
print(train_df.head())

print('>> prepare dataset...')


# # 1. DATA CLEANSING AND ANALYSIS
house = train_df


# # 2. Stability Selection via Randomized Lasso
# First extract the target variable which is our House prices
y = train_df['is_attributed']
Y = y


x = train_df.loc[:, train_df.columns != 'is_attributed']
X = x
colnames = X.columns
print_memory()


# Define dictionary to store our rankings
ranks = {}
# Create our function which stores the feature rankings to the ranks dictionary
def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))


print('>> run rlasso/Stability')
# Finally let's run our Selection Stability method with Randomized Lasso
rlasso = RandomizedLasso(alpha=0.04)
rlasso.fit(X, Y)
ranks["rlasso/Stability"] = ranking(np.abs(rlasso.scores_), colnames)
print('finished')
print_memory()


# # 3. Recursive Feature Elimination ( RFE )

# Construct our Linear Regression model
print('>> run RFE')
lr = LinearRegression(normalize=True)
lr.fit(X,Y)
#stop the search when only the last feature is left
rfe = RFE(lr, n_features_to_select=20, verbose =3)
rfe.fit(X,Y)
ranks["RFE"] = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)
print('finished')
print_memory()
# print(ranks['RFE'])


# # 4. Linear Model Feature Ranking
# Using Linear Regression
print('>> run LinReg')
lr = LinearRegression(normalize=True)
lr.fit(X,Y)
ranks["LinReg"] = ranking(np.abs(lr.coef_), colnames)
print_memory()


# Using Ridge 
print('>> run Ridge')
ridge = Ridge(alpha = 7)
ridge.fit(X,Y)
ranks['Ridge'] = ranking(np.abs(ridge.coef_), colnames)
print_memory()

# Using Lasso
print('>> run Lasso')
lasso = Lasso(alpha=.05)
lasso.fit(X, Y)
ranks["Lasso"] = ranking(np.abs(lasso.coef_), colnames)
print_memory()


# # # 5. Random Forest feature ranking
print('>> run RF')

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
forest = ExtraTreesClassifier(n_estimators=50, random_state=0, verbose=3)
forest.fit(X,Y)
ranks["RF"] = ranking(forest.feature_importances_, colnames)
print_memory()

# rf = RandomForestRegressor(n_jobs=-1, n_estimators=100, verbose=3)
# rf.fit(X,Y)
# ranks["RF"] = ranking(rf.feature_importances_, colnames)
# print_memory()


# # 6. Creating the Feature Ranking Matrix
# Create empty dictionary to store the mean value calculated from all the scores
print('>> final step')
r = {}
for name in colnames:
    r[name] = round(np.mean([ranks[method][name] 
                             for method in ranks.keys()]), 2)
    print (name)                                 
 
methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")
 
print("\t%s" % "\t".join(methods))
for name in colnames:
    print("%s\t%s" % (name, "\t".join(map(str, 
                         [ranks[method][name] for method in methods]))))

df = pd.DataFrame(ranks)
# print (df)
df.to_csv('rank_10000000_new.csv')

# Put the mean scores into a Pandas dataframe
meanplot = pd.DataFrame(list(r.items()), columns= ['Feature','Mean Ranking'])

# Sort the dataframe
meanplot = meanplot.sort_values('Mean Ranking', ascending=False)

# Let's plot the ranking of the features
sns.factorplot(x="Mean Ranking", y="Feature", data = meanplot, kind="bar", 
               size=14, aspect=1.9, palette='coolwarm')

fig=plt.gcf()
fig.set_size_inches(50,50)
savename = yearmonthdate_string + '_' + '_feature_ranking_10000000_new.png'
plt.savefig(savename)
print('done')
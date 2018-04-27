import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor


debug=0
print('debug', debug)
  

if debug==2:
    DATASET = 'full'  
else:    
    DATASET = 'day9'

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


if debug==1:
    start_point = 0 
    end_point = 10
if debug==1:
    start_point = 0 
    end_point = 5000
if debug==0:
    start_point = 0 
    end_point = 100000


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


def read_processed_h5(start_point, end_point, filename):
    with h5py.File(filename,'r') as hf:
        feature_list = list(hf.keys())
    train_df = pd.DataFrame()
    t0 = time.time()
    for feature in feature_list:
        if feature!='dump_later' and not is_processed(feature) or feature=='is_attributed' :
        # if feature!='dump_later':
            print('>> adding', feature)
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

train_df = read_processed_h5(start_point, end_point, TRAIN_HDF5)
train_df = train_df.fillna(0)
print(train_df.info())


if 'click_time' in train_df:
    train_df = train_df.drop(['click_time'], axis=1)
print(train_df.info())
print(train_df.head())

print('>> prepare dataset...')


# # 1. DATA CLEANSING AND ANALYSIS
# 
# Let's first read in the house data as a dataframe "house" and inspect the first 5 rows

# In[ ]:




house = train_df





# **Pairplot Visualisation**
# 
# Let's create some Seaborn pairplots for the features ('sqft_lot','sqft_above','price','sqft_living','bedrooms') to get a feel for how the various features are distributed vis-a-vis the price as well as the number of bedrooms

# In[ ]:


#sns.pairplot(house[['sqft_lot','sqft_above','price','sqft_living','bedrooms']], hue='bedrooms', palette='afmhot',size=1.4)



# # 2. Stability Selection via Randomized Lasso
# First extract the target variable which is our House prices
y = train_df['is_attributed']
Y = y


x = train_df.loc[:, train_df.columns != 'is_attributed']

# Drop price from the house dataframe and create a matrix out of the house data

X = x
# Store the column/feature names into a list "colnames"
colnames = house.columns
print_memory()
# Next, we create a function which will be able to conveniently store our feature rankings obtained from the various methods described here into a Python dictionary. In case you are thinking I created this function, no this is not the case. All credit goes to Ando Saabas and I am only trying to apply what he has discussed in the context of this dataset.

# In[ ]:


# Define dictionary to store our rankings
ranks = {}
# Create our function which stores the feature rankings to the ranks dictionary
def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))


# In[ ]:


# Finally let's run our Selection Stability method with Randomized Lasso
rlasso = RandomizedLasso(alpha=0.04)
rlasso.fit(X, Y)
ranks["rlasso/Stability"] = ranking(np.abs(rlasso.scores_), colnames)
print('finished')







# # 3. Recursive Feature Elimination ( RFE )

# Construct our Linear Regression model
lr = LinearRegression(normalize=True)
lr.fit(X,Y)
#stop the search when only the last feature is left
rfe = RFE(lr, n_features_to_select=20, verbose =3)
rfe.fit(X,Y)
ranks["RFE"] = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)


# # # 4. Linear Model Feature Ranking
# # 
# # Now let's apply 3 different linear models (Linear, Lasso and Ridge Regression) and how the features are selected and prioritised via these models. To achieve this, I shall use the sklearn implementation of these models and in particular the attribute .coef to return the estimated coefficients for each feature in the linear model.

# # In[ ]:


# # Using Linear Regression
# lr = LinearRegression(normalize=True)
# lr.fit(X,Y)
# ranks["LinReg"] = ranking(np.abs(lr.coef_), colnames)

# # Using Ridge 
# ridge = Ridge(alpha = 7)
# ridge.fit(X,Y)
# ranks['Ridge'] = ranking(np.abs(ridge.coef_), colnames)

# # Using Lasso
# lasso = Lasso(alpha=.05)
# lasso.fit(X, Y)
# ranks["Lasso"] = ranking(np.abs(lasso.coef_), colnames)


# # # 5. Random Forest feature ranking
# # 
# # Sklearn's Random Forest model also comes with it's own inbuilt feature ranking attribute and one can conveniently just call it via "feature_importances_". That is what we will be using as follows:

# # In[ ]:


# rf = RandomForestRegressor(n_jobs=-1, n_estimators=50, verbose=3)
# rf.fit(X,Y)
# ranks["RF"] = ranking(rf.feature_importances_, colnames);


# # # 6. Creating the Feature Ranking Matrix
# # 
# # We combine the scores from the various methods above and output it in a matrix form for convenient viewing as such:

# # In[ ]:


# # Create empty dictionary to store the mean value calculated from all the scores
# r = {}
# for name in colnames:
#     r[name] = round(np.mean([ranks[method][name] 
#                              for method in ranks.keys()]), 2)
 
# methods = sorted(ranks.keys())
# ranks["Mean"] = r
# methods.append("Mean")
 
# print("\t%s" % "\t".join(methods))
# for name in colnames:
#     print("%s\t%s" % (name, "\t".join(map(str, 
#                          [ranks[method][name] for method in methods]))))


# # Now, with the matrix above, the numbers and layout does not seem very easy or pleasant to the eye. Therefore, let's just collate the mean ranking score attributed to each of the feature and plot that via Seaborn's factorplot.

# # In[ ]:


# # Put the mean scores into a Pandas dataframe
# meanplot = pd.DataFrame(list(r.items()), columns= ['Feature','Mean Ranking'])

# # Sort the dataframe
# meanplot = meanplot.sort_values('Mean Ranking', ascending=False)


# # In[ ]:


# # Let's plot the ranking of the features
# sns.factorplot(x="Mean Ranking", y="Feature", data = meanplot, kind="bar", 
#                size=14, aspect=1.9, palette='coolwarm')


# # Well as you can see from our feature ranking endeavours, the top 3 features are 'lat', 'waterfront' and 'grade'. The bottom 3 are 'sqft_lot15', 'sqft_lot' and 'sqft_basement'. 
# # This sort of feature ranking can be really useful, especially if one has many many features in the dataset and would like to trim or cut off features that contribute negligibly.

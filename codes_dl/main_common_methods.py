
# coding: utf-8

# In[10]:


# This kernel is based on Alexander Kireev's deep learning model:
#   https://www.kaggle.com/alexanderkireev/deep-learning-support-imbalance-architect-9671
# (See notes and references below.)
# I (Andy Harless) have made the following changes:
#   1. Add 2 more (narrower) layers on top
#   2. Eliminate "day" and "wday" variables (no variation in this sample)
#   3. Change target weight from 99 to 70
#   4. Add batch normalization
#   5. Only one epoch
#   6. Eliminate weight decay
#   7. Increase batch size
#   8. Increase dropout

# version 4:  adding ipcount


# good day, my friends
# in this kernel we try to continue development of our DL models
# =================================================================================================
# we continue our work
# this kernel is attempt to configure neural network for work with imbalanced data (see ~150th row)
# =================================================================================================
# thanks for people who share his works. i hope together we can create smth interest

# https://www.kaggle.com/baomengjiao/embedding-with-neural-network
# https://www.kaggle.com/gpradk/keras-starter-nn-with-embeddings
# https://www.kaggle.com/pranav84/lightgbm-fixing-unbalanced-data-auc-0-9787
# https://www.kaggle.com/rteja1113/lightgbm-with-count-features
# https://www.kaggle.com/knowledgegrappler/a-simple-nn-solution-with-keras-0-48611-pl
# https://www.kaggle.com/isaienkov/rnn-with-keras-ridge-sgdr-0-43553
# https://www.kaggle.com/valkling/mercari-rnn-2ridge-models-with-notes-0-42755/versions#base=2202774&new=2519287

print ('Good luck')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import gc
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

# import libs
import time
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

import numpy as np 
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')

from sklearn import preprocessing


# importing all the required ML packages
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.neural_network import MLPClassifier #Neural Network
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix



path = '../input/'
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }


print('load train....')
# we save only day 9


train_set_name = 'train_0'
train_set_name2 = 'train_1'
train_set_name3 = 'train_2'


print (train_set_name)
print (train_set_name2)
print (train_set_name3)

        
print ("load train...")
train_df_full = pd.read_pickle(train_set_name)
train_df_full2 = pd.read_pickle(train_set_name2)
train_df_full3 = pd.read_pickle(train_set_name3)
train_df_full=train_df_full.append(train_df_full2)
del train_df_full2
train_df_full=train_df_full.append(train_df_full3)
del train_df_full3
gc.collect()


# train_set_name = 'train_0'
# train_df_full = pd.read_pickle(train_set_name)

print ("split train...")
train_df_full, train_df_del = train_test_split(train_df_full, test_size=0.8)
del train_df_del
gc.collect()

print(train_df_full.head())

train,test=train_test_split(train_df_full,test_size=0.3,random_state=0,stratify=train_df_full['is_attributed'])
del train_df_full
gc.collect()

train_X=train.drop(['is_attributed'], axis=1)
train_Y=train[['is_attributed']]
test_X=test.drop(['is_attributed'], axis=1)
test_Y=test[['is_attributed']]
# X=train_df_full.drop(['is_attributed'], axis=1)
# Y=train_df_full['is_attributed']



print(train_X.head())
print(test_X.head())

print ("load test...")
test_df = pd.read_pickle('test')
# print(test_df.head())
print("test size : ", len(test_df))
sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')
print(test_df.head())


print("Logistic Regression...")
start = time.time()
model = LogisticRegression()
model.fit(train_X,train_Y)
prediction4=model.predict(test_X)
end = time.time()
print ("Bagging SVC", end - start)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction4,test_Y))


save_name = 'lightgbm_lr'
predictors = ['app','device','os', 'channel', 'min', 'hour', 'ipcount', 'qty', 'ip_app_count', 'ip_app_os_count']
print("Predicting...")
sub['is_attributed'] = model.predict(test_df[predictors])
print("writing...")
sub.to_csv(save_name + '.csv',index=False)
print("done...")
# In[16]:


# print(train_X.head())
# print(train_Y.head())
# print(test_X.head())
# print(test_Y.head())


# ### SVM radial


# n_estimators = 10
# print("Bagging SVC...")
# start = time.time()
# clf = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='auto'), max_samples=1.0 / n_estimators, n_estimators=n_estimators))
# clf.fit(train_X, train_Y)
# end = time.time()
# print ("Bagging SVC", end - start, clf.score(train_X,train_Y))


# print("Random Forest...")
# start = time.time()
# clf = RandomForestClassifier(min_samples_leaf=20)
# clf.fit(train_X, train_Y)
# end = time.time()
# print ("Random Forest", end - start, clf.score(train_X,train_Y))

# In[ ]:

# print("running svm rbf...")
# model = svm.SVC(kernel='rbf', C=1, gamma=0.1)
# model.fit(train_X, train_Y) 
# prediction1=model.predict(test_X)
# print('Accuracy for rbf SVM is ',metrics.accuracy_score(prediction1,test_Y))


# # In[ ]:

# print("running svm linear...")
# model=svm.SVC(kernel='linear',C=0.1,gamma=0.1)
# model.fit(train_X,train_Y)
# prediction2=model.predict(test_X)
# print('Accuracy for linear SVM is',metrics.accuracy_score(prediction2,test_Y))


# # In[ ]:

# print("running decision tree...")
# model=DecisionTreeClassifier()
# model.fit(train_X,train_Y)
# prediction5=model.predict(test_X)
# print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction5,test_Y))


# # In[ ]:

# print("running knn...")
# model=KNeighborsClassifier() 
# model.fit(train_X,train_Y)
# prediction6=model.predict(test_X)
# print('The accuracy of the KNN is',metrics.accuracy_score(prediction6,test_Y))


# # In[ ]:

# print("running random forest...")
# model=RandomForestClassifier(n_estimators=100)
# model.fit(train_X,train_Y)
# prediction8=model.predict(test_X)
# print('The accuracy of the Random Forests is',metrics.accuracy_score(prediction8,test_Y))














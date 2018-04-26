# maiking train and val csv's

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime


import os
print(os.listdir("../input"))

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SKIP = range(1,9308569) #to skip day 6

CATEGORICAL = ['app', 'device', 'os', 'channel']

TRAIN_COLUMNS = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']

DTYPES = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        }

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32',
        }        

###################HELPER FUNCTIONS#####################################################
def timer(start_time=None):
    """Prints time
    
    Initiate a time object, and prints total time consumed when again initialized object is passed as argument
    
    Keyword Arguments:
        start_time {[object]} -- initialized time object (default: {None})
    
    """
    
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

def dataPreProcessTime(df):
    # Make some new features with click_time column
    df['date_time'] = pd.to_datetime(df['click_time'])
    df['day']      = df['date_time'].dt.dayofweek.astype('uint8')
    df['hour']      = df['date_time'].dt.hour.astype('uint8')
    # df['min'] = df['click_time'].dt.minute.astype('uint8')
    # df['sec'] = df['click_time'].dt.second.astype('uint8')
    df.drop(['date_time'], axis=1, inplace=True)
    return df
    
    

# Any results you write to the current directory are saved as output.

t = timer(None)
print("Reading train file and extracting day and hour...")
train = pd.read_csv(TRAIN_PATH,skiprows=SKIP,header=0,usecols=TRAIN_COLUMNS,dtype=dtypes)
test = pd.read_csv(TEST_PATH,header=0,usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'],dtype=dtypes)
print('train len:', len(train))
print("Parsing day, hour from date and making new features")
train = dataPreProcessTime(train)
print(train.day.unique())
timer(t)

#seperate train and val set
X_train = train.loc[train.day <= 2] #day 7,8
print('day 7,8 len:', len(X_train))
X_val = train.loc[train.day == 3] #day 9
print('day 9 len:', len(X_val))
print(X_train.day.unique())


# del train

X_train.drop(['day', 'hour'], axis=1, inplace=True)
print(X_train.head())

# # print("Val set")
# # print(X_val.head())

print("Saving train and valid...")
X_train.to_csv("../input/train_day_7_8.csv.gz", index=False, compression='gzip')
X_val.to_csv("../input/valid_day_9.csv.gz", index=False, compression='gzip')
test.to_csv("../input/test.csv.gz", index=False, compression='gzip')

print("All Done")
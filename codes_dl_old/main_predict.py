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

from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, SpatialDropout1D, GaussianDropout
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam

from keras.models import load_model

print('load model...')
model = load_model('imbalanced_data.h5')

print('load test...')
train_set_name = 'test'
test_df = pd.read_pickle(train_set_name)
print(test_df.head())

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')

def get_keras_data(dataset):
    X = {
        'app': np.array(dataset.app),
        'ch': np.array(dataset.channel),
        'dev': np.array(dataset.device),
        'os': np.array(dataset.os),
        'h': np.array(dataset.hour),
        'qty': np.array(dataset.qty),
        'ipcount': np.array(dataset.ipcount),
        'c1': np.array(dataset.ip_app_count),
        'c2': np.array(dataset.ip_app_os_count)
    }
    return X

test_df = get_keras_data(test_df)



batch_size = 8192
print("predicting....")
sub['is_attributed'] = model.predict(test_df, batch_size=batch_size, verbose=2)
del test_df; gc.collect()
print(sub.head())
print("writing....")
sub.to_csv('tddlakah1.csv', index=False, float_format='%.3f')
print("done...")
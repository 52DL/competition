"""
  __file__
    preprocess.py
  __description__
    this file make the preprocess of the original data:
    1. chage the "-1" to NAN
    2. make the category type data to one-hot style
  __author__
    qufu
"""

import numpy as np
import pandas as pd
import sys
from multiprocessing import *
sys.path.append("../")
from param_config import config

###################
### Loading Data ##
###################
print("loading data...")

dfTrain = pd.read_csv(config.original_train_data_path)
dfTest = pd.read_csv(config.original_test_data_path)
num_train, num_test = dfTrain.shape[0], dfTest.shape[0]
col = [c for c in dfTrain.columns if c not in ['id','target']]
print("%d features " % len(col))
col = [c for c in col if not c.startswith('ps_calc_')]
print("%d feature beside ps_calc_" % len(col))
print("%d train data, %d test data" % (num_train, num_test))
print("loda done")

######################
### Preprocess data ##
######################
train = dfTrain.replace(-1, np.NaN)
test = dfTest.replace(-1, np.NaN)

train.to_csv(config.processed_train_data_path,index=False)
test.to_csv(config.processed_test_data_path,index=False)

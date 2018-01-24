import os
import time
import numpy as np
import pandas as pd

import sys
sys.path.append("../")
from param_config import config

#Data loading & preprocessing
df_train = pd.read_csv('../../../input/train.csv')
df_test = pd.read_csv('../../../input/test.csv')

X_train, y_train = df_train.iloc[:,2:], df_train.target
X_test = df_test.iloc[:,1:]

cols_use = [c for c in X_train.columns if (not c.startswith('ps_calc_'))]

X_train = X_train[cols_use]
X_test = X_test[cols_use]

# get the1ow feat
feat_path_name = 'the1ow-1031'
the1ow_path = "%s/%s/All" % (config.feat_folder, feat_path_name)

the1ow_train = pd.read_csv("%s/train.feat.csv" % the1ow_path)
X_train['ps_car_13_x_ps_reg_03'] = the1ow_train['ps_car_13_x_ps_reg_03']
X_train['negative_one_vals'] = the1ow_train['negative_one_vals']
del the1ow_train

the1ow_test = pd.read_csv("%s/test.feat.csv" % the1ow_path)
X_test['ps_car_13_x_ps_reg_03'] = the1ow_test['ps_car_13_x_ps_reg_03']
X_test['negative_one_vals'] = the1ow_test['negative_one_vals']
del the1ow_test

# get andy feat
feat_path_name = 'andy-1117'
andy_path = "%s/%s/All" % (config.feat_folder, feat_path_name)

andy_train = pd.read_csv("%s/train.feat.csv" % andy_path)
X_train['ps_reg_01_plus_ps_car_02_cat'] = andy_train['ps_reg_01_plus_ps_car_02_cat']
X_train['ps_reg_01_plus_ps_car_04_cat'] = andy_train['ps_reg_01_plus_ps_car_04_cat']
del andy_train

andy_test = pd.read_csv("%s/test.feat.csv" % andy_path)
X_test['ps_reg_01_plus_ps_car_02_cat'] = andy_test['ps_reg_01_plus_ps_car_02_cat']
X_test['ps_reg_01_plus_ps_car_04_cat'] = andy_test['ps_reg_01_plus_ps_car_04_cat']
del andy_test

#col_vals_dict = {c: list(X_train[c].unique()) for c in X_train.columns if c.endswith('_cat')}
col_vals_dict = {c: list(X_train[c].unique()) for c in X_train.columns}

embed_cols = []
for c in col_vals_dict:
    if len(col_vals_dict[c])>2:
        embed_cols.append(c)
        print(c + ': %d values' % len(col_vals_dict[c])) #look at value counts to know the embedding dimensions


x1 = X_train
x2 = X_test
y1 = df_train[['id','target']]
y2 = df_test[['id']]
print(x1.shape)
print(x2.shape)
feat_path_name = 'eddy-1126'
save_path = "%s/%s/All" % (config.feat_folder, feat_path_name)
if not os.path.exists(save_path):
        os.makedirs(save_path)

x1.to_csv("%s/train.feat.csv" % save_path, index=False, header=True)
x2.to_csv("%s/test.feat.csv" % save_path, index=False, header=True)
y1.to_csv("%s/train.target.csv" % save_path, index=False, header=True)
y2.to_csv("%s/test.target.csv" % save_path, index=False, header=True)

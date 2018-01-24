import os

import sys
sys.path.append("../")
from param_config import config
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from sklearn import *
import xgboost as xgb
import lightgbm as lgb
from multiprocessing import *

highcardinality = ['ps_car_02_cat',
                       'ps_car_09_cat',
                       'ps_ind_04_cat',
                       'ps_ind_05_cat',
                       'ps_car_03_cat',
                       'ps_ind_08_bin',
                       'ps_car_05_cat',
                       'ps_car_08_cat',
                       'ps_ind_06_bin',
                       'ps_ind_07_bin',
                       'ps_ind_12_bin',
                       'ps_ind_18_bin',
                       'ps_ind_17_bin',
                       'ps_car_07_cat',
                       'ps_car_11_cat',
                       'ps_ind_09_bin',
                       'ps_car_10_cat',
                       'ps_car_04_cat',
                       'ps_car_01_cat',
                       'ps_ind_02_cat',
                       'ps_ind_10_bin',
                       'ps_ind_11_bin',
                       'ps_car_06_cat',
                       'ps_ind_13_bin',
                       'ps_ind_16_bin']


def ProjectOnMean(data1, data2, columnName):
    grpOutcomes = data1.groupby(list([columnName]))['target'].mean().reset_index()
    grpCount = data1.groupby(list([columnName]))['target'].count().reset_index()
    grpOutcomes['cnt'] = grpCount.target
    grpOutcomes.drop('cnt', inplace=True, axis=1)
    outcomes = data2['target'].values
    x = pd.merge(data2[[columnName, 'target']], grpOutcomes,
                 suffixes=('x_', ''),
                 how='left',
                 on=list([columnName]),
                 left_index=True)['target']


    return x.values

train = pd.read_csv('../../data/train.csv')
test = pd.read_csv('../../data//test.csv')
y1 = train[['id','target']]
y2 = test[['id']]

train['missing'] = (train==-1).sum(axis=1).astype(float)
test['missing'] = (test==-1).sum(axis=1).astype(float)

unwanted = train.columns[train.columns.str.startswith('ps_calc_')]
train.drop(unwanted,inplace=True,axis=1)
test.drop(unwanted,inplace=True,axis=1)

test['target'] = np.nan
feats = list(set(train.columns).difference(set(['id','target'])))
feats = list(['id'])+feats +list(['target'])
train = train[feats]
test = test[feats]

blindloodata = None
folds = 5
kf = KFold(n_splits=folds,shuffle=True,random_state=1)
for i, (train_index, test_index) in enumerate(kf.split(range(train.shape[0]))):
    print('Fold:',i)
    blindtrain = train.loc[test_index].copy()
    vistrain = train.loc[train_index].copy()

    for c in highcardinality:
        blindtrain.insert(1,'loo_'+c, ProjectOnMean(vistrain,
                           blindtrain,c))
    if(blindloodata is None):
        blindloodata = blindtrain.copy()
    else:
        blindloodata = pd.concat([blindloodata,blindtrain])

for c in highcardinality:
    test.insert(1,'loo_'+c, ProjectOnMean(train,
                     test,c))
test.drop(highcardinality,inplace=True,axis=1)

train = blindloodata
train.drop(highcardinality,inplace=True,axis=1)
train = train.fillna(train.mean())
test = test.fillna(train.mean())

print('Scale values')
ss = StandardScaler()
features = train.columns[1:-1]
ss.fit(pd.concat([train[features],test[features]]))
train[features] = ss.transform(train[features] )
test[features] = ss.transform(test[features] )
train[features] = np.round(train[features], 6)
test[features] = np.round(test[features], 6)




x1 = train
x2 = test


feat_path_name = 'scirp-1127'
save_path = "%s/%s/All" % (config.feat_folder, feat_path_name)
if not os.path.exists(save_path): 
    os.makedirs(save_path)

x1.to_csv("%s/train.feat.csv" % save_path, index=False, header=True)
x2.to_csv("%s/test.feat.csv" % save_path, index=False, header=True)
y1.to_csv("%s/train.target.csv" % save_path, index=False, header=True)
y2.to_csv("%s/test.target.csv" % save_path, index=False, header=True)

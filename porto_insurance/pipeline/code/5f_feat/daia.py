import os

import sys
sys.path.append("../")
from param_config import config

import numpy as np
import pandas as pd
from sklearn import *
import xgboost as xgb
import lightgbm as lgb
from multiprocessing import *

train = pd.read_csv('../../data/train.csv')
test = pd.read_csv('../../data//test.csv')
##############################################creatinng   kinetic features for  train #####################################################
def  kinetic(row):
    probs=np.unique(row,return_counts=True)[1]/len(row)
    kinetic=np.sum(probs**2)
    return kinetic
    

first_kin_names=[col for  col in train.columns  if '_ind_' in col]
subset_ind=train[first_kin_names]
kinetic_1=[]
for row in range(subset_ind.shape[0]):
    row=subset_ind.iloc[row]
    k=kinetic(row)
    kinetic_1.append(k)
second_kin_names= [col for  col in train.columns  if '_car_' in col and col.endswith('cat')]
subset_ind=train[second_kin_names]
kinetic_2=[]
for row in range(subset_ind.shape[0]):
    row=subset_ind.iloc[row]
    k=kinetic(row)
    kinetic_2.append(k)
third_kin_names= [col for  col in train.columns  if '_calc_' in col and  not col.endswith('bin')]
subset_ind=train[second_kin_names]
kinetic_3=[]
for row in range(subset_ind.shape[0]):
    row=subset_ind.iloc[row]
    k=kinetic(row)
    kinetic_3.append(k)
fd_kin_names= [col for  col in train.columns  if '_calc_' in col and  col.endswith('bin')]
subset_ind=train[fd_kin_names]
kinetic_4=[]
for row in range(subset_ind.shape[0]):
    row=subset_ind.iloc[row]
    k=kinetic(row)
    kinetic_4.append(k)
train['kinetic_1']=np.array(kinetic_1)
train['kinetic_2']=np.array(kinetic_2)
train['kinetic_3']=np.array(kinetic_3)
train['kinetic_4']=np.array(kinetic_4)

############################################reatinng   kinetic features for  test###############################################################

first_kin_names=[col for  col in test.columns  if '_ind_' in col]
subset_ind=test[first_kin_names]
kinetic_1=[]
for row in range(subset_ind.shape[0]):
    row=subset_ind.iloc[row]
    k=kinetic(row)
    kinetic_1.append(k)
second_kin_names= [col for  col in test.columns  if '_car_' in col and col.endswith('cat')]
subset_ind=test[second_kin_names]
kinetic_2=[]
for row in range(subset_ind.shape[0]):
    row=subset_ind.iloc[row]
    k=kinetic(row)
    kinetic_2.append(k)
third_kin_names= [col for  col in test.columns  if '_calc_' in col and  not col.endswith('bin')]
subset_ind=test[second_kin_names]
kinetic_3=[]
for row in range(subset_ind.shape[0]):
    row=subset_ind.iloc[row]
    k=kinetic(row)
    kinetic_3.append(k)
fd_kin_names= [col for  col in test.columns  if '_calc_' in col and  col.endswith('bin')]
subset_ind=test[fd_kin_names]
kinetic_4=[]
for row in range(subset_ind.shape[0]):
    row=subset_ind.iloc[row]
    k=kinetic(row)
    kinetic_4.append(k)
test['kinetic_1']=np.array(kinetic_1)
test['kinetic_2']=np.array(kinetic_2)
test['kinetic_3']=np.array(kinetic_3)
test['kinetic_4']=np.array(kinetic_4)

train = train.replace(-1, np.NaN)
d_median = train.median(axis=0)
d_mean = train.mean(axis=0)
train = train.fillna(-1)
one_hot = {c: list(train[c].unique()) for c in train.columns if c not in ['id','target']}

def transform_df(df):
    df = pd.DataFrame(df)
    dcol = [c for c in df.columns if c not in ['id','target']]
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    df['negative_one_vals'] = np.sum((df[dcol]==-1).values, axis=1)
    for c in dcol:
        if '_bin' not in c: #standard arithmetic
            df[c+str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
            df[c+str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)
            #df[c+str('_sq')] = np.power(df[c].values,2).astype(np.float32)
            #df[c+str('_sqr')] = np.square(df[c].values).astype(np.float32)
            #df[c+str('_log')] = np.log(np.abs(df[c].values) + 1)
            #df[c+str('_exp')] = np.exp(df[c].values) - 1
    return df

def multi_transform(df):
    print('Init Shape: ', df.shape)
    p = Pool(cpu_count())
    df = p.map(transform_df, np.array_split(df, cpu_count()))
    df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
    p.close(); p.join()
    print('After Shape: ', df.shape)
    return df

def gini(y, pred):
    fpr, tpr, thr = metrics.roc_curve(y, pred, pos_label=1)
    g = 2 * metrics.auc(fpr, tpr) -1
    return g

def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', gini(y, pred)


x1 = multi_transform(train)
x2 = multi_transform(test)

col = [c for c in x1.columns if c not in ['id','target']]
col = [c for c in col if not c.startswith('ps_calc_')]
print(x1.values.shape, x2.values.shape)

y1 = x1[['id','target']]
y2 = x2[['id']]
x1 = x1[col]
x2 = x2[col]

feat_path_name = 'daia-1127'
save_path = "%s/%s/All" % (config.feat_folder, feat_path_name)
if not os.path.exists(save_path): 
    os.makedirs(save_path)

x1.to_csv("%s/train.feat.csv" % save_path, index=False, header=True)
x2.to_csv("%s/test.feat.csv" % save_path, index=False, header=True)
y1.to_csv("%s/train.target.csv" % save_path, index=False, header=True)
y2.to_csv("%s/test.target.csv" % save_path, index=False, header=True)

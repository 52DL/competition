# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from sklearn.metrics import log_loss,accuracy_score
# All submission files were downloaded from different public kernels
# See the description to see the source of each submission file
submissions_path = "../subm/ensemble/"
feat_names = []
feat_names.append('xgb9')
feat_names.append('lr9')
feat_names.append('svm9')
feat_names.append('knn9')
train_org = pd.read_json('../input/train.json')
id_train = train_org['id']
test_org = pd.read_json('../input/test.json' )
id_test = test_org['id']

y = train_org['is_iceberg'].values

valids = [pd.read_csv(submissions_path+f+'_valid.csv') for f in feat_names]
valid = pd.merge(valids[0],valids[1] , how='left',on='id')
for i in range(2,len(feat_names)):
    valid = pd.merge(valid,valids[i],how='left',on='id')
cols = ['id']+feat_names
valid.columns = cols
train=valid
#valid['is_iceberg'] = y
tests = [pd.read_csv(submissions_path+f+'_submit.csv') for f in feat_names]
test = pd.merge(tests[0],tests[1] , how='left',on='id')
for i in range(2,len(feat_names)):
    test = pd.merge(test,tests[i],how='left',on='id')
cols = ['id']+feat_names
test.columns = cols
print(train.shape)
print(test.shape)
train['mean'] = train.iloc[:,1:].mean(axis=1)
test['mean'] = test.iloc[:,1:].mean(axis=1)
y_valid_pred = train['mean']
y_test_pred = test['mean']
print('loss  &  accuracy: ',log_loss(y, y_valid_pred),accuracy_score(y,np.round(y_valid_pred)))
# Create submission file
val = pd.DataFrame()
val['id'] = id_train
val['is_iceberg'] = y_valid_pred
val.to_csv('../subm/ensemble/avarage_valid.csv', float_format='%.6f', index=False)

# Create submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['is_iceberg'] = y_test_pred
sub.to_csv('../subm/ensemble/avarage_submit.csv', float_format='%.6f', index=False)


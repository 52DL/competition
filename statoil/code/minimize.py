# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from sklearn.metrics import log_loss,accuracy_score
from scipy.optimize import minimize
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
col = [c for c in train.columns if c not in ['id']]
#print(col)
train = train.loc[:,col]
#print(X.info())
print(train.shape)
test = test.loc[:,col]
print(test.shape)
def mae_func(weights,Y_values,predictions):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction
    #print(final_prediction.shape)
    #print(prediction.shape)
    #print(Y_values.shape)
    return log_loss(Y_values, final_prediction)

def get_mini(df,bestWght):
    predictions=[]
    #print(Y_values.shape)
    lls = []
    wghts = []

    for i in range(4):
        predictions.append(np.array(df.iloc[:,i]))

    if bestWght==[]:
        Y_values = y
        for i in range(100):
            starting_values = np.random.uniform(size=4)
            cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
            bounds = [(0,1)]*len(predictions)

            res = minimize(fun=mae_func, x0=starting_values,args=(Y_values, predictions), method='L-BFGS-B', bounds=bounds, options={'disp': False, 'maxiter': 100000})

            lls.append(res['fun'])
            wghts.append(res['x'])
        # Uncomment the next line if you want to see the weights and scores calculated in real time
        #    print('Weights: {weights}  Score: {score}'.format(weights=res['x'], score=res['fun']))

        bestSC = np.min(lls)
        bestWght = wghts[np.argmin(lls)]
    #print(bestSC)
    print(bestWght)
    a = 0.0*predictions[0]
    for weight, prediction in zip(bestWght, predictions):
        a += weight*prediction
    return a,bestWght

train['mean'],bestWght = get_mini(train,[])
test['mean'],_ = get_mini(test,bestWght)
y_valid_pred = train['mean']
y_test_pred = test['mean']
print('loss  &  accuracy: ',log_loss(y, y_valid_pred),accuracy_score(y,np.round(y_valid_pred)))
# Create submission file
val = pd.DataFrame()
val['id'] = id_train
val['is_iceberg'] = y_valid_pred
val.to_csv('../subm/ensemble/minimize_valid.csv', float_format='%.6f', index=False)

# Create submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['is_iceberg'] = y_test_pred
sub.to_csv('../subm/ensemble/minimize_submit.csv', float_format='%.6f', index=False)


MAX_ROUNDS = 710
OPTIMIZE_XGB_ROUNDS = False
MAX_DEPTH = 6
LEARNING_RATE = 0.024
XGB_EARLY_STOPPING_ROUNDS = 100
FEAT_FOLDER = '../../feat/solution/qufu-1129'


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from numba import jit
import time
import gc
import subprocess
import glob
from utils_csv import *

# Compute gini

# from CPMP's kernel https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
@jit
def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini

# Funcitons from olivier's kernel
# https://www.kaggle.com/ogrellier/xgb-classifier-upsampling-lb-0-283

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = -eval_gini(labels, preds)
    return [('gini', gini_score)]

def gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = eval_gini(y,preds) / eval_gini(y,y)
    return 'gini', score, True


start = time.clock()
tt_data = get_feat_data(FEAT_FOLDER)
print("loading data used time: ", (time.clock()-start))

X = tt_data['X1']
test_df = tt_data['X2']
y = tt_data['Y1']['target']
id_train = tt_data['Y1']['id'].values
id_test = tt_data['Y2']['id'].values


f_cats = [f for f in X.columns if "_cat" in f]

y_valid_pred = 0*y
y_test_pred = 0

# Set up folds
K = 5
kf = KFold(n_splits = K, random_state = 1, shuffle = True)
np.random.seed(0)

# Set up classifier
lgb = LogisticRegression(class_weight='balanced')
# Run CV

for i, (train_index, test_index) in enumerate(kf.split(X)):
    
    # Create data for this fold
    y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index]
    X_train, X_valid = X.iloc[train_index,:].copy(), X.iloc[test_index,:].copy()
    X_test = test_df.copy()
    print( "\nFold ", i)
    
    # Run model for this fold
    if OPTIMIZE_XGB_ROUNDS:
        fit_model = lgb.train( 
                             params,
                             lgb.Dataset(X_train, label=y_train), 
                             MAX_ROUNDS,
                             lgb.Dataset(X_valid, label=y_valid),
                             verbose_eval=50,
                             feval=gini_lgb,
                             early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
                             )
        print( "  Best N trees = ", fit_model.best_iteration )
        pred = fit_model.predict(X_valid, num_iteration=fit_model.best_iteration)
        probs = fit_model.predict(X_test, num_iteration=fit_model.best_iteration)
    else:
        fit_model = lgb.fit(
                               X_train,y_train 
                             )
        pred = fit_model.predict(X_valid)
        probs = fit_model.predict(X_test)

    # Save validation predictions for this fold
    print( "  Gini = ", eval_gini(y_valid, pred) )
    #y_valid_pred.iloc[test_index] = (np.exp(pred) - 1.0).clip(0,1)
    y_valid_pred.iloc[test_index] = pred
    
    # Accumulate test set predictions
    #y_test_pred += (np.exp(test_pred) - 1.0).clip(0,1)
    almost_zero = 1e-12
    almost_one = 1 - almost_zero  # To avoid division by zero
    probs[probs>almost_one] = almost_one
    probs[probs<almost_zero] = almost_zero
    y_test_pred += np.log(probs/(1-probs))
    
    del X_test, X_train, X_valid, y_train
    
y_test_pred /= K  # Average test set predictions
y_test_pred =  1  /  ( 1 + np.exp(-y_test_pred) )

print( "\nGini for full training set:" ,eval_gini(y, y_valid_pred))
"""
# Save validation predictions for stacking/ensembling
val = pd.DataFrame()
val['id'] = id_train
val['target'] = y_valid_pred.values
val.to_csv('%s/the1ow_lgb_valid.csv' % config.sub_folder, float_format='%.6f', index=False)

# Create submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_test_pred
sub.to_csv('%s/the1ow_lgb_submit.csv' % config.sub_folder, float_format='%.6f', index=False)

"""

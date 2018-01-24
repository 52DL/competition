OPTIMIZE_XGB_ROUNDS = True
MAX_ROUNDS = 2400
FEAT_FOLDER = '../../feat/solution/the1ow-1031'


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from numba import jit
import time
import gc
import subprocess
import glob
from utils_csv import *

import tensorflow as tf
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

def gini_tf(preds, y):
    score = eval_gini(y,preds) / eval_gini(y,y)
    return score


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

cols = tf.contrib.layers.real_valued_column('x', dimension=X.shape[1])
tfm = tf.contrib.learn.DNNClassifier(
        feature_columns=[cols],
        hidden_units=[16,16,16],
        n_classes=2,
        optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=0.1,
            l1_regularization_strength=0.001)
        )
tfm = tf.contrib.learn.SKCompat(tfm)

# Run CV

for i, (train_index, test_index) in enumerate(kf.split(X)):
    
    # Create data for this fold
    y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index]
    X_train, X_valid = X.iloc[train_index,:].copy(), X.iloc[test_index,:].copy()
    X_test = test_df.copy()
    print( "\nFold ", i)
    
    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
            x={'x':X_valid.values},
            y=y_valid.values,
            every_n_steps=50,
            early_stopping_metric='auc',
            early_stopping_metric_minimize=False,
            early_stopping_rounds=200)

    # Run model for this fold
    if OPTIMIZE_XGB_ROUNDS:
        fit_model = tfm.fit( 
                             x={'x':X_train.values},
                             y=y_train.values, 
                             steps=MAX_ROUNDS,
                             monitors=[validation_monitor], 
                             )
        print( "  Best N trees = " )
        pred = fit_model.predict_proba({'x':X_valid.values})
        probs = fit_model.predict_proba({'x':X_test.values})
    else:
        fit_model = tfm.fit( 
                             x={'x':X_train.values},
                             y=y_train.values, 
                             steps=MAX_ROUNDS,
                             )
        pred = fit_model.predict_proba({'x':X_valid.values})
        probs = fit_model.predict_proba({'x':X_test.values})

    # Save validation predictions for this fold
    print( "  Gini = ", eval_gini(y_valid, pred) )
    print( "  Gini = ", gini_tf(y_valid, pred) )
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
print( "\nGini for full training set:" ,gini_tf(y, y_valid_pred))

# Save validation predictions for stacking/ensembling
val = pd.DataFrame()
val['id'] = id_train
val['target'] = y_valid_pred.values
val.to_csv('%s/the1ow_tfm_valid.csv' % config.sub_folder, float_format='%.6f', index=False)

# Create submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_test_pred
sub.to_csv('%s/the1ow_tfm_submit.csv' % config.sub_folder, float_format='%.6f', index=False)


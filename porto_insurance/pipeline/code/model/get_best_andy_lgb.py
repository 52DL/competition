MAX_ROUNDS = 630
OPTIMIZE_XGB_ROUNDS = False
MAX_DEPTH = 6
LEARNING_RATE = 0.024
XGB_EARLY_STOPPING_ROUNDS = 100
FEAT_FOLDER = '../../feat/solution/andy-1117'


import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from rgf.sklearn import RGFClassifier
from numba import jit
import time
import gc
import subprocess
import glob
from utils_csv import *

import lightgbm as lgb
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
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None,    # Revised to encode validation series
                  val_series=None,
                  tst_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_val_series = pd.merge(
        val_series.to_frame(val_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=val_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_val_series.index = val_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_val_series, noise_level), add_noise(ft_tst_series, noise_level)

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
params = {    
        'learning_rate': LEARNING_RATE,
        'max_depth': MAX_DEPTH,
        'num_leaves': 31,
        'min_data_in_leaf':20,
        'feature_fraction':0.6,
        'max_bin':63,
        'lambda_l1':12,
        'lambda_l2':0,
        'boosting': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'is_training_metric':False,
        'seed':99,
        'verbose':-1
        }

# Run CV

for i, (train_index, test_index) in enumerate(kf.split(X)):
    
    # Create data for this fold
    y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index]
    X_train, X_valid = X.iloc[train_index,:].copy(), X.iloc[test_index,:].copy()
    X_test = test_df.copy()
    print( "\nFold ", i)
    # Enocode data
    for f in f_cats:
        X_train[f + "_avg"], X_valid[f + "_avg"], X_test[f + "_avg"] = target_encode(
                                                        trn_series=X_train[f],
                                                        val_series=X_valid[f],
                                                        tst_series=X_test[f],
                                                        target=y_train,
                                                        min_samples_leaf=200,
                                                        smoothing=10,
                                                        noise_level=0
                                                        ) 
    # Run model for this fold
    if OPTIMIZE_XGB_ROUNDS:
        fit_model = lgb.train( 
                             params,
                             lgb.Dataset(X_train, label=y_train), 
                             MAX_ROUNDS,
                             lgb.Dataset(X_valid, label=y_valid),
                             verbose_eval=False,
                             feval=gini_lgb,
                             early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
                             )
        print( "  Best N trees = ", fit_model.best_iteration )
        pred = fit_model.predict(X_valid, num_iteration=fit_model.best_iteration)
        probs = fit_model.predict(X_test, num_iteration=fit_model.best_iteration)
    else:
        fit_model = lgb.train( 
                               params, 
                               lgb.Dataset(X_train, label=y_train), 
                               MAX_ROUNDS, 
                               verbose_eval=50 
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

# Save validation predictions for stacking/ensembling
val = pd.DataFrame()
val['id'] = id_train
val['target'] = y_valid_pred.values
val.to_csv('%s/andy_lgb_valid.csv' % config.sub_folder, float_format='%.6f', index=False)

# Create submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_test_pred
sub.to_csv('%s/andy_lgb_submit.csv' % config.sub_folder, float_format='%.6f', index=False)


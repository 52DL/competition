USE_RGF_INSTEAD =  True
OPTIMIZE_XGB_ROUNDS = False 
XGB_LEARNING_RATE = 0.07
XGB_EARLY_STOPPING_ROUNDS = 50
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
xgbmodel = XGBClassifier(    
                        n_estimators=MAX_XGB_ROUNDS,
                        max_depth=4,
                        objective="binary:logistic",
                        learning_rate=XGB_LEARNING_RATE, 
                        subsample=.8,
                        min_child_weight=6,
                        colsample_bytree=0.8,
                        scale_pos_weight=1.6,
                        gamma=10,
                        reg_alpha=8,
                        reg_lambda=1.3,
                     )
rgf = RGFClassifier(   # See https://www.kaggle.com/scirpus/regularized-greedy-forest#241285
                    max_leaf=1200,  # Parameters suggested by olivier in link above
                    algorithm="RGF",  
                    loss="Log",
                    l2=0.01,
                    sl2=0.01,
                    normalize=False,
                    min_samples_leaf=10,
                    n_iter=None,
                    opt_interval=100,
                    learning_rate=.5,
                    calc_prob="sigmoid",
                    n_jobs=-1,
                    memory_policy="generous",
                    verbose=0
                   )

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
    if USE_RGF_INSTEAD:
        X_train = X_train.fillna(X_train.mean())
        rgf.fit(X_train, y_train)
    elif OPTIMIZE_XGB_ROUNDS:
        eval_set=[(X_valid,y_valid)]
        fit_model = xgbmodel.fit( X_train, y_train, 
                               eval_set=eval_set,
                               eval_metric=gini_xgb,
                               early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
                               verbose=False
                             )
        print( "  Best N trees = ", xgbmodel.best_ntree_limit )
        print( "  Best gini = ", xgbmodel.best_score )
    else:
        fit_model = xgbmodel.fit( X_train, y_train )
        
    # Generate validation predictions for this fold
    if USE_RGF_INSTEAD:
        pred = rgf.predict_proba(X_valid.fillna(X_train.mean()))[:,1]
    else:
        pred = fit_model.predict_proba(X_valid)[:,1]
    print( "  Gini = ", eval_gini(y_valid, pred) )
    y_valid_pred.iloc[test_index] = pred
    
    # Accumulate test set predictions
    if USE_RGF_INSTEAD:
        probs = rgf.predict_proba(X_test.fillna(X_train.mean()))[:,1]
        try:
            subprocess.call('rm -rf /tmp/rgf/*', shell=True)
            print("Clean up is successfull")
            print(glob.glob("/tmp/rgf/*"))
        except Exception as e:
            print(str(e))
    else:
        probs = fit_model.predict_proba(X_test)[:,1]
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
val.to_csv('%s/andy_rgf_valid.csv' % (config.sub_folder), float_format='%.6f', index=False)

# Create submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_test_pred
sub.to_csv('%s/andy_rgf_submit.csv' % (config.sub_folder), float_format='%.6f', index=False)


USE_RGF_INSTEAD = False 
MAX_XGB_ROUNDS = 1200
OPTIMIZE_XGB_ROUNDS = False
XGB_LEARNING_RATE = 0.07
XGB_EARLY_STOPPING_ROUNDS = 150
FEAT_FOLDER = '../../feat/solution/daia-1127'


import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
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


start = time.clock()
tt_data = get_feat_data(FEAT_FOLDER)
print("loading data used time: ", (time.clock()-start))

X = tt_data['X1']
test_df = tt_data['X2']
y = tt_data['Y1']['target']
id_train = tt_data['Y1']['id'].values
id_test = tt_data['Y2']['id'].values

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
                        min_child_weight=.8,
                        colsample_bytree=.8,
                        scale_pos_weight=0.32,
                        gamma=3.2,
                        reg_alpha=.8,
                        reg_lambda=0.5,
                     )

# Run CV

for i, (train_index, test_index) in enumerate(kf.split(X)):
    
    # Create data for this fold
    y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index]
    X_train, X_valid = X.iloc[train_index,:].copy(), X.iloc[test_index,:].copy()
    X_test = test_df.copy()
    print( "\nFold ", i)
    
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
                               verbose=100
                             )
        print( "  Best N trees = ", xgbmodel.best_ntree_limit )
        #print( "  Best gini = ", xgbmodel.best_score )
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
val.to_csv('%s/daia_xgb_valid.csv' % config.sub_folder, float_format='%.6f', index=False)

# Create submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_test_pred
sub.to_csv('%s/daia_xgb_submit.csv' % config.sub_folder, float_format='%.6f', index=False)


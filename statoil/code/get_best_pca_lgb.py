MAX_XGB_ROUNDS = 2175
OPTIMIZE_XGB_ROUNDS = True
LEARNING_RATE = 0.03
XGB_EARLY_STOPPING_ROUNDS = 200


import numpy as np
import pandas as pd
#from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from numba import jit
import time
import random
import gc
import datetime as dt
#import for image processing
#evaluation
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import log_loss,accuracy_score

#'pca+lgb'
def test_id(a,b):
    for i,j in zip(a,b):
        if i!=j:
            print('asdf')
#Load data
train_org = pd.read_json('../input/train.json')
id_train = train_org['id']
test_org = pd.read_json('../input/test.json' )
id_test = test_org['id']

y = train_org['is_iceberg'].values
#del train, test

#picf_train = pd.read_csv('../feat/picf_train_opted.csv')
#picf_test = pd.read_csv('../feat/picf_test_opted.csv')
sbmr_train = pd.read_csv('../feat/sbmr_train.csv')
sbmr_test = pd.read_csv('../feat/sbmr_test.csv')
pca_train = pd.read_csv('../feat/pca_train.csv')
pca_test = pd.read_csv('../feat/pca_test.csv')
#hog_train = pd.read_csv('../feat/hog_train.csv')
#hog_test = pd.read_csv('../feat/hog_test.csv')
train = pd.merge(sbmr_train, pca_train, how='left',on='id')
#train = pd.merge(train, hog_train, how='left',on='id')
test_id(id_train, train['id'])

#train = pca_train
print(train.info())
print(train.shape)
#train['inc_angle'] = pca_train['inc_angle']
#train['inc_angle'] = train_org['inc_angle'].replace('na', -1).astype(float)

test = pd.merge(sbmr_test, pca_test,how='left',on='id')
#test = pd.merge(test, hog_test,how='left',on='id')
test_id(id_test, test['id'])
#test = pca_test
#test['inc_angle'] = pca_test['inc_angle']
#test['inc_angle'] = test_org['inc_angle'].replace('na', -1).astype(float)
col = [c for c in train.columns if c not in ['id']]
X = train[col].values
print(X.shape)
test_df = test[col].values
print(test_df.shape)

y_valid_pred = 0.0*y
y_train_pred = 0.0*y
y_test_pred = 0.0

# Set up folds
K = 5
kf = KFold(n_splits = K, random_state = 1108, shuffle = True)
RUNs = 1
for m in [5]:
    for n in [0]:
        print('m:', m)
        print('n:', n)
        # Set up classifier
        params = {
		'learning_rate': 0.03,
		'max_depth': 5,
		'num_leaves': 32,
		'min_data_in_leaf':10,
		'feature_fraction':.8,
		'bagging_fraction':.8,
		'bagging_freq':1,
		'max_bin':31,
		'lambda_l1':0,
		'lambda_l2':0,
		'boosting_type': 'gbdt',
		#'objective': 'multiclass',
		'objective': 'binary',
		'metric': 'binary_logloss',
		#'metric': 'multi_logloss',
                #'num_class': 2,
		'is_training_metric':True,
		'seed':1,
		'verbose':-1
		}
        #params = {'learning_rate': 0.02, 'max_depth': 7, 'boosting_type': 'gbdt', 'objective': 'multiclass', 'metric' : 'multi_logloss', 'is_training_metric': True, 'num_class': 2, 'seed': 1, 'verbose':-1}
        # Run CV

        for i, (train_index, test_index) in enumerate(kf.split(X)):
            
            # Create data for this fold
            y_train, y_valid = y[train_index], y[test_index]
            X_train, X_valid = X[train_index,:], X[test_index,:]
            X_test = test_df
            print( "Fold ", i)
            
            valid_pred = 0.0 * y_valid
            train_pred = 0.0 * y_train
            for j in range(RUNs):

                seed = 1000*j
                np.random.seed(seed)
                random.seed(seed)
                print( "Run ", j)
                # Run model for this fold
                if OPTIMIZE_XGB_ROUNDS:
                    #watchlist = [(lgb.Dataset(X_train, y_train), 'train'), (lgb.Dataset(X_valid, y_valid), 'valid')]
                    #watchlist = lgb.Dataset(X_valid, label=y_valid)
                    params['seed'] = seed
                    fit_model = lgb.train(params, lgb.Dataset(X_train, label=y_train), 2000, lgb.Dataset(X_valid, label=y_valid), 
                                          verbose_eval=False, 
                                          early_stopping_rounds=200)
                    #eval_set=[(X_valid,y_valid)]
                    #fit_model = xgbmodel.fit( X_train, y_train, 
                    #                       eval_set=eval_set,
                    #                       eval_metric='logloss',
                    #                       early_stopping_np.rounds=XGB_EARLY_STOPPING_ROUNDS,
                    #                       verbose=False
                    #                     )
                    #print( "  Best N trees = ", fit_model.best_iteration )
                    #print( "  Best gini = ", fit_model.best_score )
                #else:
                    #fit_model = xgbmodel.fit( X_train, y_train )
                    
                # Generate validation predictions for this fold
                pred = fit_model.predict(X_train, num_iteration=fit_model.best_iteration)#[:,1]
                #print( "  train Gini = ", log_loss(y_train, pred), accuracy_score(y_train, np.round(pred) ))
                train_pred += pred
                pred = fit_model.predict(X_valid, num_iteration=fit_model.best_iteration)#[:,1]
                #print( "  valid Gini = ", log_loss(y_valid, pred), accuracy_score(y_valid, np.round(pred) ))
                valid_pred += pred
                
                # Accumulate test set predictions
                probs = fit_model.predict(X_test, num_iteration=fit_model.best_iteration)#[:,1]
                y_test_pred += probs
            train_pred /= RUNs        
            y_train_pred[train_index] += train_pred
            print( "  fold train Gini = ", log_loss(y_train, train_pred), accuracy_score(y_train, np.round(train_pred) ))
            valid_pred /= RUNs        
            y_valid_pred[test_index] = valid_pred
            print( "  fold valid Gini = ", log_loss(y_valid, valid_pred), accuracy_score(y_valid, np.round(valid_pred) ))
            del X_test, X_train, X_valid, y_train
        y_train_pred /= K-1
        print( "  full Gini = ", log_loss(y, y_train_pred), accuracy_score(y, np.round(y_train_pred) ))
        print( "  full Gini = ", log_loss(y, y_valid_pred), accuracy_score(y, np.round(y_valid_pred) ))

        y_test_pred /= (K * RUNs) # Average test set predictions
        print( "\nGini for full training set:" ,log_loss(y, y_valid_pred))
        # Save validation predictions for stacking/ensembling
        val = pd.DataFrame()
        val['id'] = id_train
        val['is_iceberg'] = y_valid_pred
        val.to_csv('../subm/pca+lgb_valid.csv', float_format='%.6f', index=False)

        # Create submission file
        sub = pd.DataFrame()
        sub['id'] = id_test
        sub['is_iceberg'] = y_test_pred
        sub.to_csv('../subm/pca+lgb_submit.csv', float_format='%.6f', index=False)


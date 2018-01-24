MAX_XGB_ROUNDS = 2175
OPTIMIZE_XGB_ROUNDS = True
XGB_LEARNING_RATE = 0.03
XGB_EARLY_STOPPING_ROUNDS = 200


import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from numba import jit
import time
import gc
import datetime as dt
#import for image processing
#evaluation
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import log_loss,accuracy_score

#'hog+xgb'

#Load data
train_org = pd.read_json('../input/train.json')
id_train = train_org['id']
test_org = pd.read_json('../input/test.json' )
id_test = test_org['id']

y = train_org['is_iceberg'].values
#del train, test

hog_train = pd.read_csv('../feat/hog_train.csv')
hog_test = pd.read_csv('../feat/hog_test.csv')
sbmr_train = pd.read_csv('../feat/sbmr_train.csv')
sbmr_test = pd.read_csv('../feat/sbmr_test.csv')
#pca_train = pd.read_csv('../feat/pca_train.csv')
#pca_test = pd.read_csv('../feat/pca_test.csv')
train = pd.concat([hog_train, sbmr_train], axis=1)
#train = hog_train
#train['inc_angle'] = pca_train['inc_angle']
train['inc_angle'] = train_org['inc_angle'].replace('na', -1).astype(float)

test = pd.concat([hog_test, sbmr_test],axis=1)
#test = hog_test
#test['inc_angle'] = pca_test['inc_angle']
test['inc_angle'] = test_org['inc_angle'].replace('na', -1).astype(float)
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
kf = StratifiedKFold(n_splits = K, random_state = 1108, shuffle = True)
RUNs = 1
for m in [.2,.4,.8,1.6,3.2]:
    for n in [.2,.4,.8,1.6,3.2]:
        print('m:', m)
        print('n:', n)
        # Set up classifier
        params = {'eta': 0.02,
                'max_depth': 5, 
                'objective': 'binary:logistic', 
                'eval_metric': 'logloss', 
                #'num_class': 2, 
                'subsample': 1, 
                'colsample_bytree': .6, 
                'min_child_weight': .6, 
                'scale_pos_weight': 1,
                'gamma':.6,
                'reg_alpha':m,
                'reg_lambda':n,
                'seed': 9, 
                'silent': True
                }

        # Run CV

        for i, (train_index, test_index) in enumerate(kf.split(X,y)):
            
            # Create data for this fold
            y_train, y_valid = y[train_index], y[test_index]
            X_train, X_valid = X[train_index,:], X[test_index,:]
            X_test = test_df
            print( "Fold ", i)
            
            valid_pred = 0.0 * y_valid
            train_pred = 0.0 * y_train
            for j in range(RUNs):

                seed = 1108+j
                np.random.seed(seed)
                print( "Run ", j)
                # Run model for this fold
                if OPTIMIZE_XGB_ROUNDS:
                    watchlist = [(xgb.DMatrix(X_train, y_train), 'train'), (xgb.DMatrix(X_valid, y_valid), 'valid')]
                    params['seed'] = seed
                    fit_model = xgb.train(params, xgb.DMatrix(X_train, y_train), 2000,  watchlist, verbose_eval=False, early_stopping_rounds=300)
                    #eval_set=[(X_valid,y_valid)]
                    #fit_model = xgbmodel.fit( X_train, y_train, 
                    #                       eval_set=eval_set,
                    #                       eval_metric='logloss',
                    #                       early_stopping_np.rounds=XGB_EARLY_STOPPING_ROUNDS,
                    #                       verbose=False
                    #                     )
                    #print( "  Best N trees = ", fit_model.best_ntree_limit )
                    #print( "  Best gini = ", fit_model.best_score )
                #else:
                    #fit_model = xgbmodel.fit( X_train, y_train )
                    
                # Generate validation predictions for this fold
                pred = fit_model.predict(xgb.DMatrix(X_train, y_train), ntree_limit=fit_model.best_ntree_limit)
                print( "  Gini = ", log_loss(y_train, pred), accuracy_score(y_train, np.round(pred) ))
                train_pred += pred
                pred = fit_model.predict(xgb.DMatrix(X_valid, y_valid), ntree_limit=fit_model.best_ntree_limit)
                print( "  Gini = ", log_loss(y_valid, pred), accuracy_score(y_valid, np.round(pred) ))
                valid_pred += pred
                
                # Accumulate test set predictions
                probs = fit_model.predict(xgb.DMatrix(X_test), ntree_limit=fit_model.best_ntree_limit)
                y_test_pred += probs
            train_pred /= RUNs        
            y_train_pred[train_index] += train_pred
            print( "  fold Gini = ", log_loss(y_train, train_pred), accuracy_score(y_train, np.round(train_pred) ))
            valid_pred /= RUNs        
            y_valid_pred[test_index] = valid_pred
            print( "  fold Gini = ", log_loss(y_valid, valid_pred), accuracy_score(y_valid, np.round(valid_pred) ))
            del X_test, X_train, X_valid, y_train
        y_train_pred /= K-1
        print( "  full Gini = ", log_loss(y, y_train_pred), accuracy_score(y, np.round(y_train_pred) ))
        print( "  full Gini = ", log_loss(y, y_valid_pred), accuracy_score(y, np.round(y_valid_pred) ))
        '''
        y_test_pred /= (K * RUNs) # Average test set predictions

        print( "\nGini for full training set:" ,log_loss(y, y_valid_pred))
        # Save validation predictions for stacking/ensembling
        val = pd.DataFrame()
        val['id'] = id_train
        val['is_iceberg'] = y_valid_pred
        val.to_csv('../subm/hog+xgb_valid.csv', float_format='%.6f', index=False)

        # Create submission file
        sub = pd.DataFrame()
        sub['id'] = id_test
        sub['is_iceberg'] = y_test_pred
        sub.to_csv('../subm/hog+xgb_submit.csv', float_format='%.6f', index=False)

        '''

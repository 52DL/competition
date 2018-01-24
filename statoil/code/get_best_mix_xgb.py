import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss,accuracy_score

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

train = pd.read_csv('../feat/mixIxgb_train.csv')
test = pd.read_csv('../feat/mixIxgb_test.csv')
col = [c for c in train.columns if c not in ['id']]
#print(col)
X = train[col]
#print(X.info())
X = train[col].values
#print(X[0])
print(X.shape)
test_df = test[col].values
print(test_df.shape)



y_valid_pred = 0.0*y
y_train_pred = 0.0*y
y_test_pred = 0.0

# Set up folds
K = 5
kf = KFold(n_splits = K, random_state = 1108, shuffle = True)
np.random.seed(1108)
for m in [0]:
    for n in [0]:
        print('m: ',m)
        print('n: ',n)
        log_model = XGBClassifier(
                            n_estimators=5000,
                            max_depth=5,
                            objective="binary:logistic",
                            learning_rate=.01,
                            subsample=.6,
                            min_child_weight=.4,
                            colsample_bytree=.6,
                            scale_pos_weight=1,
                            gamma=0,
                            reg_alpha=.1,
                            reg_lambda=.1,
                         )
        for i, (train_index, test_index) in enumerate(kf.split(X)):

            # Create data for this fold
            y_train, y_valid = y[train_index].copy(), y[test_index]
            X_train, X_valid = X[train_index,:].copy(), X[test_index,:].copy()
            X_test = test_df.copy()
            #print( "\nFold ", i)

            # Run model for this fold
            eval_set=[(X_valid,y_valid)]
            fit_model = log_model.fit( X_train, y_train,
                               eval_set=eval_set,
                               eval_metric='logloss',
                               early_stopping_rounds=200,
                               verbose=False
                               #verbose=1000
                             )

            # Generate validation predictions for this fold
            pred = fit_model.predict_proba(X_valid, ntree_limit=log_model.best_ntree_limit)[:,1]
            #print(fit_model.coef_)
            #for i,j in zip(feat_names, fit_model.coef_[0]):
            #    print(i, " : ", j)
            #print( "  Gini = ", log_loss(y_valid, pred), accuracy_score(y_valid,np.round(pred)))
            y_valid_pred[test_index] = pred

            # Accumulate test set predictions
            probs = fit_model.predict_proba(X_test)[:,1]
            y_test_pred += probs

            del X_test, X_train, X_valid, y_train
        y_test_pred /= K  # Average test set predictions
        print( "\nGini for full training set:" ,log_loss(y, y_valid_pred))
        # Save validation predictions for stacking/ensembling
        val = pd.DataFrame()
        val['id'] = id_train
        val['is_iceberg'] = y_valid_pred
        val.to_csv('../subm/mix+xgb_valid.csv', float_format='%.6f', index=False)

        # Create submission file
        sub = pd.DataFrame()
        sub['id'] = id_test
        sub['is_iceberg'] = y_test_pred
        sub.to_csv('../subm/mix+xgb_submit.csv', float_format='%.6f', index=False)


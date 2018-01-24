"""
  __file__
    ensemble.py
  __description__
    this file ensemble the best models
  __author__
    qufu
"""
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

raw_rank = False
feat_names = []
#feat_names.append("orig+TLvgg16+Adam")
#feat_names.append("orig+TLres50")
#feat_names.append("orig1+TLvgg16+Adam")
feat_names.append("orig1+TLres50")

#feat_names.append("trans+TLvgg16+Adam")
#feat_names.append("trans+TLres50")
feat_names.append("trans1+TLvgg16+Adam")
#feat_names.append("trans1+TLres50")

#feat_names.append("norm+TLvgg16")
#feat_names.append("norm+TLres50")
#feat_names.append("norm1+TLvgg16")
#feat_names.append("norm1+TLres50")

feat_names.append("normf1+TLvgg16")
#feat_names.append("norm+lenet")

#feat_names.append("ctt+xgb")
#feat_names.append("picf+xgb")
#feat_names.append("pca+lgb")

feat_names.append("mix+lr")
#feat_names.append("mix+xgb")
feat_names.append("mix+lgb")
feat_names.append("mix+svm")

valids = [pd.read_csv('../subm/'+f+'_valid.csv') for f in feat_names]
valid = pd.merge(valids[0],valids[1] , how='left',on='id')
for i in range(2,len(feat_names)):
    valid = pd.merge(valid,valids[i],how='left',on='id')
cols = ['id']+feat_names
valid.columns = cols

valids = [pd.read_csv('../subm/'+f+'_submit.csv') for f in feat_names]
test = pd.merge(valids[0],valids[1] , how='left',on='id')
for i in range(2,len(feat_names)):
    test = pd.merge(test,valids[i],how='left',on='id')
cols = ['id']+feat_names
test.columns = cols

def transfrom(df):
    df['max'] = df.iloc[:,:6].max(axis=1)
    df['min'] = df.iloc[:,:6].min(axis=1)
    df['mean'] = df.iloc[:,:6].mean(axis=1)
    df['median'] = df.iloc[:,:6].median(axis=1)
    df['rt.15'] = np.sum((df.iloc[:,:]>0.15).astype(int),axis=1)#/6.0
    df['rt.5'] = np.sum((df.iloc[:,:]>0.5).astype(int),axis=1)#/6.0
    df['rt.85'] = np.sum((df.iloc[:,:]>0.85).astype(int),axis=1)#/6.0
    return df

print(valid.shape)
print(test.shape)
dfTrain = pd.read_json("../input/train.json")
dfTest = pd.read_json("../input/test.json")
numTrain = dfTrain.shape[0]
numTest = dfTest.shape[0]
y = dfTrain["is_iceberg"]
id_train = np.array(dfTrain["id"])
id_test = np.array(dfTest["id"])



X = valid.drop(['id'],axis=1)
test_df = test.drop(['id'],axis=1)
X = transfrom(X)
test_df = transfrom(test_df)
y_valid_pred = 0.0*y
y_test_pred = 0.0
print(X.shape)
print(test_df.shape)
# Set up folds
K = 5
kf = KFold(n_splits = K, random_state = 1108, shuffle = True)
np.random.seed(1108)
# Set up classifier
log = 'xgb'
for m in [.6]:
    for n in [.6]:
        print('m: ',m)
        print('m: ',n)
        xgbmodel = XGBClassifier(
                                n_estimators=3000,
                                max_depth=3,
                                objective="binary:logistic",
                                learning_rate=0.005,
                                subsample=1,
                                min_child_weight=5,
                                colsample_bytree=1,
                                scale_pos_weight=1,
                                gamma=1,
                                reg_alpha=1.2,
                                reg_lambda=.4,
                             )

        # Run CV

        for i, (train_index, test_index) in enumerate(kf.split(X)):
            
          # Create data for this fold
          y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index]
          X_train, X_valid = X.iloc[train_index,:].copy(), X.iloc[test_index,:].copy()
          X_test = test_df.copy()
          print( "\nFold ", i)

          eval_set=[(X_valid,y_valid)]
          fit_model = xgbmodel.fit( X_train, y_train,
                                   eval_set=eval_set,
                                   eval_metric='logloss',
                                   early_stopping_rounds=100,
                                   #verbose=100
                                   verbose=False
                                 )
          print( "  Best N trees = ", xgbmodel.best_ntree_limit )
          print( "  Best gini = ", xgbmodel.best_score )

          # Generate validation predictions for this fold
          pred = fit_model.predict_proba(X_valid, ntree_limit=xgbmodel.best_ntree_limit+0)[:,1]
          print( "  Gini = ", log_loss(y_valid, pred) )
          y_valid_pred[test_index] = pred

          # Accumulate test set predictions
          probs = fit_model.predict_proba(X_test, ntree_limit=xgbmodel.best_ntree_limit+0)[:,1]  
              
          y_test_pred += probs
            
          del X_test, X_train, X_valid, y_train
           
        y_test_pred /= K  # Average test set predictions

        print( "\nGini for full training set:" ,log_loss(y, y_valid_pred))

        # Create submission file
        val = pd.DataFrame()
        val['id'] = id_train
        val['is_iceberg'] = y_valid_pred
        val.to_csv('../subm/ensemble/%s9_valid.csv' % log, float_format='%.6f', index=False)

        # Create submission file
        sub = pd.DataFrame()
        sub['id'] = id_test
        sub['is_iceberg'] = y_test_pred
        sub.to_csv('../subm/ensemble/%s9_submit.csv' % log, float_format='%.6f', index=False)

        '''
        log_model.fit(srank, y)
        rank_res_train = log_model.predict_proba(srank)[:,1]
        rank_res_test = log_model.predict_proba(trank)[:,1]
        output = pd.DataFrame({'id':id_test, 'target':rank_res_test})
        #output.to_csv("../subm/ensemble_rank_logisticregression.csv" , index=False)
        print("rank gini: %.5f" % log_loss(y, rank_res_train))

        if raw_rank:
          log_model = LogisticRegression()
          log_model.fit(sraw, y)
          raw_res_train = log_model.predict_proba(sraw)[:,1]
          raw_res_test = log_model.predict_proba(traw)[:,1]
          output = pd.DataFrame({'id':id_test, 'target':raw_res_test})
          #output.to_csv("../subm/ensemble_raw_logisticregression.csv" , index=False)
          print("raw gini: %.5f" % log_loss(y, raw_res_train))


        '''

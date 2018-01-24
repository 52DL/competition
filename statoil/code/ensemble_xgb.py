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
from sklearn.metrics import log_loss,accuracy_score
from xgboost import XGBClassifier

raw_rank = False
feat_names = []
#feat_names.append("orig+TLvgg16+Adam")
#feat_names.append("orig+TLres50")
feat_names.append("orig1+TLvgg16+Adam")
feat_names.append("orig1+TLres50")

feat_names.append("all+TLvgg16+Adam")
#feat_names.append("trans+TLvgg16+Adam")
#feat_names.append("trans+TLres50")
feat_names.append("trans1+TLvgg16+Adam")
#feat_names.append("trans1+TLres50")

#feat_names.append("norm+TLvgg16")
#feat_names.append("norm+TLres50")
feat_names.append("norm1+TLvgg16")
#feat_names.append("norm1+TLres50")

feat_names.append("normf1+TLvgg16")
#feat_names.append("norm+lenet")

feat_names.append("ctt+xgb")
feat_names.append("picf+xgb")
feat_names.append("pca+lgb")

feat_names.append("mix+lr")
#feat_names.append("mix+xgb")
feat_names.append("mix+lgb")
feat_names.append("mix+svm")
#feat_names.append("mix+dnn")
"""
feat_names = []
feat_names.append("andy_xgb_all")
feat_names.append("andy_rgf")
feat_names.append("andy_lgb_all")
feat_names.append("the1ow_xgb_all")
feat_names.append("the1ow_rgf")
feat_names.append("the1ow_lgb_all")
feat_names.append("eddy_nnad5")
feat_names.append("scirp_gp")
feat_names.append("daia_xgb_all")
"""

pred_path = "_valid.csv"
subm_path = "_submit.csv"

dfTrain = pd.read_json("../input/train.json")
dfTest = pd.read_json("../input/test.json")
numTrain = dfTrain.shape[0]
numTest = dfTest.shape[0]
y = dfTrain["is_iceberg"]
id_train = np.array(dfTrain["id"])
id_test = np.array(dfTest["id"])


srank = np.zeros((numTrain, len(feat_names)))
trank = np.zeros((numTest, len(feat_names)))

for i, feat_name in enumerate(feat_names):
  train = pd.read_csv("../subm/%s%s" % ( feat_name,pred_path))
  test = pd.read_csv("../subm/%s%s" % ( feat_name,subm_path))

  train_raw = np.array(train["is_iceberg"])
  test_raw = np.array(test["is_iceberg"])
  #print(train_raw.shape)
  #print(test_raw.shape)
  print("%s: %.5f" % (feat_name, log_loss(y, train_raw)))
  #print("%s: %.5f" % (feat_name, gini_normalized(y2, test_raw)))
  
  srank[:,i] = train_raw
  trank[:,i] = test_raw

X = pd.DataFrame(srank)
test_df = pd.DataFrame(trank)
y_valid_pred = 0*y
y_test_pred = 0

# Set up folds
K = 5
kf = KFold(n_splits = K, random_state = 1108, shuffle = True)
np.random.seed(1108)
# Set up classifier
log = 'xgb'
for m in [0]:
    for n in [0]:
        print('m & n:',m,n)
        xgbmodel = XGBClassifier(
                                n_estimators=1000,
                                max_depth=5,
                                objective="binary:logistic",
                                learning_rate=0.01,
                                subsample=1,
                                min_child_weight=.7,
                                colsample_bytree=1,
                                scale_pos_weight=1,
                                gamma=0,
                                reg_alpha=.2,
                                reg_lambda=.8,
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
                                   verbose=False
                                 )
          #print( "  Best N trees = ", xgbmodel.best_ntree_limit )
          #print( "  Best gini = ", xgbmodel.best_score )

          # Generate validation predictions for this fold
          pred = fit_model.predict_proba(X_valid, ntree_limit=xgbmodel.best_ntree_limit+0)[:,1]
          print( "  Gini = ", log_loss(y_valid, pred) )
          y_valid_pred[test_index] = pred

          # Accumulate test set predictions
          probs = fit_model.predict_proba(X_test, ntree_limit=xgbmodel.best_ntree_limit+0)[:,1]  
              
          y_test_pred += probs
            
          del X_test, X_train, X_valid, y_train
           
        y_test_pred /= K  # Average test set predictions

        print( "\nGini for full training set:" ,log_loss(y, y_valid_pred),accuracy_score(y,np.round(y_valid_pred)))

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

"""
  __file__
    ensemble.py
  __description__
    this file ensemble the best models
  __author__
    qufu
"""
from sklearn.model_selection import KFold,train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss,accuracy_score
raw_rank = False
feat_names = []
#feat_names.append("orig+TLvgg16+Adam")
#feat_names.append("orig+TLres50")
#feat_names.append("orig1+TLvgg16+Adam")
feat_names.append("orig1+TLres50")

#feat_names.append("all+TLvgg16+Adam")

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

#feat_names.append("mix+lr")
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

valids = [pd.read_csv('../subm/'+f+'_valid.csv') for f in feat_names]
valid = pd.merge(valids[0],valids[1] , how='left',on='id')
for i in range(2,len(feat_names)):
    valid = pd.merge(valid,valids[i],how='left',on='id')
cols = ['id']+feat_names
valid.columns = cols
train=valid
#valid['is_iceberg'] = y
tests = [pd.read_csv('../subm/'+f+'_submit.csv') for f in feat_names]
test = pd.merge(tests[0],tests[1] , how='left',on='id')
for i in range(2,len(feat_names)):
    test = pd.merge(test,tests[i],how='left',on='id')
cols = ['id']+feat_names
test.columns = cols
col = [c for c in train.columns if c not in ['id']]
#print(col)
X = train.loc[:,col]
#print(X.info())
print(X.shape)
test_df = test.loc[:,col]
print(test_df.shape)
from sklearn.preprocessing import MinMaxScaler
#blend = np.concatenate([X,test_df])
#scaling = MinMaxScaler(feature_range=(-1,1)).fit(blend)
#blend = scaling.transform(blend)
#X = pd.DataFrame(blend[:X.shape[0],:],columns=X.columns)
#test_df = pd.DataFrame(blend[X.shape[0]:,:],columns=test_df.columns)
scaling = MinMaxScaler(feature_range=(-1,1)).fit(X)
X = pd.DataFrame(scaling.transform(X),columns=X.columns)
test_df = pd.DataFrame(scaling.transform(test_df),columns=test_df.columns)

y_valid_pred = 0*y
y_test_pred = 0

# Set up folds
K = 5
kf = KFold(n_splits = K, random_state = 1108, shuffle = True)
np.random.seed(1108)

log = 'svm'
if log == 'svm':
    from sklearn.svm import SVC
    log_model = SVC(C=0.003,probability=True,random_state=1)
if log == 'lr':
    from sklearn.linear_model import LogisticRegression
    C_regu = 1
    lr_model = LogisticRegression(penalty='l1',C=C_regu)
    log_model = lr_model
# Run CV

for i, (train_index, test_index) in enumerate(kf.split(X)):
    
  # Create data for this fold
  y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index]
  X_train, X_valid = X.iloc[train_index,:].copy(), X.iloc[test_index,:].copy()
  X_test = test_df.copy()
  print( "\nFold ", i)
  
  # Run model for this fold
  fit_model = log_model.fit( X_train, y_train )
      
  # Generate validation predictions for this fold
  pred = fit_model.predict_proba(X_valid)[:,1]
  #print(fit_model.coef_)
  #for i,j in zip(feat_names, fit_model.coef_[0]):
  #    print(i, " : ", j)
  print( "  Gini = ", log_loss(y_valid, pred) )
  y_valid_pred.iloc[test_index] = pred
  
  # Accumulate test set predictions
  probs = fit_model.predict_proba(X_test)[:,1]
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

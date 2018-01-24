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

raw_rank = False
feat_names = []
feat_names.append("andy_xgb")
feat_names.append("andy_rgf")
feat_names.append("andy_lgb")
feat_names.append("the1ow_xgb")
feat_names.append("the1ow_rgf")
feat_names.append("the1ow_lgb")
feat_names.append("eddy_nnad5")
feat_names.append("daia_xgb")
feat_names.append("scirp_gp")

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

dfTrain = pd.read_csv("../../../input/train.csv")
dfTest = pd.read_csv("../../../input/test.csv")
numTrain = dfTrain.shape[0]
numTest = dfTest.shape[0]
y = dfTrain["target"]
id_train = np.array(dfTrain["id"])
id_test = np.array(dfTest["id"])

def eval_gini(y_true, y_prob):
    """
    Original author CPMP : https://www.kaggle.com/cpmpml
    In kernel : https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
    """
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

#https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703#5897
def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

def get_rank(df):
  rank = df.argsort().argsort()
  rank = 1.0*rank/np.max(rank)
  return rank

srank = np.zeros((numTrain, len(feat_names)))
trank = np.zeros((numTest, len(feat_names)))

for i, feat_name in enumerate(feat_names):
  train = pd.read_csv("../subm/%s%s" % ( feat_name,pred_path))
  test = pd.read_csv("../subm/%s%s" % ( feat_name,subm_path))

  train_raw = np.array(train["target"])
  test_raw = np.array(test["target"])
  #print(train_raw.shape)
  #print(test_raw.shape)
  print("%s: %.5f" % (feat_name, eval_gini(y, train_raw)))
  #print("%s: %.5f" % (feat_name, gini_normalized(y2, test_raw)))
  
  train_rank = get_rank(train_raw)
  test_rank = get_rank(test_raw)

  srank[:,i] = train_rank
  trank[:,i] = test_rank

  if raw_rank:
    srank[:,i] = train_raw
    trank[:,i] = test_raw

X = pd.DataFrame(srank)
test_df = pd.DataFrame(trank)
y_valid_pred = 0*y
y_test_pred = 0

# Set up folds
K = 5
kf = KFold(n_splits = K, random_state = 1, shuffle = True)
np.random.seed(0)

log = 'qda'
if log == 'lr':
    from sklearn.linear_model import LogisticRegression
    C_regu = 0.003
    lr_model = LogisticRegression(C=C_regu)
    log_model = lr_model
if log == 'nb':
    from sklearn.naive_bayes import GaussianNB
    lr_model = GaussianNB()
    log_model = lr_model
if log == 'lda':
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lr_model = LinearDiscriminantAnalysis()
    log_model = lr_model
if log == 'qda':
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    lr_model = QuadraticDiscriminantAnalysis()
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
  #for i,j in zip(feat_names, fit_model.coef_[0]):
  #    print(i, " : ", j)
  print( "  Gini = ", eval_gini(y_valid, pred) )
  y_valid_pred.iloc[test_index] = pred
  
  # Accumulate test set predictions
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

# Create submission file
val = pd.DataFrame()
val['id'] = id_train
val['target'] = y_valid_pred
val.to_csv('../subm/ensemble/%s9_valid.csv' % log, float_format='%.6f', index=False)

# Create submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_test_pred
sub.to_csv('../subm/ensemble/%s9_submit.csv' % log, float_format='%.6f', index=False)

'''
log_model.fit(srank, y)
rank_res_train = log_model.predict_proba(srank)[:,1]
rank_res_test = log_model.predict_proba(trank)[:,1]
output = pd.DataFrame({'id':id_test, 'target':rank_res_test})
#output.to_csv("../subm/ensemble_rank_logisticregression.csv" , index=False)
print("rank gini: %.5f" % eval_gini(y, rank_res_train))

if raw_rank:
  log_model = LogisticRegression()
  log_model.fit(sraw, y)
  raw_res_train = log_model.predict_proba(sraw)[:,1]
  raw_res_test = log_model.predict_proba(traw)[:,1]
  output = pd.DataFrame({'id':id_test, 'target':raw_res_test})
  #output.to_csv("../subm/ensemble_raw_logisticregression.csv" , index=False)
  print("raw gini: %.5f" % eval_gini(y, raw_res_train))


'''

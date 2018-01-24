"""
  __file__
    ensemble.py
  __description__
    this file ensemble the best models
  __author__
    qufu
"""

import numpy as np
import pandas as pd

feat_names = []
feat_names.append("andy_xgb")
feat_names.append("andy_rgf")
feat_names.append("the1ow_xgb")
feat_names.append("the1ow_rgf")

raw_rank = True

pred_path = "_valid.csv"
subm_path = "_submit.csv"

dfTrain = pd.read_csv("../../../input/train.csv")
dfTest = pd.read_csv("../../../input/test.csv")
numTrain = dfTrain.shape[0]
numTest = dfTest.shape[0]
y = np.array(dfTrain["target"])
y2 = np.ones((numTest,))
y2[:numTrain] = y
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
if raw_rank:
  sraw = np.zeros((numTrain, len(feat_names)))
  traw = np.zeros((numTest, len(feat_names)))

for i, feat_name in enumerate(feat_names):
  train = pd.read_csv("../subm/%s%s" % ( feat_name,pred_path))
  test = pd.read_csv("../subm/%s%s" % ( feat_name,subm_path))

  train_raw = np.array(train["target"])
  test_raw = np.array(test["target"])
  print(train_raw.shape)
  print(test_raw.shape)
  print("%s: %.5f" % (feat_name, eval_gini(y, train_raw)))
  #print("%s: %.5f" % (feat_name, gini_normalized(y2, test_raw)))
  
  train_rank = get_rank(train_raw)
  test_rank = get_rank(test_raw)

  srank[:,i] = train_rank
  trank[:,i] = test_rank

  if raw_rank:
    sraw[:,i] = train_raw
    traw[:,i] = test_raw

from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
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




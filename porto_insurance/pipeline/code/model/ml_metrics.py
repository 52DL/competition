"""
  __file__
    ml_metrics.py
  __description__
    this file provide the gini func
  __author__
    qufu
"""

import numpy as np

# Define the gini metric - from https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703#5897
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

# Create an XGBoost-compatible metric from Gini
def gini_xgb(preds, dtrain):
  labels = dtrain.get_label()
  gini_score = gini_normalized(labels, preds)
  return [('gini', gini_score)]

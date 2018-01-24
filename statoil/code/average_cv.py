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
raw_rank = False
feat_names = []
feat_names.append("orig+TLvgg16+Adam")
#feat_names.append("orig+TLres50")
feat_names.append("trans+TLvgg16+Adam")
#feat_names.append("trans+TLres50")
feat_names.append("norm+TLvgg16")
feat_names.append("norm+TLres50")
feat_names.append("ctt+xgb")

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
np.random.seed(1108)

preds = X.mean(axis=1)
print("cv score", log_loss(y, preds))

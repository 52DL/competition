"""
  __file__
    genFeat_nan_feat.py
  __description__
    this file create  features that tells the station of nan 
  __author__
    qufu
"""

import sys
import pickle
import numpy as np
import pandas as pd
from feat_utils import try_divide, dump_feat_name
from multiprocessing import *
sys.path.append("../")
from param_config import config
 
if __name__ == "__main__":
  
  ##loading data##
  dfTrain = pd.read_csv(config.processed_train_data_path)
  dfTest = pd.read_csv(config.processed_test_data_path)
  with open("%s/stratifiedKFold.%s.pkl" % (config.data_folder, config.stratified_label), "rb") as f:
    skf = pickle.load(f)

  ##generating feature##
  print("==================================================")
  print("Generate both features...")

  dfTrain = dfTrain.fillna(-1)
  dfTest = dfTest.fillna(-1)

  dcol = [c for c in dfTrain.columns if c not in ['id','target']]
  dfTrain['nan_sum'] = np.sum((dfTrain[dcol]==-1).values, axis=1)
  dfTest['nan_sum'] = np.sum((dfTest[dcol]==-1).values, axis=1)
  feat_names = [name for name in dfTrain.columns if "nan" in name]
  feat_name_file = "%s/nan.feat_name" % config.feat_folder

  print("For cross-validation...")
  for run in range(config.n_runs):
    ## use 33% for training and 67 % for validation
    ## so we switch trainInd and validInd
    for fold, (validInd, trainInd) in enumerate(skf[run]):
      print("Run: %d, Fold: %d" % (run+1, fold+1))
      path = "%s/Run%d/Fold%d" % (config.feat_folder, run+1, fold+1)

      for feat_name in feat_names:
        X_train = dfTrain[feat_name].values[trainInd]
        X_valid = dfTrain[feat_name].values[validInd]

        with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
          pickle.dump(X_train, f, 2)
        with open("%s/valid.%s.feat.pkl" % (path, feat_name), "wb") as f:
          pickle.dump(X_valid, f, 2)
        #X_train.to_csv("%s/train.%s.feat" % (path, feat_name))
        #X_test.to_csv("%s/test.%s.feat" % (path, feat_name))
  
  print("Done.")

  print("For training and testing...")
  path = "%s/All" % config.feat_folder
  ## use full version for X_train
                      
  for feat_name in feat_names:
    X_train = dfTrain[feat_name].values
    X_test = dfTest[feat_name].values

    with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
      pickle.dump(X_train, f, 2)
    with open("%s/test.%s.feat.pkl" % (path, feat_name), "wb") as f:
      pickle.dump(X_test, f, 2)
   # X_train.to_csv("%s/train.%s.feat" % (path, feat_name))
   # X_test.to_csv("%s/test.%s.feat" % (path, feat_name))
    
  ## save feat name
  print("Feature names are stored in %s" % feat_name_file)
  ## dump feat name
  dump_feat_name(feat_names, feat_name_file)
  print("All Done.")

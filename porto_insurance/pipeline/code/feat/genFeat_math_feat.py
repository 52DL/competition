"""
  __file__
    genFeat_math_feat.py
  __description__
    this file create standard arithmetic
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

def transform_df(df):
  df = pd.DataFrame(df)
  dcol = [c for c in df.columns if c not in ['id','target']]
  for c in dcol:
    if '_bin' not in c:
      df[c+'math'+str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
      df[c+'math'+str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)
      #df[c+str('_sq')] = np.power(df[c].values,2).astype(np.float32)
      #df[c+str('_sqr')] = np.square(df[c].values).astype(np.float32)
      #df[c+str('_log')] = np.log(np.abs(df[c].values) + 1)
      #df[c+str('_exp')] = np.exp(df[c].values) - 1
  return df
 
def multi_transform(df):
  print('Init Shape: ', df.shape)
  p = Pool(cpu_count())
  df = p.map(transform_df, np.array_split(df, cpu_count()))
  df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
  p.close(); p.join()
  print('After Shape: ', df.shape)
  return df

if __name__ == "__main__":
  
  ##loading data##
  dfTrain = pd.read_csv(config.processed_train_data_path)
  dfTest = pd.read_csv(config.processed_test_data_path)
  print(dfTrain.shape)
  print(dfTest.shape)
  with open("%s/stratifiedKFold.%s.pkl" % (config.data_folder, config.stratified_label), "rb") as f:
    skf = pickle.load(f)

  ##generating feature##
  print("==================================================")
  print("Generate math features...")

  d_median = dfTrain.median(axis=0)
  d_mean = dfTrain.mean(axis=0)
  dfTrain = dfTrain.fillna(-1)
  dfTest = dfTest.fillna(-1)

  dfTrain = multi_transform(dfTrain)
  dfTest = multi_transform(dfTest)
  feat_names = [name for name in dfTrain.columns if "math" in name]
  feat_name_file = "%s/math.feat_name" % config.feat_folder

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

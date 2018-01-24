"""
  __file__
    genFeat_oh_feat.py
  __description__
    this file create one-hot arithmetic
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
  for c in one_hot:
    if len(one_hot[c])>2 and len(one_hot[c]) < 7:
      for val in one_hot[c]:
        df[c+'_oh_' + str(val)] = (df[c].values == val).astype(np.int)
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
  with open("%s/stratifiedKFold.%s.pkl" % (config.data_folder, config.stratified_label), "rb") as f:
    skf = pickle.load(f)

  ##generating feature##
  print("==================================================")
  print("Generate one-hot features...")

  dfTrain = dfTrain.fillna(-1)
  dfTest = dfTest.fillna(-1)
  
  one_hot = {c: list(dfTrain[c].unique()) for c in dfTrain.columns if c not in ['id','target']}
  dfTrain = multi_transform(dfTrain)
  dfTest = multi_transform(dfTest)
  feat_names = [name for name in dfTrain.columns if "oh" in name]
  feat_name_file = "%s/oh.feat_name" % config.feat_folder

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

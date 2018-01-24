"""
  __file__
    gen_info.py
  __description__
    this file make the info for the desied feature
  __author__
    qufu
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
sys.path.append("../")
from param_config import config

def gen_info(feat_path_name):
  ###############
  ## Load Data ##
  ###############
  ## load data
  dfTrain = pd.read_csv(config.processed_train_data_path)
  dfTest = pd.read_csv(config.processed_test_data_path)
  dfTrain = dfTrain.fillna(-1)
  dfTest = dfTest.fillna(-1)
  dfTest["target"] = np.ones((dfTest.shape[0]))

  with open("%s/stratifiedKFold.%s.pkl" % (config.data_folder, config.stratified_label), "rb") as f:
    skf = pickle.load(f)

  #######################
  ## Generate Features ##
  #######################
  print("Generate info...")
  print("For cross-validation...")
  for run in range(config.n_runs):
    ## use 33% for training and 67 % for validation
    ## so we switch trainInd and validInd
    for fold, (validInd, trainInd) in enumerate(skf[run]):
      print("Run: %d, Fold: %d" % (run+1, fold+1))
      path = "%s/%s/Run%d/Fold%d" % (config.feat_folder, feat_path_name, run+1, fold+1)
      if not os.path.exists(path):
        os.makedirs(path)

      #############################
      ## get and dump group info ##
      #############################
      np.savetxt("%s/train.feat.group" % path, [len(trainInd)], fmt="%d")
      np.savetxt("%s/valid.feat.group" % path, [len(validInd)], fmt="%d")

      #############################
      ## dump all the id info ##
      #############################
      dfTrain["id"].iloc[trainInd].to_csv("%s/train.id.info" % path, index=False, header=True)
      dfTrain["id"].iloc[validInd].to_csv("%s/valid.id.info" % path, index=False, header=True)
  print("Done.")

  print("For training and testing...")
  path = "%s/%s/All" % (config.feat_folder, feat_path_name)
  if not os.path.exists(path):
    os.makedirs(path)
  ## group
  np.savetxt("%s/%s/All/train.feat.group" % (config.feat_folder, feat_path_name), [dfTrain.shape[0]], fmt="%d")
  np.savetxt("%s/%s/All/test.feat.group" % (config.feat_folder, feat_path_name), [dfTest.shape[0]], fmt="%d")

  ## info
  #dfTrain.to_csv("%s/%s/All/train.info" % (config.feat_folder, feat_path_name), index=False, header=True)
  #dfTest.to_csv("%s/%s/All/test.info" % (config.feat_folder, feat_path_name), index=False, header=True)


  #############################
  ## dump all the id info ##
  #############################
  dfTrain["id"].to_csv("%s/train.id.info" % path, index=False, header=True)
  dfTest["id"].to_csv("%s/test.id.info" % path, index=False, header=True)

  print("All Done.")


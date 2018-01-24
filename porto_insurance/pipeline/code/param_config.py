"""
  __file__
    param_config.py
  __description__
    this file make some configrations for the project
  __author__
    qufu
"""

import os
import numpy as np

############
## Config ##
############
class ParamConfig:
  def __init__(self,
          feat_folder):
    self.n_classes = 4
    self.preload = True
    ## CV params
    self.n_runs = 3
    self.n_folds = 3
    self.stratified_label = "target"
    self.stratified_key = "target"
    #self.stratified_key = "ps_int_05_cat"

    ## Path
    self.data_folder = "../../data"
    self.feat_folder = feat_folder
    self.sub_folder = "../subm"
    self.original_train_data_path = "%s/train.csv" % self.data_folder
    self.original_test_data_path = "%s/test.csv" % self.data_folder
    self.processed_train_data_path = "%s/train.processed.csv" % self.feat_folder
    self.processed_test_data_path = "%s/test.processed.csv" % self.feat_folder

    ## Others

    ## Create feat folder
    if not os.path.exists(self.feat_folder):
      os.makedirs(self.feat_folder)

    ## Create sub folder
    if not os.path.exists(self.sub_folder):
      os.makedirs(self.sub_folder)

    ## Create feat for training and testing feature
    if not os.path.exists("%s/All" % self.feat_folder):
      os.makedirs("%s/All" % self.feat_folder)

    ## Create feat for each run and fold
    for run in range(1, self.n_runs+1):
      for fold in range(1, self.n_folds+1):
        path = "%s/Run%d/Fold%d" % (self.feat_folder, run, fold)
        if not os.path.exists(path):
          os.makedirs(path)

## initialize one param config
config = ParamConfig(feat_folder="../../feat/solution")


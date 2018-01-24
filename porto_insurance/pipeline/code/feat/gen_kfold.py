"""
  __file__
    gen_kfold.py
  __description__
    this file generate the kfold files for the model CV
  __author__
    qufu
"""

import sys
import pandas as pd
import pickle 
from sklearn.cross_validation import StratifiedKFold
sys.path.append("../")
from param_config import config

if __name__ == "__main__":
  
  ##load data##
  dfTrain = pd.read_csv(config.processed_train_data_path)

  skf = [0]*config.n_runs
  stratified_label = config.stratified_label
  key = config.stratified_key
  for run in range(config.n_runs):
    random_seed = 2017 + 1000 * (run + 1)
    skf[run] = StratifiedKFold(dfTrain[key], n_folds=config.n_folds, shuffle=True, random_state=random_seed)
    for fold, (validInd, trainInd) in enumerate(skf[run]):
      print("================================")
      print("Index for run: %s, fold: %s" % (run+1, fold+1))
      print("Train (num = %s)" % len(trainInd))
      print(trainInd[:10])
      print("Valid (num = %s)" % len(validInd))
      print(validInd[:10])
  with open("%s/stratifiedKFold.%s.pkl" % (config.data_folder, stratified_label), "wb") as f:
    pickle.dump(skf, f, 2)

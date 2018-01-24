"""
  __file__
    train_model.py
  __description__
    this file finds the best params for cv
  __author__
    qufu
"""

import sys
import csv
import os
import pickle
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import hstack
## sklearn
from sklearn.base import BaseEstimator
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import Ridge, Lasso, LassoLars, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
## hyperopt
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
## keras
#from keras.models import Sequential
#from keras.layers.core import Dense, Dropout, Activation
#from keras.layers.normalization import BatchNormalization
#from keras.layers.advanced_activations import PReLU
#from keras.utils import np_utils, generic_utils
## cutomized module
from model_library_config import feat_folders, feat_names, param_spaces, int_feat
sys.path.append("../")
from param_config import config
from ml_metrics import gini_xgb
from utils_csv import *


global trial_counter
global log_handler

output_path = "../../Output"

### global params
## you can use bagging to stabilize the predictions
bootstrap_ratio = 1
bootstrap_replacement = False
bagging_size= 1

ebc_hard_threshold = False
verbose_level = 1

#### warpper for hyperopt for logging the training reslut
# adopted from
#
def hyperopt_wrapper(param, feat_folder, feat_name, cv_data, tt_data):
  global trial_counter
  global log_handler
  trial_counter += 1

  # convert integer feat
  for f in int_feat:
#    if param.has_key(f):
    if f in param:
      param[f] = int(param[f])

  print("------------------------------------------------------------")
  print("Trial %d" % trial_counter)

  print("        Model")
  print("              %s" % feat_name)
  print("        Param")
  for k,v in sorted(param.items()):
    print("              %s: %s" % (k,v))
  print("        Result")
  print("                    Run      Fold      Bag      Kappa      Shape")
  start = time.clock()
  ## evaluate performance
  gini_cv_mean, gini_cv_std = hyperopt_obj(param, feat_folder, feat_name, trial_counter, cv_data, tt_data)
  print("train one trail time used:",(time.clock()-start))
  ## log
  var_to_log = [
          "%d " % trial_counter,
          "%.6f " % gini_cv_mean,
          "%.6f " % gini_cv_std
  ]
  
  for k,v in sorted(param.items()):
    var_to_log.append("%s " % v)
  writer.writerow(var_to_log)
  log_handler.flush()

  return {'loss': -gini_cv_mean, 'attachments': {'std': gini_cv_std}, 'status': STATUS_OK}

def hyperopt_obj(param, feat_folder, feat_name, trial_counter, cv_data, tt_data):
  gini_cv = np.zeros((config.n_runs, config.n_folds), dtype=float)
  for run in range(1,config.n_runs+1):
    for fold in range(1,config.n_folds+1):
      rng = np.random.RandomState(2017 + 1000 * run + 10 * fold)

      #### all the path
      save_path = "%s/Run%d/Fold%d" % (output_path, run, fold)
      # result path
      pred_valid_path = "%s/valid.raw.pred.%s_[Id@%d].csv" % (save_path, feat_name, trial_counter)
      if not os.path.exists(save_path):
        os.makedirs(save_path)
      
      if config.preload:
        X_train = cv_data[str(run)+str(fold)]["X1"]
        X_valid = cv_data[str(run)+str(fold)]["X2"]
        labels_train = cv_data[str(run)+str(fold)]["Y1"]
        labels_valid = cv_data[str(run)+str(fold)]["Y2"]
        numTrain = cv_data[str(run)+str(fold)]["num1"]
        numValid = cv_data[str(run)+str(fold)]["num2"]
      else:
        path = "%s/Run%d/Fold%d" % (feat_folder, run, fold)
        # feat path
        feat_train_path = "%s/train.feat.csv" % path
        feat_valid_path = "%s/valid.feat.csv" % path

        # target path
        target_train_path = "%s/train.target.csv" % path
        target_valid_path = "%s/valid.target.csv" % path

        ## load feat
        X_train = pd.read_csv(feat_train_path).values
        X_valid = pd.read_csv(feat_valid_path).values
        labels_train = pd.read_csv(target_train_path).values
        labels_valid = pd.read_csv(target_valid_path).values

        ## load valid info
        numTrain = X_train.shape[0]
        numValid = X_valid.shape[0]



      ##############
      ## Training ##
      ##############
      ## you can use bagging to stabilize the predictions
      preds_bagging = np.zeros((numValid, bagging_size), dtype=float)
      for n in range(bagging_size):
        if bootstrap_replacement:
          sampleSize = int(numTrain*bootstrap_ratio)
          index_base = rng.randint(numTrain, size=sampleSize)
          index_meta = [i for i in range(numTrain) if i not in index_base]
        else:
          randnum = rng.uniform(size=numTrain)
          index_base = [i for i in range(numTrain) if randnum[i] < bootstrap_ratio]
          index_meta = [i for i in range(numTrain) if randnum[i] >= bootstrap_ratio]

        if "booster" in param:
          dvalid_base = xgb.DMatrix(X_valid, label=labels_valid)
          dtrain_base = xgb.DMatrix(X_train[index_base], label=labels_train[index_base])

          watchlist = []
          if verbose_level >= 2:
            watchlist  = [(dtrain_base, 'train'), (dvalid_base, 'valid')]

	## various models
        if param["task"] in ["classification"]:
	  ## regression & pairwise ranking with xgboost
          bst = xgb.train(param, dtrain_base, param['num_round'], watchlist, feval=gini_xgb)
          pred = bst.predict(dvalid_base)

	## weighted averageing over different models
        pred_valid = pred
	## this bagging iteration
        preds_bagging[:,n] = pred_valid
      pred_raw = np.mean(preds_bagging, axis=1)
      pred_rank = pred_raw.argsort().argsort()
      pred_valid = 1.0*pred_rank/np.max(pred_rank)
      gini_valid = gini_xgb(pred_valid, dvalid_base)
      print("                    {:>3}       {:>3}      {:>3}    {:>8}  {} x {}".format(
                                run, fold, n+1, np.round(gini_valid[0][1],6), X_train.shape[0], X_train.shape[1]))
      gini_cv[run-1,fold-1] = gini_valid[0][1]
      ## save this prediction
      dfPred = pd.DataFrame({"target": labels_valid, "prediction": pred_valid})
      dfPred.to_csv(pred_valid_path, index=False, header=True,
		   columns=["target", "prediction"])

  gini_cv_mean = np.mean(gini_cv)
  gini_cv_std = np.std(gini_cv)
  if verbose_level >= 1:
    print("              Mean: %.6f" % gini_cv_mean)
    print("              Std: %.6f" % gini_cv_std)

  ####################
  #### Retraining ####
  ####################
  #### all the path
  save_path = "%s/All" % output_path
  subm_path = "%s/Subm" % output_path
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  if not os.path.exists(subm_path):
    os.makedirs(subm_path)
  # result path
  pred_test_path = "%s/test.raw.pred.%s_[Id@%d].csv" % (save_path, feat_name, trial_counter)
  # submission path (relevance as in [1,2,3,4])
  subm_path = "%s/test.pred.%s_[Id@%d]_[Mean%.6f]_[Std%.6f].csv" % (subm_path, feat_name, trial_counter, gini_cv_mean, gini_cv_std)
  
  if config.preload:
    X_train = tt_data["X1"]
    X_test = tt_data["X2"]
    labels_train = tt_data["Y1"]
    labels_test = tt_data["Y2"]
    numTrain = tt_data["num1"]
    numTest = tt_data["num2"]
    id_test = tt_data["id_test"]
  else:
    path = "%s/All" % (feat_folder)
    # feat path
    feat_train_path = "%s/train.feat.csv" % path
    feat_test_path = "%s/test.feat.csvt" % path

    # target path
    target_train_path = "%s/train.target.csvt" % path
    target_test_path = "%s/test.target.csv" % path
    id_test_path = "%s/test.id.info" % path

    #### load data
    ## load feat
    X_train = pd.read_csv(feat_train_path).values
    X_test = pd.read_csv(feat_test_path).values
    labels_train = pd.read_csv(target_train_path).values
    labels_test = pd.read_csv(target_test_path).values

    ## load valid info
    numTrain = X_train.shape[0]
    numTest = X_test.shape[0]

    ## load test info
    info_test = pd.read_csv(id_test_path)
    id_test = info_test["id"]

 
  ## bagging
  preds_bagging = np.zeros((numTest, bagging_size), dtype=float)
  for n in range(bagging_size):
    if bootstrap_replacement:
      sampleSize = int(numTrain*bootstrap_ratio)
      #index_meta = rng.randint(numTrain, size=sampleSize)
      #index_base = [i for i in range(numTrain) if i not in index_meta]
      index_base = rng.randint(numTrain, size=sampleSize)
      index_meta = [i for i in range(numTrain) if i not in index_base]
    else:
      randnum = rng.uniform(size=numTrain)
      index_base = [i for i in range(numTrain) if randnum[i] < bootstrap_ratio]
      index_meta = [i for i in range(numTrain) if randnum[i] >= bootstrap_ratio]

    if "booster" in param:
      dtest = xgb.DMatrix(X_test, label=labels_test)
      dtrain = xgb.DMatrix(X_train[index_base], label=labels_train[index_base])

      watchlist = []
      if verbose_level >= 2:
        watchlist  = [(dtrain, 'train')]

    ## train
    if param["task"] in ["classification"]:
      bst = xgb.train(param, dtrain, param['num_round'], watchlist, feval=gini_xgb)
      pred = bst.predict(dtest)

    pred_test = pred
    preds_bagging[:,n] = pred_test
  pred_raw = np.mean(preds_bagging, axis=1)
  pred_rank = pred_raw.argsort().argsort()
  pred_test = 1.0*pred_rank/np.max(pred_rank)
  ## write
  output = pd.DataFrame({"id": id_test, "target": pred_test})
  output.to_csv(pred_test_path, index=False)

  ## write
  output = pd.DataFrame({"id": id_test, "target": pred_test})
  output.to_csv(subm_path, index=False)

  return gini_cv_mean, gini_cv_std
 
####################
## Model Buliding ##
####################
def check_model(models, feat_name):
  if models == "all":
    return True
  for model in models:
    if model in feat_name:
      return True
  return False

if __name__ == "__main__":
  specified_models = sys.argv[1:]
  if len(specified_models) == 0:
    print("You have to specify which model to train.\n")
    print("Usage: python ./train_model_library_lsa.py model1 model2 model3 ...\n")
    print("Example: python ./train_model_library_lsa.py reg_skl_ridge reg_skl_lasso reg_skl_svr\n")
    print("See model_library_config_lsa.py for a list of available models (i.e., Model@model_name)")
    sys.exit()
  log_path = "%s/Log" % output_path
  if not os.path.exists(log_path):
    os.makedirs(log_path)
 
  for feat_name, feat_folder in zip(feat_names, feat_folders):
    if not check_model(specified_models, feat_name):
      continue
    param_space = param_spaces[feat_name]
    log_file = "%s/%s_hyperopt.log" % (log_path, feat_name)
    log_handler = open( log_file, 'w' )
    writer = csv.writer( log_handler )
    headers = [ 'trial_counteri','gini_mean', 'gini_std' ]
    for k,v in sorted(param_space.items()):
      headers.append(k)
    writer.writerow( headers )
    log_handler.flush()

    print("************************************************************")
    print("Search for the best params")
    #global trial_counter
    trial_counter = 0
    trials = Trials()
    if config.preload:
      start = time.clock()
      cv_data,tt_data = get_feat_data(feat_folder)
      print("loding data time used:", (time.clock() - start))
      objective = lambda p: hyperopt_wrapper(p, feat_folder, feat_name, cv_data, tt_data)
    else :
      objective = lambda p: hyperopt_wrapper(p, feat_folder, feat_name, None, None)

    best_params = fmin(objective, param_space, algo=tpe.suggest,
		       trials=trials, max_evals=param_space["max_evals"])
    for f in int_feat:
      if f in best_params:
        best_params[f] = int(best_params[f])
    print("************************************************************")
    print("Best params")
    for k,v in best_params.items():
      print("        %s: %s" % (k,v))
    trial_ginis = -np.asarray(trials.losses(), dtype=float)
    best_gini_mean = max(trial_ginis)
    ind = np.where(trial_ginis == best_gini_mean)[0][0]
    best_gini_std = trials.trial_attachments(trials.trials[ind])['std']
    print("Kappa stats")
    print("        Mean: %.6f\n        Std: %.6f" % (best_gini_mean, best_gini_std))

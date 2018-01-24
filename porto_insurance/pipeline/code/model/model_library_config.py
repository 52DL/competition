"""
  __file__
    model_library_config.py
  __description__
    this file gives the params about diff models
  __author__
    qufu
"""
import numpy as np
from hyperopt import hp

############
## Config ##
############

#debug = True
debug = False

## xgboost
xgb_random_seed = 2017
xgb_nthread = 4 
xgb_dmatrix_silent = True

## sklearn
skl_random_seed = 2017
skl_n_jobs = 4

if debug:
    xgb_nthread = 1
    skl_n_jobs = 1
    xgb_min_num_round = 5
    xgb_max_num_round = 10
    xgb_num_round_step = 5
    skl_min_n_estimators = 5
    skl_max_n_estimators = 10
    skl_n_estimators_step = 5
    libfm_min_iter = 5
    libfm_max_iter = 10
    iter_step = 5
    hyperopt_param = {}
    hyperopt_param["xgb_max_evals"] = 1
    hyperopt_param["rf_max_evals"] = 1
    hyperopt_param["etr_max_evals"] = 1
    hyperopt_param["gbm_max_evals"] = 1
    hyperopt_param["lr_max_evals"] = 1
    hyperopt_param["ridge_max_evals"] = 1
    hyperopt_param["lasso_max_evals"] = 1
    hyperopt_param['svr_max_evals'] = 1
    hyperopt_param['dnn_max_evals'] = 1
    hyperopt_param['libfm_max_evals'] = 1
    hyperopt_param['rgf_max_evals'] = 1
else:
    xgb_min_num_round = 10
    xgb_max_num_round = 500
    xgb_num_round_step = 10
    skl_min_n_estimators = 10
    skl_max_n_estimators = 500
    skl_n_estimators_step = 10
    libfm_min_iter = 10
    libfm_max_iter = 500
    iter_step = 10
    hyperopt_param = {}
    hyperopt_param["xgb_max_evals"] = 200
    hyperopt_param["rf_max_evals"] = 200
    hyperopt_param["etr_max_evals"] = 200
    hyperopt_param["gbm_max_evals"] = 200
    hyperopt_param["lr_max_evals"] = 200
    hyperopt_param["ridge_max_evals"] = 200
    hyperopt_param["lasso_max_evals"] = 200
    hyperopt_param['svr_max_evals'] = 200
    hyperopt_param['dnn_max_evals'] = 200
    hyperopt_param['libfm_max_evals'] = 200
    hyperopt_param['rgf_max_evals'] = 200

########################################
## Parameter Space for XGBoost models ##
########################################

## classification with linear booster
param_space_cls_xgb_linear = {
    'task': 'classification',
    'booster': 'gblinear',
    'objective': 'binary:logistic',
    'eta' : hp.quniform('eta', 0.01, 1, 0.01),
    'lambda' : hp.quniform('lambda', 0, 5, 0.05),
    'alpha' : hp.quniform('alpha', 0, 0.5, 0.005),
    'lambda_bias' : hp.quniform('lambda_bias', 0, 3, 0.1),
    'num_round' : hp.quniform('num_round', xgb_min_num_round, xgb_max_num_round, xgb_num_round_step),
    'nthread': xgb_nthread,
    'silent' : 1,
    'seed': xgb_random_seed,
    "max_evals": hyperopt_param["xgb_max_evals"],
}

## classification with tree booster
param_space_cls_xgb_tree = {
    'task': 'classification',
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eta': hp.quniform('eta', 0.01, 1, 0.01),
    'gamma': hp.quniform('gamma', 0, 2, 0.1),
    'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
    'max_depth': hp.quniform('max_depth', 1, 10, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1, 0.1),
    'num_round': hp.quniform('num_round', xgb_min_num_round, xgb_max_num_round, xgb_num_round_step),
    'nthread': xgb_nthread,
    'silent': 1,
    'seed': xgb_random_seed,
    "max_evals": hyperopt_param["xgb_max_evals"],
}

## integer features
int_feat = ["num_round", "n_estimators", "max_depth", "degree",
        "hidden_units", "hidden_layers", "batch_size", "nb_epoch",
        "dim", "iter", "max_leaf_forest", "num_iteration_opt", "num_tree_search", "min_pop", "opt_interval"]
####################
## All the Models ##
####################
feat_folders = []
feat_names = []
param_spaces = {}

####################################
## [feat@the1ow_1023] ##
#####################################
#############
## xgboost ##
#############
## regression with xgboost tree booster
feat_folder = "../../feat/solution/the1ow_1023"
feat_name = "[Pre@solution]_[feat@the1ow_1023]_[Model@cls_xgb_tree]"
feat_folders.append( feat_folder )
feat_names.append( feat_name )
param_spaces[feat_name] = param_space_cls_xgb_tree

## regression with xgboost linear booster
feat_folder = "../../feat/solution/the1ow_1023"
feat_name = "[Pre@solution]_[feat@the1ow_1023]_[Model@cls_xgb_linear]"
feat_folders.append( feat_folder )
feat_names.append( feat_name )
param_spaces[feat_name] = param_space_cls_xgb_linear



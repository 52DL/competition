"""
  __file__
    combine_feat_the1ow_1023.py
  __description__
    this file provides one combination of the features
  __author__
    qufu
"""

import sys
import pickle
import pandas as pd
sys.path.append("../")
from param_config import config
from gen_info import gen_info
from combine_feat_csv import combine_feat, SimpleTransform


if __name__ == "__main__":
  feat_names = [

   ## one hot feats
  ('ps_car_10_cat_oh_1', SimpleTransform()),
  ('ps_car_10_cat_oh_0', SimpleTransform()),
  ('ps_car_10_cat_oh_2', SimpleTransform()),
  ('ps_calc_04_oh_3', SimpleTransform()),
  ('ps_calc_04_oh_2', SimpleTransform()),
  ('ps_calc_04_oh_1', SimpleTransform()),
  ('ps_calc_04_oh_4', SimpleTransform()),
  ('ps_calc_04_oh_0', SimpleTransform()),
  ('ps_calc_04_oh_5', SimpleTransform()),
  ('ps_ind_04_cat_oh_1.0', SimpleTransform()),
  ('ps_ind_04_cat_oh_0.0', SimpleTransform()),
  ('ps_ind_04_cat_oh_-1.0', SimpleTransform()),
  ('ps_car_07_cat_oh_1.0', SimpleTransform()),
  ('ps_car_07_cat_oh_-1.0', SimpleTransform()),
  ('ps_car_07_cat_oh_0.0', SimpleTransform()),
  ('ps_car_02_cat_oh_1.0', SimpleTransform()),
  ('ps_car_02_cat_oh_0.0', SimpleTransform()),
  ('ps_car_02_cat_oh_-1.0', SimpleTransform()),
  ('ps_car_11_oh_2.0', SimpleTransform()),
  ('ps_car_11_oh_3.0', SimpleTransform()),
  ('ps_car_11_oh_1.0', SimpleTransform()),
  ('ps_car_11_oh_0.0', SimpleTransform()),
  ('ps_car_11_oh_-1.0', SimpleTransform()),
  ('ps_car_05_cat_oh_1.0', SimpleTransform()),
  ('ps_car_05_cat_oh_-1.0', SimpleTransform()),
  ('ps_car_05_cat_oh_0.0', SimpleTransform()),
  ('ps_ind_14_oh_0', SimpleTransform()),
  ('ps_ind_14_oh_1', SimpleTransform()),
  ('ps_ind_14_oh_2', SimpleTransform()),
  ('ps_ind_14_oh_3', SimpleTransform()),
  ('ps_ind_14_oh_4', SimpleTransform()),
  ('ps_car_03_cat_oh_-1.0', SimpleTransform()),
  ('ps_car_03_cat_oh_0.0', SimpleTransform()),
  ('ps_car_03_cat_oh_1.0', SimpleTransform()),
  ('ps_car_09_cat_oh_0.0', SimpleTransform()),
  ('ps_car_09_cat_oh_2.0', SimpleTransform()),
  ('ps_car_09_cat_oh_3.0', SimpleTransform()),
  ('ps_car_09_cat_oh_1.0', SimpleTransform()),
  ('ps_car_09_cat_oh_-1.0', SimpleTransform()),
  ('ps_car_09_cat_oh_4.0', SimpleTransform()),
  ('ps_ind_02_cat_oh_2.0', SimpleTransform()),
  ('ps_ind_02_cat_oh_1.0', SimpleTransform()),
  ('ps_ind_02_cat_oh_4.0', SimpleTransform()),
  ('ps_ind_02_cat_oh_3.0', SimpleTransform()),
  ('ps_ind_02_cat_oh_-1.0', SimpleTransform()),

  ## math feats
  ('ps_ind_01math_median_range', SimpleTransform()),
  ('ps_ind_01math_mean_range', SimpleTransform()),
  ('ps_ind_02_catmath_median_range', SimpleTransform()),
  ('ps_ind_02_catmath_mean_range', SimpleTransform()),
  ('ps_ind_03math_median_range', SimpleTransform()),
  ('ps_ind_03math_mean_range', SimpleTransform()),
  ('ps_ind_04_catmath_median_range', SimpleTransform()),
  ('ps_ind_04_catmath_mean_range', SimpleTransform()),
  ('ps_ind_05_catmath_median_range', SimpleTransform()),
  ('ps_ind_05_catmath_mean_range', SimpleTransform()),
  ('ps_ind_14math_median_range', SimpleTransform()),
  ('ps_ind_14math_mean_range', SimpleTransform()),
  ('ps_ind_15math_median_range', SimpleTransform()),
  ('ps_ind_15math_mean_range', SimpleTransform()),
  ('ps_reg_01math_median_range', SimpleTransform()),
  ('ps_reg_01math_mean_range', SimpleTransform()),
  ('ps_reg_02math_median_range', SimpleTransform()),
  ('ps_reg_02math_mean_range', SimpleTransform()),
  ('ps_reg_03math_median_range', SimpleTransform()),
  ('ps_reg_03math_mean_range', SimpleTransform()),
  ('ps_car_01_catmath_median_range', SimpleTransform()),
  ('ps_car_01_catmath_mean_range', SimpleTransform()),
  ('ps_car_02_catmath_median_range', SimpleTransform()),
  ('ps_car_02_catmath_mean_range', SimpleTransform()),
  ('ps_car_03_catmath_median_range', SimpleTransform()),
  ('ps_car_03_catmath_mean_range', SimpleTransform()),
  ('ps_car_04_catmath_median_range', SimpleTransform()),
  ('ps_car_04_catmath_mean_range', SimpleTransform()),
  ('ps_car_05_catmath_median_range', SimpleTransform()),
  ('ps_car_05_catmath_mean_range', SimpleTransform()),
  ('ps_car_06_catmath_median_range', SimpleTransform()),
  ('ps_car_06_catmath_mean_range', SimpleTransform()),
  ('ps_car_07_catmath_median_range', SimpleTransform()),
  ('ps_car_07_catmath_mean_range', SimpleTransform()),
  ('ps_car_08_catmath_median_range', SimpleTransform()),
  ('ps_car_08_catmath_mean_range', SimpleTransform()),
  ('ps_car_09_catmath_median_range', SimpleTransform()),
  ('ps_car_09_catmath_mean_range', SimpleTransform()),
  ('ps_car_10_catmath_median_range', SimpleTransform()),
  ('ps_car_10_catmath_mean_range', SimpleTransform()),
  ('ps_car_11_catmath_median_range', SimpleTransform()),
  ('ps_car_11_catmath_mean_range', SimpleTransform()),
  ('ps_car_11math_median_range', SimpleTransform()),
  ('ps_car_11math_mean_range', SimpleTransform()),
  ('ps_car_12math_median_range', SimpleTransform()),
  ('ps_car_12math_mean_range', SimpleTransform()),
  ('ps_car_13math_median_range', SimpleTransform()),
  ('ps_car_13math_mean_range', SimpleTransform()),
  ('ps_car_14math_median_range', SimpleTransform()),
  ('ps_car_14math_mean_range', SimpleTransform()),
  ('ps_car_15math_median_range', SimpleTransform()),
  ('ps_car_15math_mean_range', SimpleTransform()),
  ('ps_calc_01math_median_range', SimpleTransform()),
  ('ps_calc_01math_mean_range', SimpleTransform()),
  ('ps_calc_02math_median_range', SimpleTransform()),
  ('ps_calc_02math_mean_range', SimpleTransform()),
  ('ps_calc_03math_median_range', SimpleTransform()),
  ('ps_calc_03math_mean_range', SimpleTransform()),
  ('ps_calc_04math_median_range', SimpleTransform()),
  ('ps_calc_04math_mean_range', SimpleTransform()),
  ('ps_calc_05math_median_range', SimpleTransform()),
  ('ps_calc_05math_mean_range', SimpleTransform()),
  ('ps_calc_06math_median_range', SimpleTransform()),
  ('ps_calc_06math_mean_range', SimpleTransform()),
  ('ps_calc_07math_median_range', SimpleTransform()),
  ('ps_calc_07math_mean_range', SimpleTransform()),
  ('ps_calc_08math_median_range', SimpleTransform()),
  ('ps_calc_08math_mean_range', SimpleTransform()),
  ('ps_calc_09math_median_range', SimpleTransform()),
  ('ps_calc_09math_mean_range', SimpleTransform()),
  ('ps_calc_10math_median_range', SimpleTransform()),
  ('ps_calc_10math_mean_range', SimpleTransform()),
  ('ps_calc_11math_median_range', SimpleTransform()),
  ('ps_calc_11math_mean_range', SimpleTransform()),
  ('ps_calc_12math_median_range', SimpleTransform()),
  ('ps_calc_12math_mean_range', SimpleTransform()),
  ('ps_calc_13math_median_range', SimpleTransform()),
  ('ps_calc_13math_mean_range', SimpleTransform()),
  ('ps_calc_14math_median_range', SimpleTransform()),
  ('ps_calc_14math_mean_range', SimpleTransform()),
  
  ## both feats
  ('both_ps_car_13_x_ps_reg_03', SimpleTransform()),

  ## nan feats
  ('nan_sum', SimpleTransform()),

  ]

  ##loading data##
  dfTrain = pd.read_csv(config.original_train_data_path)
  dfTest = pd.read_csv(config.original_test_data_path)
  with open("%s/stratifiedKFold.%s.pkl" % (config.data_folder, config.stratified_label), "rb") as f:
    skf = pickle.load(f)
  gen_info(feat_path_name="the1ow_1023")
  combine_feat(feat_names, feat_path_name="the1ow_1023", dfTrain=dfTrain, dfTest=dfTest, skf=skf)

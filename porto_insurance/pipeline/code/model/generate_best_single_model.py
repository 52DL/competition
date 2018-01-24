"""
  __file__
    generate_best_single_model.py
  __description__
    this file trains the best single model
  __author__
    qufu
"""

import os

feat_names = [
       "[Pre@solution]_[feat@the1ow_1023]_[Model@cls_xgb_tree]", 
  ]
 
for feat_name in feat_names:
  cmd = "python ./train_model.py %s" % feat_name
  os.system( cmd )

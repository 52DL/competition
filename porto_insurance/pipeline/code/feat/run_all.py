"""
  __file__
    run_all.py
  __description__
    this file get the features in one shot
  __author__
    qufu
"""

import os

cmd = "python ./preprocess.py"
os.system(cmd)

cmd = "python ./gen_kfold.py"
os.system(cmd)

cmd = "python ./genFeat_oh_feat.py"
os.system(cmd)

cmd = "python ./genFeat_math_feat.py"
os.system(cmd)

cmd = "python ./genFeat_both_feat.py"
os.system(cmd)

cmd = "python ./genFeat_nan_feat.py"
os.system(cmd)


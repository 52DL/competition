# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

# All submission files were downloaded from different public kernels
# See the description to see the source of each submission file
submissions_path = "../subm/ensemble"
all_files = ['lr9',
        'nb9',
        #'qda',
        'lda9']

all_files_bak = ['andy_rgf_submit.csv',
        'andy_xgb_submit.csv',
        'andy_lgb_submit.csv',
        'the1ow_xgb_submit.csv',
        'the1ow_rgf_submit.csv',
        'eddy_nnad5_submit.csv',  #NN_eddy_submit.csv',
        'scirp_gp_submit.csv',
        'daia_xgb_submit.csv',
        'the1ow_lgb_submit.csv']

all_files_bak = ['andy_rgf_submit.csv',
        'andy_xgb_all_submit.csv',
        'andy_lgb_all_submit.csv',
        'the1ow_xgb_all_submit.csv',
        'the1ow_rgf_submit.csv',
        'eddy_nnad5_submit.csv',  #NN_eddy_submit.csv',
        'the1ow_lgb_all_submit.csv']
# Read and concatenate submissions
outs = [pd.read_csv(os.path.join(submissions_path, f+'_submit.csv'), index_col=0)\
                for f in all_files]
concat_df = pd.concat(outs, axis=1)
cols = list(map(lambda x: "target_" + str(x), range(len(concat_df.columns))))
concat_df.columns = cols

# Apply ranking, normalization and averaging
concat_df["target"] = (concat_df.rank() / concat_df.shape[0]).mean(axis=1)
concat_df.drop(cols, axis=1, inplace=True)
# Write the output
concat_df.to_csv("../subm/average/1129avarage-best9.csv",float_format='%.6f')


# Read and concatenate submissions
outs = [pd.read_csv(os.path.join(submissions_path, f+'_valid.csv'), index_col=0)\
                for f in all_files]
concat_df = pd.concat(outs, axis=1)
cols = list(map(lambda x: "target_" + str(x), range(len(concat_df.columns))))
concat_df.columns = cols

# Apply ranking, normalization and averaging
concat_df["target"] = (concat_df.rank() / concat_df.shape[0]).mean(axis=1)
concat_df.drop(cols, axis=1, inplace=True)

def eval_gini(y_true, y_prob):
    """
    Original author CPMP : https://www.kaggle.com/cpmpml
    In kernel : https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini
dfTrain = pd.read_csv("../../../input/train.csv")
dfTest = pd.read_csv("../../../input/test.csv")
numTrain = dfTrain.shape[0]
numTest = dfTest.shape[0]
y = dfTrain["target"]

print(eval_gini(y,concat_df["target"]))

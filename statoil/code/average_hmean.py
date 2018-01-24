# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from scipy.stats.mstats import hmean
 

# All submission files were downloaded from different public kernels
# See the description to see the source of each submission file
submissions_path = "../subm/"

all_files = ['andy_rgf',
        'andy_xgb',
        'andy_lgb',
        'the1ow_xgb',
        'the1ow_rgf',
        'eddy_nnad5',  #NN_eddy_submit.csv',
        'scirp_gp',
        'daia_xgb',
        'the1ow_lgb']

dfTrain = pd.read_csv("../../../input/train.csv")
dfTest = pd.read_csv("../../../input/test.csv")
y = dfTrain["target"]

# Read and concatenate submissions
def get_rank(df):
    rank = df.argsort().argsort()
    rank = 1.0*rank/np.max(rank)
    return rank

outs = [pd.read_csv(os.path.join(submissions_path, f+'_submit.csv'), index_col=0)\
                for f in all_files]
print(len(outs))
for i in range(len(outs)):
    probs = get_rank(outs[i]['target'])
    almost_zero = 1e-12
    probs[probs<almost_zero] = almost_zero
    outs[i]['target'] = probs

print(len(outs))
concat_df = pd.concat(outs)
print(concat_df.head())
print(concat_df.shape)
#cols = list(map(lambda x: "target_" + str(x), range(len(concat_df.columns))))
#print(cols)
#concat_df.columns = cols
adf = concat_df['target']
sadf = pd.DataFrame()
sadf['target'] = adf.groupby(level=0).apply(hmean)
#concat_df.drop(cols, axis=1, inplace=True)
# Write the output
sadf["id"] = dfTest["id"].values
print(sadf.head())
print(sadf.shape)
sadf.to_csv("../subm/average/1129avarage-hmean.csv",float_format='%.6f')

del concat_df
del outs
###################################################################################


outs = [pd.read_csv(os.path.join(submissions_path, f+'_valid.csv'), index_col=0)\
                for f in all_files]
for out in outs:
    probs = get_rank(out['target'])
    print(probs.shape)
    almost_zero = 1e-12
    probs[probs<almost_zero] = almost_zero
    out['target'] = probs

concat_df = pd.concat(outs)
print(concat_df.head())
print(concat_df.shape)
#cols = list(map(lambda x: "target_" + str(x), range(len(concat_df.columns))))
#print(cols)
#concat_df.columns = cols
# Apply ranking, normalization and averaging
a = concat_df['target']
preds = a.groupby(level=0).apply(hmean)


def eval_gini(y_true, y_prob):
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

print(eval_gini(y,preds))

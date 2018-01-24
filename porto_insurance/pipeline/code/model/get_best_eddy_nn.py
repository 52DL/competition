'''
    This script provides code for training a neural network with entity embeddings
    of the 'cat' variables. For more details on entity embedding, see:
    https://github.com/entron/entity-embedding-rossmann
    
    8-Fold training with 3 averaged runs per fold. Results may improve with more folds & runs.
'''

import numpy as np
import pandas as pd
from numba import jit
import time
import gc
import subprocess
import glob
#random seeds for stochastic parts of neural network 
np.random.seed(0)
from tensorflow import set_random_seed
set_random_seed(15)

from keras.models import load_model, Model
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Concatenate, Reshape, Dropout, Merge
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback

from sklearn.model_selection import KFold
from utils_csv import *

FEAT_FOLDER = '../../feat/solution/eddy-1126'
use_int = False

#Data loading & preprocessing
start = time.clock()
tt_data = get_feat_data(FEAT_FOLDER)
print("loading data used time: ", (time.clock()-start))

X_tr = tt_data['X1']
X_te = tt_data['X2']
y_tr = tt_data['Y1']['target']
id_train = tt_data['Y1']['id'].values
id_test = tt_data['Y2']['id'].values

col_vals_dict = {c: list(X_tr[c].unique()) for c in X_tr.columns if c.endswith('_cat')}

embed_cols = []
for c in col_vals_dict:
    if len(col_vals_dict[c])>2:
        embed_cols.append(c)
if use_int:
    embed_cols.append('ps_ind_01')
    embed_cols.append('ps_ind_03')
    embed_cols.append('ps_ind_14')
    embed_cols.append('ps_ind_15')
    embed_cols.append('ps_car_11')

#converting data to list format to match the network structure
def preproc(X_train, X_val, X_test):

    input_list_train = []
    input_list_val = []
    input_list_test = []
    
    #the cols to be embedded: rescaling to range [0, # values)
    for c in embed_cols:
        raw_vals = np.unique(X_train[c])
        val_map = {}
        for i in range(len(raw_vals)):
            val_map[raw_vals[i]] = i       
        input_list_train.append(X_train[c].map(val_map).values)
        input_list_val.append(X_val[c].map(val_map).fillna(0).values)
        input_list_test.append(X_test[c].map(val_map).fillna(0).values)
     
    #the rest of the columns
    other_cols = [c for c in X_train.columns if (not c in embed_cols)]
    input_list_train.append(X_train[other_cols].values)
    input_list_val.append(X_val[other_cols].values)
    input_list_test.append(X_test[other_cols].values)
    
    return input_list_train, input_list_val, input_list_test    

#gini scoring function from kernel at: 
#https://www.kaggle.com/tezdhar/faster-gini-calculation
# from CPMP's kernel https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
@jit
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
 
class roc_auc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_epoch_end(self, epoch, logs={}):
        #y_pred = self.model.predict_proba(self.x, verbose=0)
        #roc = eval_gini(self.y, y_pred[:,0])
        #logs['norm_gini'] = roc

        y_pred_val = self.model.predict(self.x_val, verbose=0)
        roc_val = eval_gini(self.y_val, y_pred_val[:,0])
        logs['norm_gini_val'] = roc_val

        print('\rnorm_gini:  - norm_gini_val: %s' % (roc_val))
        return

#network training
K = 5
runs_per_fold = 1
n_epochs = 100
patience = 9
y_valid_pred = 0*y_tr
y_test_pred = 0

kfold = KFold(n_splits = K, 
                            random_state = 1, 
                            shuffle = True)    

for i, (train_index, test_index) in enumerate(kfold.split(X_tr)):

    X_train, X_valid = X_tr.loc[train_index].copy(), X_tr.loc[test_index].copy()
    y_train, y_valid = y_tr[train_index], y_tr[test_index]
    
    X_test = X_te.copy()
    print('\n')
    print("Fold %s begins: " % i) 
    #upsampling adapted from kernel: 
    #https://www.kaggle.com/ogrellier/xgb-classifier-upsampling-lb-0-283
    pos = (pd.Series(y_train == 1))
    
    # Add positive examples
    X_train = pd.concat([X_train, X_train.loc[pos]], axis=0)
    y_train = pd.concat([y_train, y_train.loc[pos]], axis=0)
    
    # Shuffle data
    idx = np.arange(len(X_train))
    np.random.shuffle(idx)
    X_train = X_train.iloc[idx]
    y_train = y_train.iloc[idx]
    
    #preprocessing
    proc_X_train, proc_X_valid, proc_X_test = preproc(X_train, X_valid, X_test)
    print('\n')
    #print(len(proc_X_train))
    #track oof prediction for cv scores
    val_preds = 0
    
    cb = [roc_auc_callback(training_data=(proc_X_train, y_train.values),
                          validation_data=(proc_X_valid, y_valid.values)),
            EarlyStopping(monitor='norm_gini_val', patience=patience, mode='max', verbose=1),
            ModelCheckpoint(
                    './keras/keras-5fold-v1-fold-' + str('%02d' % (i + 1)) + '.check',
                    monitor='norm_gini_val', mode='max', # mode must be set to max or Keras will be confused
                    save_best_only=True,
                    verbose=1)
          ]
#    for j in range(runs_per_fold):
 #   print("\nrun %s begin:" % j) 
    if use_int:
        NN = build_embedding_network_func_int()
        #NN = build_embedding_network_funcbn_int()
    else:
        NN = build_embedding_network_func()
        #NN = build_embedding_network_funcbn()
    NN.fit(proc_X_train, y_train.values, epochs=n_epochs, batch_size=4096, verbose=2, callbacks=cb)
    del NN
    NN_best = load_model('./keras/keras-5fold-v1-fold-' + str('%02d' % (i + 1)) + '.check')
    pred = NN_best.predict(proc_X_valid)[:,0] / runs_per_fold
    probs = NN_best.predict(proc_X_test)[:,0] / runs_per_fold
        
    # Save validation predictions for this fold
    print( "  Gini = \n" )
    #y_valid_pred.iloc[test_index] = (np.exp(pred) - 1.0).clip(0,1)
    y_valid_pred.iloc[test_index] = pred

    # Accumulate test set predictions
    #y_test_pred += (np.exp(test_pred) - 1.0).clip(0,1)
    almost_zero = 1e-12
    almost_one = 1 - almost_zero  # To avoid division by zero
    probs[probs>almost_one] = almost_one
    probs[probs<almost_zero] = almost_zero
    y_test_pred += np.log(probs/(1-probs))

    del X_test, X_train, X_valid

y_test_pred /= K  # Average test set predictions
y_test_pred =  1  /  ( 1 + np.exp(-y_test_pred) )

print( "\nGini for full training set:" ,eval_gini(y_tr, y_valid_pred))

# Save validation predictions for stacking/ensembling
val = pd.DataFrame()
val['id'] = id_train
val['target'] = y_valid_pred.values
val.to_csv('%s/eddy_nnad5_valid.csv' % config.sub_folder, float_format='%.6f', index=False)

# Create submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_test_pred
sub.to_csv('%s/eddy_nnad5_submit.csv' % config.sub_folder, float_format='%.6f', index=False)
"""    

    full_val_preds[test_index] += val_preds
        
    cv_gini = eval_gini(y_valid.values, val_preds)
    cv_ginis.append(cv_gini)
    print ('\nFold %i prediction cv gini: %.5f\n' %(i,cv_gini))
    
print('Mean out of fold gini: %.5f' % np.mean(cv_ginis))
print('Full validation gini: %.5f' % eval_gini(y_train.values, full_val_preds))

y_pred_final = np.mean(y_preds, axis=1)

df_sub = pd.DataFrame({'id' : df_test.id, 
                       'target' : y_pred_final},
                       columns = ['id','target'])
df_sub.to_csv('NN_submit.csv', index=False)

pd.DataFrame(full_val_preds).to_csv('NN_valid.csv',index=False)
"""    

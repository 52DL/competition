MAX_EPOCHS = 10
OPTIMIZE_XGB_ROUNDS = True
MAX_DEPTH = 6
LEARNING_RATE = 0.024
XGB_EARLY_STOPPING_ROUNDS = 100
FEAT_FOLDER = '../../feat/solution/the1ow-1031'


import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from keras.wrappers.scikit_learn import KerasClassifier

from numba import jit
import time
import gc
import subprocess
import glob
from utils_csv import *

# Compute gini

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

# Funcitons from olivier's kernel
# https://www.kaggle.com/ogrellier/xgb-classifier-upsampling-lb-0-283

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = -eval_gini(labels, preds)
    return [('gini', gini_score)]

def gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = eval_gini(y,preds) / eval_gini(y,y)
    return 'gini', score, True

class roc_auc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict_proba(self.x, verbose=0)
        roc = eval_gini(self.y, y_pred[:,0])
        logs['norm_gini'] = roc

        y_pred_val = self.model.predict_proba(self.x_val, verbose=0)
        roc_val = eval_gini(self.y_val, y_pred_val[:,0])
        logs['norm_gini_val'] = roc_val

        print('\n\rnorm_gini: %s - norm_gini_val: %s' % (roc,roc_val))
        return

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod(
            (datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' %
              (thour, tmin, round(tsec, 2)))

def scale_data(X, scaler=None):
    if not scaler:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

start = time.clock()
tt_data = get_feat_data(FEAT_FOLDER)
print("loading data used time: ", (time.clock()-start))

X = tt_data['X1']
test_df = tt_data['X2']
y = tt_data['Y1']['target'].values
id_train = tt_data['Y1']['id'].values
id_test = tt_data['Y2']['id'].values
X,_ = scale_data(X)
test_df,_ = scale_data(test_df)

#f_cats = [f for f in X.columns if "_cat" in f]

y_valid_pred = 0*y
y_test_pred = 0

# Set up folds
K = 2
kf = KFold(n_splits = K, random_state = 1, shuffle = True)
np.random.seed(0)

patience = 10
batchsize = 128
    
def baseline_model():
    model = Sequential()
    model.add(
	Dense(
	    200,
	    input_dim=X_train.shape[1],
	    kernel_initializer='glorot_normal',
	    ))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(100, kernel_initializer='glorot_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(50, kernel_initializer='glorot_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Dense(25, kernel_initializer='glorot_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(optimizer='adam', metrics = ['accuracy'], loss='binary_crossentropy')

    return model


# Run CV

for i, (train_index, test_index) in enumerate(kf.split(X)):
    
    # Create data for this fold
    y_train, y_valid = y[train_index], y[test_index]
    X_train, X_valid = X[train_index,:], X[test_index,:]
    X_test = test_df.copy()
    print( "\nFold ", i)

    callbacks = [
            roc_auc_callback(training_data=(X_train, y_train),validation_data=(X_valid, y_valid)),  # call this before EarlyStopping
            EarlyStopping(monitor='norm_gini_val', patience=patience, mode='max', verbose=1),    
            ModelCheckpoint(
                    './keras/keras-5fold-run-01-v1-fold-' + str('%02d' % (i + 1)) + '.check',
                    monitor='norm_gini_val', mode='max', # mode must be set to max or Keras will be confused
                    save_best_only=True,
                    verbose=1)
            ]

    # Run model for this fold
    nnet = KerasClassifier(
	build_fn=baseline_model,
	epochs=MAX_EPOCHS,
	batch_size=batchsize,
	validation_data=(X_valid, y_valid),
	verbose=2,
	shuffle=True,
	callbacks=callbacks)

    fit_model = nnet.fit(X_train, y_train)

    del nnet
    fit_model = load_model('./keras/keras-5fold-run-01-v1-fold-' + str('%02d' % (i + 1)) + '.check')
    pred = fit_model.predict(X_valid, verbose=0)[:,0]
    probs = fit_model.predict(X_test, verbose=0)[:,0]

    # Save validation predictions for this fold
    print( "  Gini = ", eval_gini(y_valid, pred) )
    #y_valid_pred.iloc[test_index] = (np.exp(pred) - 1.0).clip(0,1)
    y_valid_pred[test_index] = pred
    
    # Accumulate test set predictions
    #y_test_pred += (np.exp(test_pred) - 1.0).clip(0,1)
    almost_zero = 1e-12
    almost_one = 1 - almost_zero  # To avoid division by zero
    probs[probs>almost_one] = almost_one
    probs[probs<almost_zero] = almost_zero
    y_test_pred += np.log(probs/(1-probs))
    
    del X_test, X_train, X_valid, y_train
    
y_test_pred /= K  # Average test set predictions
y_test_pred =  1  /  ( 1 + np.exp(-y_test_pred) )

print( "\nGini for full training set:" ,eval_gini(y, y_valid_pred))
"""
# Save validation predictions for stacking/ensembling
val = pd.DataFrame()
val['id'] = id_train
val['target'] = y_valid_pred.values
val.to_csv('%s/the1ow_lgb_valid.csv' % config.sub_folder, float_format='%.6f', index=False)

# Create submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_test_pred
sub.to_csv('%s/the1ow_lgb_submit.csv' % config.sub_folder, float_format='%.6f', index=False)

"""
# Save validation predictions for stacking/ensembling

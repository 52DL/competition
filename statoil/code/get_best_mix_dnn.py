
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from os.path import join as opj
import time
import os
import gc
import glob
import subprocess
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#Import Keras.
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation, concatenate
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
#from keras.layers.merge import Concatenate
from keras.models import Model
from keras import initializers
from keras.optimizers import Adam
from keras import optimizers
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.applications.vgg16 import VGG16
from keras import backend as K
model_type = 'mix+dnn'
def test_id(a,b):
    for i,j in zip(a,b):
        if i!=j:
            print('asdf')
#Load data
train_org = pd.read_json('../input/train.json')
id_train = train_org['id']
test_org = pd.read_json('../input/test.json' )
id_test = test_org['id']

y = train_org['is_iceberg'].values
#del train, test

picf_train = pd.read_csv('../feat/picf_train_opted.csv')
#print('picf:',picf_train.shape)
picf_test = pd.read_csv('../feat/picf_test_opted.csv')
sbmr_train = pd.read_csv('../feat/sbmr_train.csv')
#print('sbmr:',sbmr_train.shape)
sbmr_test = pd.read_csv('../feat/sbmr_test.csv')
pca_train = pd.read_csv('../feat/pca_train.csv')
#print('pca:',pca_train.shape)
pca_test = pd.read_csv('../feat/pca_test.csv')
hog_train = pd.read_csv('../feat/hog_train.csv')
#print('hog:',hog_train.shape)
hog_test = pd.read_csv('../feat/hog_test.csv')
fft_train = pd.read_csv('../feat/fft_train.csv')
#print('fft:',fft_train.shape)
fft_test = pd.read_csv('../feat/fft_test.csv')
ir20_train = pd.read_csv('../feat/ir20_train.csv')
#print('ir20:',ir20_train.shape)
ir20_test = pd.read_csv('../feat/ir20_test.csv')
train = pd.merge(sbmr_train, picf_train, how='left',on='id')
#print('sbmr+picf',train.shape)
train = pd.merge(train, pca_train, how='left',on='id')
#print('+pca',train.shape)
train = pd.merge(train, hog_train, how='left',on='id')
#print('+hog',train.shape)
train = pd.merge(train, fft_train, how='left',on='id')
#print('+fft',train.shape)
train = pd.merge(train, ir20_train, how='left',on='id')
#print('+ir20',train.shape)
#train = fft_train
#train = picf_train
#train = pca_train

test_id(id_train, train['id'])

#train['inc_angle'] = pca_train['inc_angle']
train['inc_angle'] = train_org['inc_angle'].replace('na', -1).astype(float)
#print(train.info())
print(train.shape)

test = pd.merge(sbmr_test, picf_test,how='left',on='id')
#print('sbmr+picf',test.shape)
test = pd.merge(test, pca_test, how='left',on='id')
#print('+pca',test.shape)
test = pd.merge(test, hog_test, how='left',on='id')
#print('+hog',test.shape)
test = pd.merge(test, fft_test, how='left',on='id')
#print('+fft',test.shape)
test = pd.merge(test, ir20_test, how='left',on='id')
#print('+ir20',test.shape)
#train = pd.merge(train, hog_train, how='left',on='id')
#test = pd.merge(test, hog_test,how='left',on='id')
test_id(id_test, test['id'])
#test = fft_test
#test = picf_test
#test = pca_test
#test['inc_angle'] = pca_test['inc_angle']
test['inc_angle'] = test_org['inc_angle'].replace('na', 0).astype(float)
print(test.shape)

col = [c for c in train.columns if c not in ['id']]
#print(col)
X_train = train.loc[:,col]
#print(X.info())
print(X_train.shape)
X_test = test.loc[:,col]
print(X_test.shape)
from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)
#X_train = pd.DataFrame(scaling.transform(X_train),columns=X_train.columns)
#X_test = pd.DataFrame(scaling.transform(X_test),columns=X_test.columns)

############################################
#define our model:(we can change the model)
############################################
def getModel():
    model = Sequential()

    model.add(Dense(X_train.shape[1],input_shape=[X_train.shape[1]]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))


    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    mypotim=Adam(lr=0.0005, decay=1e-6)
    model.compile(loss='binary_crossentropy', optimizer = mypotim, metrics=['accuracy'])
    return model


def get_callbacks(i,j,filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    #lrreduc = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience/2, verbose=1, epsilon=1e-4, mode='min')
    #log = TensorBoard(log_dir='../log/'+model_type+'/'+str(i)+str(j))
    #return [es, msave, lrreduc, log]
    #return [es, msave, log]
    return [es, msave]

############################################
# Set up folds: (we can't change the fold stucture!!!)
############################################
k = 5 #don't chage
kf = KFold(n_splits = k, random_state = 1108, shuffle = True) #don't change

############################################
# begin to train: (we can change super paramters)
############################################
y_valid_pred = 0.0*y
y_test_pred = 0.0

RUNS = 1
OPTIMIZE_ROUNDS = True
MAX_ROUNDS = 150
BATCH_SIZE = 32
PATIENCE = 100
VERBOSE = 0
np.random.seed(1108)

for i, (train_index, test_index) in enumerate(kf.split(X_train)):

    # Create data for this fold
    y_train_cv, y_valid_cv = y[train_index], y[test_index]
    X_train_cv, X_valid_cv = X_train[train_index,:], X_train[test_index,:]
    print( "\nFold ", i)

    pred = 0.0*y_valid_cv
    for j in range(RUNS):
        print( "\n   Run %s/%s " % (j+1,RUNS))
        # Run model for this fold
       	gmodel=getModel()
        file_path = ".model_weights/hdf5"+str(i)+str(j)
        if OPTIMIZE_ROUNDS:
            gmodel.fit(
                      X_train_cv,
                      y_train_cv,
                      batch_size=BATCH_SIZE,
                      epochs=MAX_ROUNDS,
                      verbose=VERBOSE,
                      callbacks=get_callbacks(i,j,filepath=file_path, patience=PATIENCE),
                      validation_data=(X_valid_cv, y_valid_cv), 
		      shuffle=True,
                      )
            gmodel.load_weights(filepath=file_path)
            score = gmodel.evaluate(X_valid_cv, y_valid_cv, verbose=VERBOSE)
            print('      Test loss:', score[0])
            print('      Test accuracy:', score[1])

            pred += gmodel.predict(X_valid_cv).reshape(pred.shape[0])
            y_test_pred += gmodel.predict(X_test).reshape(X_test.shape[0])
        else:
            gmodel.fit(X_train_cv, y_train_cv,
                      batch_size=BATCH_SIZE,
                      epochs=MAX_ROUNDS,
                      verbose=VERBOSE,
                                 )
            score = gmodel.evaluate(X_valid_cv, y_valid_cv, verbose=VERBOSE)
            print('      Test loss:', score[0])
            print('      Test accuracy:', score[1])

            pred += gmodel.predict(X_valid_cv).reshape(pred.shape[0])
            y_test_pred += gmodel.predict(X_test).reshape(X_test.shape[0])
        K.clear_session()
    # Save validation predictions for this fold
    pred = pred / RUNS
    print( "     fold cv = ", log_loss(y_valid_cv, pred) )
    y_valid_pred[test_index] = pred
    del X_train_cv, X_valid_cv

print( "\n   full cv = " ,log_loss(y, y_valid_pred))

y_test_pred /= (k * RUNS)

# Save validation predictions for stacking/ensembling
val = pd.DataFrame()
val['id'] = id_train
val['is_iceberg'] = y_valid_pred
val.to_csv('../subm/%s_valid_test.csv'%model_type, float_format='%.6f', index=False)

# Create submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['is_iceberg'] = y_test_pred
sub.to_csv('../subm/%s_submit_test.csv'%model_type, float_format='%.6f', index=False)

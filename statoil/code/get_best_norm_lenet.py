
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
model_type = 'norm+lenet'

#Load the data.
train = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")
y=train['is_iceberg']
id_train = train['id']
id_test = test['id']
test['inc_angle']=pd.to_numeric(test['inc_angle'], errors='coerce')
train['inc_angle']=pd.to_numeric(train['inc_angle'], errors='coerce')#We have only 133 NAs.
train['inc_angle']=train['inc_angle'].fillna(method='pad')
X_angle=train['inc_angle']
X_test_angle=test['inc_angle']

############################################
#Generate the training data: (we can make other data)
############################################
def transform (df):
    images = []
    for i, row in df.iterrows():
        band_1 = np.array(row['band_1']).reshape(75,75)
        band_2 = np.array(row['band_2']).reshape(75,75)
        band_3 = band_1 + band_2
        
        band_1_norm = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        band_2_norm = (band_2 - band_2. mean()) / (band_2.max() - band_2.min())
        band_3_norm = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())
        
        images.append(np.dstack((band_1_norm, band_2_norm, band_3_norm)))
    
    return np.array(images)

X_train = transform(train)
X_test = transform(test)

gen = ImageDataGenerator(horizontal_flip = True,
                         vertical_flip = True,
                         width_shift_range = 0.,
                         height_shift_range = 0.,
                         channel_shift_range=0,
                         zoom_range = 0.2,
                         rotation_range = 10)


############################################
#define our model:(we can change the model)
############################################
def getModel():
    model = Sequential()

    model.add(Conv2D(128, kernel_size=(3,3), input_shape=(75,75,3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    #model.add(Dropout(0.2))

    model.add(Conv2D(256, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(Dropout(0.2))

    model.add(Conv2D(256, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(Dropout(0.3))

    model.add(Conv2D(128, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))


    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    mypotim=Adam(lr=0.0085, decay=1e-2)
    model.compile(loss='binary_crossentropy', optimizer = mypotim, metrics=['accuracy'])
    return model
# Here is the function that merges our two generators
# We use the exact same generator with the same random seed for both the y and angle arrays
def gen_flow_for_two_inputs(X1, X2, y, batch_size):
    genX1 = gen.flow(X1,y,  batch_size=batch_size,seed=55)
    #genX2 = gen.flow(X1,X2, batch_size=batch_size,seed=55)
    while True:
            X1i = genX1.next()
            #X2i = genX2.next()
            #Assert arrays are equal - this was for peace of mind, but slows down training
            #np.testing.assert_array_equal(X1i[0],X2i[0])
            #yield [X1i[0], X2i[1]], X1i[1]
            yield X1i[0], X1i[1]


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
y_valid_pred = 0*y
y_test_pred = 0

RUNS = 5
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
    angle_train_cv, angle_valid_cv = X_angle[train_index], X_angle[test_index]
    print( "\nFold ", i)

    pred = 0*y_valid_cv
    for j in range(RUNS):
        print( "\n   Run %s/%s " % (j+1,RUNS))
        # Run model for this fold
       	gmodel=getModel()
        file_path = ".model_weights/hdf5"+str(i)+str(j)
        if OPTIMIZE_ROUNDS:
            gmodel.fit_generator(
                      gen_flow_for_two_inputs(X_train_cv, angle_train_cv, y_train_cv, BATCH_SIZE),
		      steps_per_epoch=24,
                      epochs=MAX_ROUNDS,
		      shuffle=True,
                      verbose=VERBOSE,
                      #validation_data=([X_valid_cv,angle_valid_cv], y_valid_cv),
                      validation_data=(X_valid_cv, y_valid_cv),
                      callbacks=get_callbacks(i,j,filepath=file_path, patience=PATIENCE)
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
val.to_csv('../subm/%s_valid.csv'%model_type, float_format='%.6f', index=False)

# Create submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['is_iceberg'] = y_test_pred
sub.to_csv('../subm/%s_submit.csv'%model_type, float_format='%.6f', index=False)

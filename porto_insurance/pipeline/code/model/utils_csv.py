"""
  __file__
    utils.py
  __description__
    this file provide the preloaded data
  __author__
    qufu
"""
import sys
import pandas as pd
from scipy.sparse import hstack
sys.path.append("../")
from param_config import config
from keras.models import load_model, Model
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Concatenate, Reshape, Dropout, Merge, BatchNormalization
from keras.layers.embeddings import Embedding
from keras import optimizers

def get_feat_data(feat_folder):
  
  tt_data = {}
  path = "%s/All" % (feat_folder)
  # feat path
  feat_train_path = "%s/train.feat.csv" % path
  feat_test_path = "%s/test.feat.csv" % path

  # target path
  target_train_path = "%s/train.target.csv" % path
  target_test_path = "%s/test.target.csv" % path

  #### load data
  ## load feat
  X_train = pd.read_csv(feat_train_path)
  X_test = pd.read_csv(feat_test_path)
  labels_train = pd.read_csv(target_train_path)
  labels_test = pd.read_csv(target_test_path) # to make the lable(r,) not (r,1)

 
  tt_data = { "X1":X_train, "X2":X_test, "Y1":labels_train, "Y2":labels_test}
  return tt_data

def build_embedding_network():

    models = []

    model_ps_ind_02_cat = Sequential()
    model_ps_ind_02_cat.add(Embedding(5, 3, input_length=1))
    model_ps_ind_02_cat.add(Reshape(target_shape=(3,)))
    models.append(model_ps_ind_02_cat)

    model_ps_ind_04_cat = Sequential()
    model_ps_ind_04_cat.add(Embedding(3, 2, input_length=1))
    model_ps_ind_04_cat.add(Reshape(target_shape=(2,)))
    models.append(model_ps_ind_04_cat)

    model_ps_ind_05_cat = Sequential()
    model_ps_ind_05_cat.add(Embedding(8, 5, input_length=1))
    model_ps_ind_05_cat.add(Reshape(target_shape=(5,)))
    models.append(model_ps_ind_05_cat)

    model_ps_car_01_cat = Sequential()
    model_ps_car_01_cat.add(Embedding(13, 7, input_length=1))
    model_ps_car_01_cat.add(Reshape(target_shape=(7,)))
    models.append(model_ps_car_01_cat)

    model_ps_car_02_cat = Sequential()
    model_ps_car_02_cat.add(Embedding(3, 2, input_length=1))
    model_ps_car_02_cat.add(Reshape(target_shape=(2,)))
    models.append(model_ps_car_02_cat)

    model_ps_car_03_cat = Sequential()
    model_ps_car_03_cat.add(Embedding(3, 2, input_length=1))
    model_ps_car_03_cat.add(Reshape(target_shape=(2,)))
    models.append(model_ps_car_03_cat)

    model_ps_car_04_cat = Sequential()
    model_ps_car_04_cat.add(Embedding(10, 5, input_length=1))
    model_ps_car_04_cat.add(Reshape(target_shape=(5,)))
    models.append(model_ps_car_04_cat)

    model_ps_car_05_cat = Sequential()
    model_ps_car_05_cat.add(Embedding(3, 2, input_length=1))
    model_ps_car_05_cat.add(Reshape(target_shape=(2,)))
    models.append(model_ps_car_05_cat)
    model_ps_car_06_cat = Sequential()
    model_ps_car_06_cat.add(Embedding(18, 8, input_length=1))
    model_ps_car_06_cat.add(Reshape(target_shape=(8,)))
    models.append(model_ps_car_06_cat)

    model_ps_car_07_cat = Sequential()
    model_ps_car_07_cat.add(Embedding(3, 2, input_length=1))
    model_ps_car_07_cat.add(Reshape(target_shape=(2,)))
    models.append(model_ps_car_07_cat)

    model_ps_car_09_cat = Sequential()
    model_ps_car_09_cat.add(Embedding(6, 3, input_length=1))
    model_ps_car_09_cat.add(Reshape(target_shape=(3,)))
    models.append(model_ps_car_09_cat)

    model_ps_car_10_cat = Sequential()
    model_ps_car_10_cat.add(Embedding(3, 2, input_length=1))
    model_ps_car_10_cat.add(Reshape(target_shape=(2,)))
    models.append(model_ps_car_10_cat)

    model_ps_car_11_cat = Sequential()
    model_ps_car_11_cat.add(Embedding(104, 10, input_length=1))
    model_ps_car_11_cat.add(Reshape(target_shape=(10,)))
    models.append(model_ps_car_11_cat)

    model_ps_reg_01_plus_ps_car_02_cat = Sequential()
    model_ps_reg_01_plus_ps_car_02_cat.add(Embedding(24, 8, input_length=1))
    model_ps_reg_01_plus_ps_car_02_cat.add(Reshape(target_shape=(8,)))
    models.append(model_ps_reg_01_plus_ps_car_02_cat)

    model_ps_reg_01_plus_ps_car_04_cat = Sequential()
    model_ps_reg_01_plus_ps_car_04_cat.add(Embedding(100, 10, input_length=1))
    model_ps_reg_01_plus_ps_car_04_cat.add(Reshape(target_shape=(10,)))
    models.append(model_ps_reg_01_plus_ps_car_04_cat)

    model_rest = Sequential()
    model_rest.add(Dense(57, input_dim=26))
    models.append(model_rest)

    model = Sequential()
    #model.add(Concatenate(models)
    model.add(Merge(models, mode='concat'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(.35))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(.15))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dropout(.15))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model



def build_embedding_network_func():

    inputs = []
    models = []

    
    input_ps_ind_02_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_ind_02_cat)
    model_ps_ind_02_cat = Embedding(5, 3, input_length=1)(input_ps_ind_02_cat)
    model_ps_ind_02_cat = Reshape(target_shape=(3,))(model_ps_ind_02_cat)
    models.append(model_ps_ind_02_cat)

    input_ps_ind_04_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_ind_04_cat)
    model_ps_ind_04_cat = Embedding(3, 2, input_length=1)(input_ps_ind_04_cat)
    model_ps_ind_04_cat = Reshape(target_shape=(2,))(model_ps_ind_04_cat)
    models.append(model_ps_ind_04_cat)

    input_ps_ind_05_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_ind_05_cat)
    model_ps_ind_05_cat = Embedding(8, 5, input_length=1)(input_ps_ind_05_cat)
    model_ps_ind_05_cat = Reshape(target_shape=(5,))(model_ps_ind_05_cat)
    models.append(model_ps_ind_05_cat)

    input_ps_car_01_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_01_cat)
    model_ps_car_01_cat = Embedding(13, 7, input_length=1)(input_ps_car_01_cat)
    model_ps_car_01_cat = Reshape(target_shape=(7,))(model_ps_car_01_cat)
    models.append(model_ps_car_01_cat)

    input_ps_car_02_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_02_cat)
    model_ps_car_02_cat = Embedding(3, 2, input_length=1)(input_ps_car_02_cat)
    model_ps_car_02_cat = Reshape(target_shape=(2,))(model_ps_car_02_cat)
    models.append(model_ps_car_02_cat)

    input_ps_car_03_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_03_cat)
    model_ps_car_03_cat = Embedding(3, 2, input_length=1)(input_ps_car_03_cat)
    model_ps_car_03_cat = Reshape(target_shape=(2,))(model_ps_car_03_cat)
    models.append(model_ps_car_03_cat)

    input_ps_car_04_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_04_cat)
    model_ps_car_04_cat = Embedding(10, 5, input_length=1)(input_ps_car_04_cat)
    model_ps_car_04_cat = Reshape(target_shape=(5,))(model_ps_car_04_cat)
    models.append(model_ps_car_04_cat)

    input_ps_car_05_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_05_cat)
    model_ps_car_05_cat = Embedding(3, 2, input_length=1)(input_ps_car_05_cat)
    model_ps_car_05_cat = Reshape(target_shape=(2,))(model_ps_car_05_cat)
    models.append(model_ps_car_05_cat)

    input_ps_car_06_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_06_cat)
    model_ps_car_06_cat = Embedding(18, 8, input_length=1)(input_ps_car_06_cat)
    model_ps_car_06_cat = Reshape(target_shape=(8,))(model_ps_car_06_cat)
    models.append(model_ps_car_06_cat)

    input_ps_car_07_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_07_cat)
    model_ps_car_07_cat = Embedding(3, 2, input_length=1)(input_ps_car_07_cat)
    model_ps_car_07_cat = Reshape(target_shape=(2,))(model_ps_car_07_cat)
    models.append(model_ps_car_07_cat)

    input_ps_car_09_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_09_cat)
    model_ps_car_09_cat = Embedding(6, 3, input_length=1)(input_ps_car_09_cat)
    model_ps_car_09_cat = Reshape(target_shape=(3,))(model_ps_car_09_cat)
    models.append(model_ps_car_09_cat)

    input_ps_car_10_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_10_cat)
    model_ps_car_10_cat = Embedding(3, 2, input_length=1)(input_ps_car_10_cat)
    model_ps_car_10_cat = Reshape(target_shape=(2,))(model_ps_car_10_cat)
    models.append(model_ps_car_10_cat)

    input_ps_car_11_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_11_cat)
    model_ps_car_11_cat = Embedding(104, 10, input_length=1)(input_ps_car_11_cat)
    model_ps_car_11_cat = Reshape(target_shape=(10,))(model_ps_car_11_cat)
    models.append(model_ps_car_11_cat)

    input_ps_reg_01_plus_ps_car_02_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_reg_01_plus_ps_car_02_cat)
    model_ps_reg_01_plus_ps_car_02_cat = Embedding(24, 8, input_length=1)(input_ps_reg_01_plus_ps_car_02_cat)
    model_ps_reg_01_plus_ps_car_02_cat = Reshape(target_shape=(8,))(model_ps_reg_01_plus_ps_car_02_cat)
    models.append(model_ps_reg_01_plus_ps_car_02_cat)

    input_ps_reg_01_plus_ps_car_04_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_reg_01_plus_ps_car_04_cat)
    model_ps_reg_01_plus_ps_car_04_cat = Embedding(100, 10, input_length=1)(input_ps_reg_01_plus_ps_car_04_cat)
    model_ps_reg_01_plus_ps_car_04_cat = Reshape(target_shape=(10,))(model_ps_reg_01_plus_ps_car_04_cat)
    models.append(model_ps_reg_01_plus_ps_car_04_cat)

    input_others = Input(shape=(26,), dtype='float32')
    inputs.append(input_others)
    model_others = Dense(57,activation='relu', input_dim=26)(input_others)
    models.append(model_others)


    x  = Concatenate()(models)
    #x = add(Concatenate(x = )
    x = Dense(256, activation='relu')(x)
    x = Dropout(.35)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(.15)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(.15)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    adam = optimizers.Adadelta()
    model.compile(loss='binary_crossentropy', optimizer=adam)

    return model

def build_embedding_network_func_int():

    inputs = []
    models = []

    
    input_ps_ind_02_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_ind_02_cat)
    model_ps_ind_02_cat = Embedding(5, 3, input_length=1)(input_ps_ind_02_cat)
    model_ps_ind_02_cat = Reshape(target_shape=(3,))(model_ps_ind_02_cat)
    models.append(model_ps_ind_02_cat)

    input_ps_ind_04_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_ind_04_cat)
    model_ps_ind_04_cat = Embedding(3, 2, input_length=1)(input_ps_ind_04_cat)
    model_ps_ind_04_cat = Reshape(target_shape=(2,))(model_ps_ind_04_cat)
    models.append(model_ps_ind_04_cat)

    input_ps_ind_05_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_ind_05_cat)
    model_ps_ind_05_cat = Embedding(8, 5, input_length=1)(input_ps_ind_05_cat)
    model_ps_ind_05_cat = Reshape(target_shape=(5,))(model_ps_ind_05_cat)
    models.append(model_ps_ind_05_cat)

    input_ps_car_01_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_01_cat)
    model_ps_car_01_cat = Embedding(13, 7, input_length=1)(input_ps_car_01_cat)
    model_ps_car_01_cat = Reshape(target_shape=(7,))(model_ps_car_01_cat)
    models.append(model_ps_car_01_cat)

    input_ps_car_02_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_02_cat)
    model_ps_car_02_cat = Embedding(3, 2, input_length=1)(input_ps_car_02_cat)
    model_ps_car_02_cat = Reshape(target_shape=(2,))(model_ps_car_02_cat)
    models.append(model_ps_car_02_cat)

    input_ps_car_03_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_03_cat)
    model_ps_car_03_cat = Embedding(3, 2, input_length=1)(input_ps_car_03_cat)
    model_ps_car_03_cat = Reshape(target_shape=(2,))(model_ps_car_03_cat)
    models.append(model_ps_car_03_cat)

    input_ps_car_04_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_04_cat)
    model_ps_car_04_cat = Embedding(10, 5, input_length=1)(input_ps_car_04_cat)
    model_ps_car_04_cat = Reshape(target_shape=(5,))(model_ps_car_04_cat)
    models.append(model_ps_car_04_cat)

    input_ps_car_05_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_05_cat)
    model_ps_car_05_cat = Embedding(3, 2, input_length=1)(input_ps_car_05_cat)
    model_ps_car_05_cat = Reshape(target_shape=(2,))(model_ps_car_05_cat)
    models.append(model_ps_car_05_cat)

    input_ps_car_06_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_06_cat)
    model_ps_car_06_cat = Embedding(18, 8, input_length=1)(input_ps_car_06_cat)
    model_ps_car_06_cat = Reshape(target_shape=(8,))(model_ps_car_06_cat)
    models.append(model_ps_car_06_cat)

    input_ps_car_07_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_07_cat)
    model_ps_car_07_cat = Embedding(3, 2, input_length=1)(input_ps_car_07_cat)
    model_ps_car_07_cat = Reshape(target_shape=(2,))(model_ps_car_07_cat)
    models.append(model_ps_car_07_cat)

    input_ps_car_09_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_09_cat)
    model_ps_car_09_cat = Embedding(6, 3, input_length=1)(input_ps_car_09_cat)
    model_ps_car_09_cat = Reshape(target_shape=(3,))(model_ps_car_09_cat)
    models.append(model_ps_car_09_cat)

    input_ps_car_10_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_10_cat)
    model_ps_car_10_cat = Embedding(3, 2, input_length=1)(input_ps_car_10_cat)
    model_ps_car_10_cat = Reshape(target_shape=(2,))(model_ps_car_10_cat)
    models.append(model_ps_car_10_cat)

    input_ps_car_11_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_11_cat)
    model_ps_car_11_cat = Embedding(104, 10, input_length=1)(input_ps_car_11_cat)
    model_ps_car_11_cat = Reshape(target_shape=(10,))(model_ps_car_11_cat)
    models.append(model_ps_car_11_cat)

    input_ps_reg_01_plus_ps_car_02_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_reg_01_plus_ps_car_02_cat)
    model_ps_reg_01_plus_ps_car_02_cat = Embedding(24, 8, input_length=1)(input_ps_reg_01_plus_ps_car_02_cat)
    model_ps_reg_01_plus_ps_car_02_cat = Reshape(target_shape=(8,))(model_ps_reg_01_plus_ps_car_02_cat)
    models.append(model_ps_reg_01_plus_ps_car_02_cat)

    input_ps_reg_01_plus_ps_car_04_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_reg_01_plus_ps_car_04_cat)
    model_ps_reg_01_plus_ps_car_04_cat = Embedding(100, 10, input_length=1)(input_ps_reg_01_plus_ps_car_04_cat)
    model_ps_reg_01_plus_ps_car_04_cat = Reshape(target_shape=(10,))(model_ps_reg_01_plus_ps_car_04_cat)
    models.append(model_ps_reg_01_plus_ps_car_04_cat)

    input_ps_ind_01 = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_ind_01)
    model_ps_ind_01 = Embedding(8, 4, input_length=1)(input_ps_ind_01)
    model_ps_ind_01 = Reshape(target_shape=(4,))(model_ps_ind_01)
    models.append(model_ps_ind_01)

    input_ps_ind_03 = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_ind_03)
    model_ps_ind_03 = Embedding(12, 5, input_length=1)(input_ps_ind_03)
    model_ps_ind_03 = Reshape(target_shape=(5,))(model_ps_ind_03)
    models.append(model_ps_ind_03)

    input_ps_ind_14 = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_ind_14)
    model_ps_ind_14 = Embedding(5, 3, input_length=1)(input_ps_ind_14)
    model_ps_ind_14 = Reshape(target_shape=(3,))(model_ps_ind_14)
    models.append(model_ps_ind_14)

    input_ps_ind_15 = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_ind_15)
    model_ps_ind_15 = Embedding(14, 5, input_length=1)(input_ps_ind_15)
    model_ps_ind_15 = Reshape(target_shape=(5,))(model_ps_ind_15)
    models.append(model_ps_ind_15)

    input_ps_car_11 = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_11)
    model_ps_car_11 = Embedding(5, 3, input_length=1)(input_ps_car_11)
    model_ps_car_11 = Reshape(target_shape=(3,))(model_ps_car_11)
    models.append(model_ps_ind_11)

    input_others = Input(shape=(21,), dtype='float32')
    inputs.append(input_others)
    model_others = Dense(37,activation='relu', input_dim=21)(input_others)
    models.append(model_others)


    x  = Concatenate()(models)
    #x = add(Concatenate(x = )
    x = Dense(256, activation='relu')(x)
    x = Dropout(.35)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(.15)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(.15)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    adam = optimizers.Adadelta()
    model.compile(loss='binary_crossentropy', optimizer=adam)

    return model

def build_embedding_network_funcbn():

    inputs = []
    models = []

    
    input_ps_ind_02_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_ind_02_cat)
    model_ps_ind_02_cat = Embedding(5, 3, input_length=1)(input_ps_ind_02_cat)
    model_ps_ind_02_cat = Reshape(target_shape=(3,))(model_ps_ind_02_cat)
    models.append(model_ps_ind_02_cat)

    input_ps_ind_04_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_ind_04_cat)
    model_ps_ind_04_cat = Embedding(3, 2, input_length=1)(input_ps_ind_04_cat)
    model_ps_ind_04_cat = Reshape(target_shape=(2,))(model_ps_ind_04_cat)
    models.append(model_ps_ind_04_cat)

    input_ps_ind_05_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_ind_05_cat)
    model_ps_ind_05_cat = Embedding(8, 5, input_length=1)(input_ps_ind_05_cat)
    model_ps_ind_05_cat = Reshape(target_shape=(5,))(model_ps_ind_05_cat)
    models.append(model_ps_ind_05_cat)

    input_ps_car_01_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_01_cat)
    model_ps_car_01_cat = Embedding(13, 7, input_length=1)(input_ps_car_01_cat)
    model_ps_car_01_cat = Reshape(target_shape=(7,))(model_ps_car_01_cat)
    models.append(model_ps_car_01_cat)

    input_ps_car_02_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_02_cat)
    model_ps_car_02_cat = Embedding(3, 2, input_length=1)(input_ps_car_02_cat)
    model_ps_car_02_cat = Reshape(target_shape=(2,))(model_ps_car_02_cat)
    models.append(model_ps_car_02_cat)

    input_ps_car_03_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_03_cat)
    model_ps_car_03_cat = Embedding(3, 2, input_length=1)(input_ps_car_03_cat)
    model_ps_car_03_cat = Reshape(target_shape=(2,))(model_ps_car_03_cat)
    models.append(model_ps_car_03_cat)

    input_ps_car_04_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_04_cat)
    model_ps_car_04_cat = Embedding(10, 5, input_length=1)(input_ps_car_04_cat)
    model_ps_car_04_cat = Reshape(target_shape=(5,))(model_ps_car_04_cat)
    models.append(model_ps_car_04_cat)

    input_ps_car_05_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_05_cat)
    model_ps_car_05_cat = Embedding(3, 2, input_length=1)(input_ps_car_05_cat)
    model_ps_car_05_cat = Reshape(target_shape=(2,))(model_ps_car_05_cat)
    models.append(model_ps_car_05_cat)

    input_ps_car_06_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_06_cat)
    model_ps_car_06_cat = Embedding(18, 8, input_length=1)(input_ps_car_06_cat)
    model_ps_car_06_cat = Reshape(target_shape=(8,))(model_ps_car_06_cat)
    models.append(model_ps_car_06_cat)

    input_ps_car_07_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_07_cat)
    model_ps_car_07_cat = Embedding(3, 2, input_length=1)(input_ps_car_07_cat)
    model_ps_car_07_cat = Reshape(target_shape=(2,))(model_ps_car_07_cat)
    models.append(model_ps_car_07_cat)

    input_ps_car_09_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_09_cat)
    model_ps_car_09_cat = Embedding(6, 3, input_length=1)(input_ps_car_09_cat)
    model_ps_car_09_cat = Reshape(target_shape=(3,))(model_ps_car_09_cat)
    models.append(model_ps_car_09_cat)

    input_ps_car_10_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_10_cat)
    model_ps_car_10_cat = Embedding(3, 2, input_length=1)(input_ps_car_10_cat)
    model_ps_car_10_cat = Reshape(target_shape=(2,))(model_ps_car_10_cat)
    models.append(model_ps_car_10_cat)

    input_ps_car_11_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_11_cat)
    model_ps_car_11_cat = Embedding(104, 10, input_length=1)(input_ps_car_11_cat)
    model_ps_car_11_cat = Reshape(target_shape=(10,))(model_ps_car_11_cat)
    models.append(model_ps_car_11_cat)

    input_ps_reg_01_plus_ps_car_02_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_reg_01_plus_ps_car_02_cat)
    model_ps_reg_01_plus_ps_car_02_cat = Embedding(24, 8, input_length=1)(input_ps_reg_01_plus_ps_car_02_cat)
    model_ps_reg_01_plus_ps_car_02_cat = Reshape(target_shape=(8,))(model_ps_reg_01_plus_ps_car_02_cat)
    models.append(model_ps_reg_01_plus_ps_car_02_cat)

    input_ps_reg_01_plus_ps_car_04_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_reg_01_plus_ps_car_04_cat)
    model_ps_reg_01_plus_ps_car_04_cat = Embedding(100, 10, input_length=1)(input_ps_reg_01_plus_ps_car_04_cat)
    model_ps_reg_01_plus_ps_car_04_cat = Reshape(target_shape=(10,))(model_ps_reg_01_plus_ps_car_04_cat)
    models.append(model_ps_reg_01_plus_ps_car_04_cat)

    input_others = Input(shape=(26,), dtype='float32')
    inputs.append(input_others)
    model_others = Dense(57,activation='relu', input_dim=26)(input_others)
    models.append(model_others)


    x  = Concatenate()(models)
    #x = add(Concatenate(x = )
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(.35)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(.15)(x)
    x = Dense(16, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(.15)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    adam = optimizers.Adadelta()
    model.compile(loss='binary_crossentropy', optimizer=adam)

    return model


def build_embedding_network_funcbn_int():

    inputs = []
    models = []

    
    input_ps_ind_02_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_ind_02_cat)
    model_ps_ind_02_cat = Embedding(5, 3, input_length=1)(input_ps_ind_02_cat)
    model_ps_ind_02_cat = Reshape(target_shape=(3,))(model_ps_ind_02_cat)
    models.append(model_ps_ind_02_cat)

    input_ps_ind_04_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_ind_04_cat)
    model_ps_ind_04_cat = Embedding(3, 2, input_length=1)(input_ps_ind_04_cat)
    model_ps_ind_04_cat = Reshape(target_shape=(2,))(model_ps_ind_04_cat)
    models.append(model_ps_ind_04_cat)

    input_ps_ind_05_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_ind_05_cat)
    model_ps_ind_05_cat = Embedding(8, 5, input_length=1)(input_ps_ind_05_cat)
    model_ps_ind_05_cat = Reshape(target_shape=(5,))(model_ps_ind_05_cat)
    models.append(model_ps_ind_05_cat)

    input_ps_car_01_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_01_cat)
    model_ps_car_01_cat = Embedding(13, 7, input_length=1)(input_ps_car_01_cat)
    model_ps_car_01_cat = Reshape(target_shape=(7,))(model_ps_car_01_cat)
    models.append(model_ps_car_01_cat)

    input_ps_car_02_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_02_cat)
    model_ps_car_02_cat = Embedding(3, 2, input_length=1)(input_ps_car_02_cat)
    model_ps_car_02_cat = Reshape(target_shape=(2,))(model_ps_car_02_cat)
    models.append(model_ps_car_02_cat)

    input_ps_car_03_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_03_cat)
    model_ps_car_03_cat = Embedding(3, 2, input_length=1)(input_ps_car_03_cat)
    model_ps_car_03_cat = Reshape(target_shape=(2,))(model_ps_car_03_cat)
    models.append(model_ps_car_03_cat)

    input_ps_car_04_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_04_cat)
    model_ps_car_04_cat = Embedding(10, 5, input_length=1)(input_ps_car_04_cat)
    model_ps_car_04_cat = Reshape(target_shape=(5,))(model_ps_car_04_cat)
    models.append(model_ps_car_04_cat)

    input_ps_car_05_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_05_cat)
    model_ps_car_05_cat = Embedding(3, 2, input_length=1)(input_ps_car_05_cat)
    model_ps_car_05_cat = Reshape(target_shape=(2,))(model_ps_car_05_cat)
    models.append(model_ps_car_05_cat)

    input_ps_car_06_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_06_cat)
    model_ps_car_06_cat = Embedding(18, 8, input_length=1)(input_ps_car_06_cat)
    model_ps_car_06_cat = Reshape(target_shape=(8,))(model_ps_car_06_cat)
    models.append(model_ps_car_06_cat)

    input_ps_car_07_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_07_cat)
    model_ps_car_07_cat = Embedding(3, 2, input_length=1)(input_ps_car_07_cat)
    model_ps_car_07_cat = Reshape(target_shape=(2,))(model_ps_car_07_cat)
    models.append(model_ps_car_07_cat)

    input_ps_car_09_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_09_cat)
    model_ps_car_09_cat = Embedding(6, 3, input_length=1)(input_ps_car_09_cat)
    model_ps_car_09_cat = Reshape(target_shape=(3,))(model_ps_car_09_cat)
    models.append(model_ps_car_09_cat)

    input_ps_car_10_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_10_cat)
    model_ps_car_10_cat = Embedding(3, 2, input_length=1)(input_ps_car_10_cat)
    model_ps_car_10_cat = Reshape(target_shape=(2,))(model_ps_car_10_cat)
    models.append(model_ps_car_10_cat)

    input_ps_car_11_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_11_cat)
    model_ps_car_11_cat = Embedding(104, 10, input_length=1)(input_ps_car_11_cat)
    model_ps_car_11_cat = Reshape(target_shape=(10,))(model_ps_car_11_cat)
    models.append(model_ps_car_11_cat)

    input_ps_reg_01_plus_ps_car_02_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_reg_01_plus_ps_car_02_cat)
    model_ps_reg_01_plus_ps_car_02_cat = Embedding(24, 8, input_length=1)(input_ps_reg_01_plus_ps_car_02_cat)
    model_ps_reg_01_plus_ps_car_02_cat = Reshape(target_shape=(8,))(model_ps_reg_01_plus_ps_car_02_cat)
    models.append(model_ps_reg_01_plus_ps_car_02_cat)

    input_ps_reg_01_plus_ps_car_04_cat = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_reg_01_plus_ps_car_04_cat)
    model_ps_reg_01_plus_ps_car_04_cat = Embedding(100, 10, input_length=1)(input_ps_reg_01_plus_ps_car_04_cat)
    model_ps_reg_01_plus_ps_car_04_cat = Reshape(target_shape=(10,))(model_ps_reg_01_plus_ps_car_04_cat)
    models.append(model_ps_reg_01_plus_ps_car_04_cat)

    input_ps_ind_01 = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_ind_01)
    model_ps_ind_01 = Embedding(8, 4, input_length=1)(input_ps_ind_01)
    model_ps_ind_01 = Reshape(target_shape=(4,))(model_ps_ind_01)
    models.append(model_ps_ind_01)

    input_ps_ind_03 = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_ind_03)
    model_ps_ind_03 = Embedding(12, 5, input_length=1)(input_ps_ind_03)
    model_ps_ind_03 = Reshape(target_shape=(5,))(model_ps_ind_03)
    models.append(model_ps_ind_03)

    input_ps_ind_14 = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_ind_14)
    model_ps_ind_14 = Embedding(5, 3, input_length=1)(input_ps_ind_14)
    model_ps_ind_14 = Reshape(target_shape=(3,))(model_ps_ind_14)
    models.append(model_ps_ind_14)

    input_ps_ind_15 = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_ind_15)
    model_ps_ind_15 = Embedding(14, 5, input_length=1)(input_ps_ind_15)
    model_ps_ind_15 = Reshape(target_shape=(5,))(model_ps_ind_15)
    models.append(model_ps_ind_15)

    input_ps_car_11 = Input(shape=(1,), dtype='int32')
    inputs.append(input_ps_car_11)
    model_ps_car_11 = Embedding(5, 3, input_length=1)(input_ps_car_11)
    model_ps_car_11 = Reshape(target_shape=(3,))(model_ps_car_11)
    models.append(model_ps_ind_11)

    input_others = Input(shape=(21,), dtype='float32')
    inputs.append(input_others)
    model_others = Dense(37,activation='relu', input_dim=21)(input_others)
    models.append(model_others)


    x  = Concatenate()(models)
    #x = add(Concatenate(x = )
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(.35)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(.15)(x)
    x = Dense(16, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(.15)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    adam = optimizers.Adadelta()
    model.compile(loss='binary_crossentropy', optimizer=adam)

    return model



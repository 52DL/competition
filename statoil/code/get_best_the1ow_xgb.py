MAX_XGB_ROUNDS = 2175
OPTIMIZE_XGB_ROUNDS = True
XGB_LEARNING_RATE = 0.03
XGB_EARLY_STOPPING_ROUNDS = 100


import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from rgf.sklearn import RGFClassifier
from numba import jit
import time
import gc
import subprocess
import glob
from multiprocessing import Pool
from tqdm import tqdm
import gc
import datetime as dt
from random import choice, sample, shuffle, uniform, seed
from math import exp, expm1, log1p, log10, log2, sqrt, ceil, floor, isfinite, isnan
from itertools import combinations
#import for image processing
import cv2
from scipy.stats import kurtosis, skew
from scipy.ndimage import laplace, sobel
#evaluation
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import log_loss,accuracy_score
import xgboost as xgb


#start = time.clock()
#tt_data = get_feat_data(FEAT_FOLDER)
#print("loading data used time: ", (time.clock()-start))
def read_jason(file='', loc='../input/'):

    df = pd.read_json('{}{}'.format(loc, file))
    df['inc_angle'] = df['inc_angle'].replace('na', -1).astype(float)
    #print(df['inc_angle'].value_counts())
    
    band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]])
    band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]])
    df = df.drop(['band_1', 'band_2'], axis=1)
    ida = df['id']    
    bands = np.stack((band1, band2,  0.5 * (band1 + band2)), axis=-1)
    del band1, band2
    
    return df, bands, ida
#forked from
#https://www.kaggle.com/the1owl/planet-understanding-the-amazon-from-space/natural-growth-patterns-fractals-of-nature/notebook
def img_to_stats(paths):
    
    img_id, img = paths[0], paths[1]
    
    #ignored error    
    np.seterr(divide='ignore', invalid='ignore')
    
    bins = 20
    scl_min, scl_max = -50, 50
    opt_poly = True
    #opt_poly = False
    
    try:
        st = []
        st_interv = []
        hist_interv = []
        for i in range(img.shape[2]):
            img_sub = np.squeeze(img[:, :, i])
            
            #median, max and min
            sub_st = []
            sub_st += [np.mean(img_sub), np.std(img_sub), np.max(img_sub), np.median(img_sub), np.min(img_sub)]
            sub_st += [(sub_st[2] - sub_st[3]), (sub_st[2] - sub_st[4]), (sub_st[3] - sub_st[4])] 
            sub_st += [(sub_st[-3] / sub_st[1]), (sub_st[-2] / sub_st[1]), (sub_st[-1] / sub_st[1])] #normalized by stdev
            st += sub_st
            #Laplacian, Sobel, kurtosis and skewness
            st_trans = []
            st_trans += [laplace(img_sub, mode='reflect', cval=0.0).ravel().var()] #blurr
            sobel0 = sobel(img_sub, axis=0, mode='reflect', cval=0.0).ravel().var()
            sobel1 = sobel(img_sub, axis=1, mode='reflect', cval=0.0).ravel().var()
            st_trans += [sobel0, sobel1]
            st_trans += [kurtosis(img_sub.ravel()), skew(img_sub.ravel())]
            
            if opt_poly:
                st_interv.append(sub_st)
                #
                st += [x * y for x, y in combinations(st_trans, 2)]
                st += [x + y for x, y in combinations(st_trans, 2)]
                st += [x - y for x, y in combinations(st_trans, 2)]                
 
            #hist
            #hist = list(cv2.calcHist([img], [i], None, [bins], [0., 1.]).flatten())
            hist = list(np.histogram(img_sub, bins=bins, range=(scl_min, scl_max))[0])
            hist_interv.append(hist)
            st += hist
            st += [hist.index(max(hist))] #only the smallest index w/ max value would be incl
            st += [np.std(hist), np.max(hist), np.median(hist), (np.max(hist) - np.median(hist))]

        if opt_poly:
            for x, y in combinations(st_interv, 2):
                st += [float(x[j]) * float(y[j]) for j in range(len(st_interv[0]))]

            for x, y in combinations(hist_interv, 2):
                hist_diff = [x[j] * y[j] for j in range(len(hist_interv[0]))]
                st += [hist_diff.index(max(hist_diff))] #only the smallest index w/ max value would be incl
                st += [np.std(hist_diff), np.max(hist_diff), np.median(hist_diff), (np.max(hist_diff) - np.median(hist_diff))]
                
        #correction
        nan = -999
        for i in range(len(st)):
            if isnan(st[i]) == True:
                st[i] = nan
                
    except:
        print('except: ')
    
    return [img_id, st]


def extract_img_stats(paths):
    imf_d = {}
    p = Pool(8) #(cpu_count())
    ret = p.map(img_to_stats, paths)
    for i in tqdm(range(len(ret)), miniters=100):
        imf_d[ret[i][0]] = ret[i][1]

    ret = []
    fdata = [imf_d[i] for i, j in paths]
    return np.array(fdata, dtype=np.float32)


def process(df, bands):

    data = extract_img_stats([(k, v) for k, v in zip(df['id'].tolist(), bands)]); gc.collect()
    data = np.concatenate([data, df['inc_angle'].values[:, np.newaxis]], axis=-1); gc.collect()

    print(data.shape)
    return data

#Load data
train, train_bands, id_train = read_jason(file='train.json', loc='../input/')
test, test_bands, id_test = read_jason(file='test.json', loc='../input/')

X = process(df=train, bands=train_bands)
y = train['is_iceberg'].values

test_df = process(df=test, bands=test_bands)

y_valid_pred = 0.0*y
y_test_pred = 0.0

# Set up folds
K = 5
kf = KFold(n_splits = K, random_state = 1108, shuffle = True)
np.random.seed(1108)

# Set up classifier
xgbmodel = XGBClassifier(    
                        n_estimators=MAX_XGB_ROUNDS,
                        max_depth=3,
                        objective="binary:logistic",
                        learning_rate=XGB_LEARNING_RATE, 
                        subsample=.9,
                        min_child_weight=.2,
                        colsample_bytree=.9,
                        scale_pos_weight=1,
                        gamma=0,
                        reg_alpha=0.2,
                        reg_lambda=0.4,
                     )

# Run CV

for i, (train_index, test_index) in enumerate(kf.split(X)):
    
    # Create data for this fold
    y_train, y_valid = y[train_index], y[test_index]
    X_train, X_valid = X[train_index,:], X[test_index,:]
    X_test = test_df
    print( "\nFold ", i)
    
    # Run model for this fold
    if OPTIMIZE_XGB_ROUNDS:
        eval_set=[(X_valid,y_valid)]
        fit_model = xgbmodel.fit( X_train, y_train, 
                               eval_set=eval_set,
                               eval_metric='logloss',
                               early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
                               verbose=50
                             )
        print( "  Best N trees = ", xgbmodel.best_ntree_limit )
        print( "  Best gini = ", xgbmodel.best_score )
    else:
        fit_model = xgbmodel.fit( X_train, y_train )
        
    pred = fit_model.predict_proba(X_train, ntree_limit=xgbmodel.best_ntree_limit+1)[:,1]
    print( "  train Gini = ", log_loss(y_train, pred), accuracy_score(y_train,np.round(pred)) )
    # Generate validation predictions for this fold
    pred = fit_model.predict_proba(X_valid, ntree_limit=xgbmodel.best_ntree_limit+1)[:,1]
    print( "  valid Gini = ", log_loss(y_valid, pred), accuracy_score(y_valid,np.round(pred)) )
    y_valid_pred[test_index] = pred
    
    # Accumulate test set predictions
    probs = fit_model.predict_proba(X_test, ntree_limit=xgbmodel.best_ntree_limit+1)[:,1]
    y_test_pred += probs
    
    del X_test, X_train, X_valid, y_train
    
y_test_pred /= K  # Average test set predictions

print( "\nGini for full training set:" ,log_loss(y, y_valid_pred))

# Save validation predictions for stacking/ensembling
val = pd.DataFrame()
val['id'] = id_train
val['is_iceberg'] = y_valid_pred
val.to_csv('../subm/ctt+xgb_valid.csv', float_format='%.6f', index=False)

# Create submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['is_iceberg'] = y_test_pred
sub.to_csv('../subm/ctt+xgb_submit.csv', float_format='%.6f', index=False)


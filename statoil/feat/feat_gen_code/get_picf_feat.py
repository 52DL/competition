
import numpy as np
import pandas as pd
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


def process(df, bands, ids):

    data = extract_img_stats([(k, v) for k, v in zip(df['id'].tolist(), bands)]); gc.collect()
    #data = np.concatenate([data, ids.values[:, np.newaxis]], axis=-1); gc.collect()

    print(data.shape)
    return data

#Load data
train, train_bands, id_train = read_jason(file='train.json', loc='../../input/')
test, test_bands, id_test = read_jason(file='test.json', loc='../../input/')

X = process(df=train, bands=train_bands, ids=id_train)
X = pd.DataFrame(X)
X['id'] = id_train
X.to_csv('../picf_train_opted.csv',index=False)

test_df = process(df=test, bands=test_bands, ids=id_test)
test_df = pd.DataFrame(test_df)
test_df['id'] = id_test
test_df.to_csv('../picf_test_opted.csv',index=False)

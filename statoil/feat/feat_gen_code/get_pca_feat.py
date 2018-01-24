import numpy as np
import pandas as pd
from sklearn import *
import lightgbm as lgb
import random

train = pd.read_json("../../input/train.json").fillna(-1.0).replace('na', -1.0)
test = pd.read_json("../../input/test.json").fillna(-1.0).replace('na', -1.0)
#train['angle_l'] = train['inc_angle'].apply(lambda x: len(str(x))) <= 7
#test['angle_l'] = test['inc_angle'].apply(lambda x: len(str(x))) <= 7
train['null_angle'] = (train['inc_angle']==-1).values
test['null_angle'] = (test['inc_angle']==-1).values
x1 = train[train['inc_angle']!= -1.0]
x2 = train[train['inc_angle']== -1.0]
del train;
print(x1.values.shape, x2.values.shape)

pca_b1 = decomposition.PCA(n_components=50, whiten=False, random_state=12)
pca_b2 = decomposition.PCA(n_components=50, whiten=False, random_state=13)
etc = ensemble.ExtraTreesRegressor(n_estimators=200, max_depth=7, n_jobs=-1, random_state=14)

band1 = [np.array(band).astype(np.float32).flatten() for band in x1["band_1"]]
band2 = [np.array(band).astype(np.float32).flatten() for band in x1["band_2"]]
band1 = pd.DataFrame(pca_b1.fit_transform(band1))
band1.columns = [str(c)+'_1' for c in band1.columns]
band2 = pd.DataFrame(pca_b2.fit_transform(band2))
band2.columns = [str(c)+'_2' for c in band2.columns]
features = pd.concat((band1, band2), axis=1, ignore_index=True)
etc.fit(features, x1.inc_angle)

band1 = [np.array(band).astype(np.float32).flatten() for band in x2["band_1"]]
band2 = [np.array(band).astype(np.float32).flatten() for band in x2["band_2"]]
band1 = pd.DataFrame(pca_b1.transform(band1))
band1.columns = [str(c)+'_1' for c in band1.columns]
band2 = pd.DataFrame(pca_b2.fit_transform(band2))
band2.columns = [str(c)+'_2' for c in band2.columns]
features = pd.concat((band1, band2), axis=1, ignore_index=True)
x2['inc_angle'] = etc.predict(features)

train = pd.concat((x1, x2), axis=0, ignore_index=True).reset_index(drop=True)
del x1; del x2;
print(train.values.shape)

pca_b1 = decomposition.PCA(n_components=50, whiten=True, random_state=15)
pca_b2 = decomposition.PCA(n_components=50, whiten=True, random_state=16)
pca_b3 = decomposition.PCA(n_components=50, whiten=True, random_state=17)
pca_b4 = decomposition.PCA(n_components=50, whiten=True, random_state=18)

band1 = [np.array(band).astype(np.float32).flatten() for band in train["band_1"]]
band2 = [np.array(band).astype(np.float32).flatten() for band in train["band_2"]]
pd_band1 = pd.DataFrame(band1)
pd_band2 = pd.DataFrame(band2)
pd_band3 = pd.DataFrame(np.dot(np.diag(train['inc_angle'].values), ((pd_band1 + pd_band2) / 2)))
pd_band4 = pd.DataFrame(np.dot(np.diag(train['inc_angle'].values), ((pd_band1 - pd_band2) / 2)))
band1 = pd.DataFrame(pca_b1.fit_transform(pd_band1))
band1.columns = [str(c)+'_1' for c in band1.columns]
band2 = pd.DataFrame(pca_b2.fit_transform(pd_band2))
band2.columns = [str(c)+'_2' for c in band2.columns]
band3 = pd.DataFrame(pca_b3.fit_transform(pd_band3.values))
band3.columns = [str(c)+'_3' for c in band3.columns]
band4 = pd.DataFrame(pca_b4.fit_transform(pd_band4.values))
band4.columns = [str(c)+'_4' for c in band4.columns]
features = pd.concat((band1, band2, band3, band4), axis=1, ignore_index=True).reset_index(drop=True)
features.columns = ['pca_'+str(c) for c in features.columns]
features['inc_angle'] = train['inc_angle']
#features['angle_l'] = train['angle_l']
features['null_angle'] = train['null_angle']
features['band1_min'] = pd_band1.min(axis=1, numeric_only=True)
features['band2_min'] = pd_band2.min(axis=1, numeric_only=True)
features['band3_min'] = pd_band3.min(axis=1, numeric_only=True)
features['band4_min'] = pd_band4.min(axis=1, numeric_only=True)
features['band1_max'] = pd_band1.max(axis=1, numeric_only=True)
features['band2_max'] = pd_band2.max(axis=1, numeric_only=True)
features['band3_max'] = pd_band3.max(axis=1, numeric_only=True)
features['band4_max'] = pd_band4.max(axis=1, numeric_only=True)
features['band1_med'] = pd_band1.median(axis=1, numeric_only=True)
features['band2_med'] = pd_band2.median(axis=1, numeric_only=True)
features['band3_med'] = pd_band3.median(axis=1, numeric_only=True)
features['band4_med'] = pd_band4.median(axis=1, numeric_only=True)
features['band1_mea'] = pd_band1.mean(axis=1, numeric_only=True)
features['band2_mea'] = pd_band2.mean(axis=1, numeric_only=True)
features['band3_mea'] = pd_band3.mean(axis=1, numeric_only=True)
features['band4_mea'] = pd_band4.mean(axis=1, numeric_only=True)
features['id'] = train['id']
del pd_band1; del pd_band2; del pd_band3; del pd_band4
features1 = features.copy()
features1.to_csv('../pca_train.csv',index=False,float_format='%.6f')

########################################################################
band1 = [np.array(band).astype(np.float32).flatten() for band in test["band_1"]]
band2 = [np.array(band).astype(np.float32).flatten() for band in test["band_2"]]
pd_band1 = pd.DataFrame(band1)
pd_band2 = pd.DataFrame(band2)
pd_band3 = pd.DataFrame(np.dot(np.diag(test['inc_angle'].values), ((pd_band1 + pd_band2) / 2)))
pd_band4 = pd.DataFrame(np.dot(np.diag(test['inc_angle'].values), ((pd_band1 - pd_band2) / 2)))
band1 = pd.DataFrame(pca_b1.transform(pd_band1))
band1.columns = [str(c)+'_1' for c in band1.columns]
band2 = pd.DataFrame(pca_b2.transform(pd_band2))
band2.columns = [str(c)+'_2' for c in band2.columns]
band3 = pd.DataFrame(pca_b3.transform(pd_band3.values))
band3.columns = [str(c)+'_3' for c in band3.columns]
band4 = pd.DataFrame(pca_b4.transform(pd_band4.values))
band4.columns = [str(c)+'_4' for c in band4.columns]
features = pd.concat((band1, band2, band3, band4), axis=1, ignore_index=True).reset_index(drop=True)
features.columns = ['pca_'+str(c) for c in features.columns]
features['inc_angle'] = test['inc_angle']
#features['angle_l'] = test['angle_l']
features['null_angle'] = test['null_angle']
features['band1_min'] = pd_band1.min(axis=1, numeric_only=True)
features['band2_min'] = pd_band2.min(axis=1, numeric_only=True)
features['band3_min'] = pd_band3.min(axis=1, numeric_only=True)
features['band4_min'] = pd_band4.min(axis=1, numeric_only=True)
features['band1_max'] = pd_band1.max(axis=1, numeric_only=True)
features['band2_max'] = pd_band2.max(axis=1, numeric_only=True)
features['band3_max'] = pd_band3.max(axis=1, numeric_only=True)
features['band4_max'] = pd_band4.max(axis=1, numeric_only=True)
features['band1_med'] = pd_band1.median(axis=1, numeric_only=True)
features['band2_med'] = pd_band2.median(axis=1, numeric_only=True)
features['band3_med'] = pd_band3.median(axis=1, numeric_only=True)
features['band4_med'] = pd_band4.median(axis=1, numeric_only=True)
features['band1_mea'] = pd_band1.mean(axis=1, numeric_only=True)
features['band2_mea'] = pd_band2.mean(axis=1, numeric_only=True)
features['band3_mea'] = pd_band3.mean(axis=1, numeric_only=True)
features['band4_mea'] = pd_band4.mean(axis=1, numeric_only=True)
features['id'] = test['id']
del pd_band1; del pd_band2; del pd_band3
features2 = features.copy()
features2.to_csv('../pca_test.csv',index=False,float_format='%.6f')



import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

train = pd.read_csv('../irgpu_train.csv')
test = pd.read_csv('../irgpu_test.csv')
id_train = train['id']
id_test = test['id']
train.drop(['id'], axis = 1, inplace = True)
test.drop(['id'], axis = 1, inplace = True)
pca = PCA(n_components=20, whiten=False, random_state=12)
train = pca.fit_transform(train)
test = pca.transform(test)

# Save validation predictions for stacking/ensembling
val = pd.DataFrame(train)
val.columns = ['ir_' + str(c) for c in val.columns]
val['id'] = id_train
val.to_csv('../ir20_train.csv', float_format='%.6f', index=False)

# Create submission file
sub = pd.DataFrame(test)
sub.columns = ['ir_' + str(c) for c in sub.columns]
sub['id'] = id_test
sub.to_csv('../ir20_test.csv', float_format='%.6f', index=False)


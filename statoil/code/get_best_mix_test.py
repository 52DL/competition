import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,accuracy_score

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

train = pd.read_csv('../feat/mixlr_train.csv')
test = pd.read_csv('../feat/mixlr_test.csv')
col = [c for c in train.columns if c not in ['id']]
#print(col)
X = train[col]
print(X.info())
X = train[col].values
#print(X[0])
print(X.shape)
test_df = test[col].values
print(test_df.shape)
from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range=(-1,1)).fit(X)
X = scaling.transform(X)
test_df = scaling.transform(test_df)

from sklearn.decomposition import PCA
pca = PCA(n_components=30, whiten=False, random_state=12)
X = pca.fit_transform(X)
test_df = pca.transform(test_df)

y_valid_pred = 0.0*y
y_train_pred = 0.0*y
y_test_pred = 0.0

# Set up folds
K = 5
kf = KFold(n_splits = K, random_state = 1108, shuffle = True)
np.random.seed(1108)
for m in [1]:
    print('m: ',m)
    log_model = LogisticRegression(penalty="l1",C=.1,random_state=m)
    if False:
        from sklearn.neighbors import KNeighborsClassifier
        log_model = KNeighborsClassifier(n_neighbors=5, weights='distance')
    if True:
        from sklearn.naive_bayes import GaussianNB
        log_model = GaussianNB()
    for i, (train_index, test_index) in enumerate(kf.split(X)):

        # Create data for this fold
        y_train, y_valid = y[train_index].copy(), y[test_index]
        X_train, X_valid = X[train_index,:].copy(), X[test_index,:].copy()
        X_test = test_df.copy()
        print( "\nFold ", i)

        # Run model for this fold
        fit_model = log_model.fit( X_train, y_train )

        # Generate validation predictions for this fold
        pred = fit_model.predict_proba(X_valid)[:,1]
        #print(fit_model.coef_)
        #for i,j in zip(feat_names, fit_model.coef_[0]):
        #    print(i, " : ", j)
        print( "  Gini = ", log_loss(y_valid, pred), accuracy_score(y_valid,np.round(pred)))
        y_valid_pred[test_index] = pred

        # Accumulate test set predictions
        probs = fit_model.predict_proba(X_test)[:,1]
        y_test_pred += probs

        del X_test, X_train, X_valid, y_train
    y_test_pred /= K  # Average test set predictions
    print( "\nGini for full training set:" ,log_loss(y, y_valid_pred))
    # Save validation predictions for stacking/ensembling
    val = pd.DataFrame()
    val['id'] = id_train
    val['is_iceberg'] = y_valid_pred
    val.to_csv('../subm/mix+test_valid.csv', float_format='%.6f', index=False)

    # Create submission file
    sub = pd.DataFrame()
    sub['id'] = id_test
    sub['is_iceberg'] = y_test_pred
    sub.to_csv('../subm/mix+test_submit.csv', float_format='%.6f', index=False)


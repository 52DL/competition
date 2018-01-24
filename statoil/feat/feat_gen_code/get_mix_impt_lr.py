import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import gc
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,accuracy_score

def test_id(a,b):
    for i,j in zip(a,b):
        if i!=j:
            print('asdf')
#Load data
train_org = pd.read_json('../../input/train.json')
id_train = train_org['id']
test_org = pd.read_json('../../input/test.json' )
id_test = test_org['id']

y = train_org['is_iceberg'].values
#del train, test

picf_train = pd.read_csv('../picf_train_opted.csv')
#print('picf:',picf_train.shape)
picf_test = pd.read_csv('../picf_test_opted.csv')
sbmr_train = pd.read_csv('../sbmr_train.csv')
#print('sbmr:',sbmr_train.shape)
sbmr_test = pd.read_csv('../sbmr_test.csv')
pca_train = pd.read_csv('../pca_train.csv')
#print('pca:',pca_train.shape)
pca_test = pd.read_csv('../pca_test.csv')
hog_train = pd.read_csv('../hog_train.csv')
#print('hog:',hog_train.shape)
hog_test = pd.read_csv('../hog_test.csv')
fft_train = pd.read_csv('../fft_train.csv')
#print('fft:',fft_train.shape)
fft_test = pd.read_csv('../fft_test.csv')
ir20_train = pd.read_csv('../ir20_train.csv')
#print('ir20:',ir20_train.shape)
ir20_test = pd.read_csv('../ir20_test.csv')
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
train = picf_train
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
test = picf_test
#test = pca_test
#test['inc_angle'] = pca_test['inc_angle']
test['inc_angle'] = test_org['inc_angle'].replace('na', 0).astype(float)
print(test.shape)

result_train = train.loc[:,:]
result_test = test.loc[:,:]
col = [c for c in train.columns if c not in ['id']]
#print(col)
X = train.loc[:,col]
#print(X.info())
print(X.shape)
test_df = test.loc[:,col]
print(test_df.shape)

# Set up folds
K = 5
kf = KFold(n_splits = K, random_state = 1108, shuffle = True)
log_model = LogisticRegression(penalty="l1",C=.1,random_state=1)
np.random.seed(1108)
global cv
cv = [[],1]
def get_cv(cols):
    y_valid_pred = 0.0*y
    X1 = X.values
    gain_dict = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):

        # Create data for this fold
        y_train, y_valid = y[train_index].copy(), y[test_index]
        X_train, X_valid = X1[train_index,:].copy(), X1[test_index,:].copy()
        #X_test = test_df.copy()
        #print( "\nFold ", i)

        # Run model for this fold
        fit_model = log_model.fit( X_train, y_train )
        gain_dict.append(fit_model.coef_[0]) 
        # Generate validation predictions for this fold
        pred = fit_model.predict_proba(X_valid)[:,1]
        #print(fit_model.coef_)
        #for i,j in zip(cols, fit_model.coef_[0]):
        #    print(i, " : ", j)
        #print( "  Gini = ", log_loss(y_valid, pred), accuracy_score(y_valid,np.round(pred)))
        y_valid_pred[test_index] = pred

        # Accumulate test set predictions
        #probs = fit_model.predict_proba(X_test)[:,1]
        #y_test_pred += probs

        del X_train, X_valid, y_train, y_valid
    logloss = log_loss(y, y_valid_pred)
    gain = []
    #print(gain_dict)
    for i in range(len(cols)):
        gain.append(abs(gain_dict[0][i])+abs(gain_dict[1][i])+abs(gain_dict[2][i])+abs(gain_dict[3][i])+abs(gain_dict[4][i]))
    #print("feature_", col_name," gain: ",gain)
    return [gain,logloss]#,accuracy_score(y,np.round(y_valid_pred)))


epoch = 0
cv_dict = {}
drop_dict = {}
num_drop = 50
while(True):
    print('\nepoch :',epoch, X.shape)
    cols = X.columns
    cv = get_cv(cols)
    cv_dict[epoch] = cv[1]
    print('cv :',cv[1], X.shape)
    if epoch > 0 and cv_dict[epoch] > cv_dict[epoch-1]:
        if num_drop == 1:
            print('time to quit')
            break
        print('time to 1')
        X[drop_dict[epoch-1]] = result_train[drop_dict[epoch-1]]
        test_df[drop_dict[epoch-1]] = result_test[drop_dict[epoch-1]]
        epoch -= 1
        num_drop = int(num_drop/2)
        cols = X.columns
        cv = get_cv(cols)
        cv_dict[epoch] = cv[1]
        print('cv :',cv[1], X.shape)
    dropgain = cv[0]
    cols = np.array(cols)[:,np.newaxis]
    dropgain = np.array(dropgain)[:,np.newaxis]
    temp = pd.DataFrame(np.concatenate([cols,dropgain],axis=1))
    temp = temp.sort_values(by=1,ascending=True)
    #print(temp)
    if num_drop > len(cols):
        num_drop = 1
    print('drop nums: ', num_drop)
    dropcols = temp.values[:num_drop,0]
    drop_dict[epoch] = dropcols
    X.drop(dropcols,axis=1,inplace=True)
    test_df.drop(dropcols,axis=1,inplace=True)
    epoch += 1
#print(cv_dict)
#print(drop_dict)
index = sorted(cv_dict.items(), key=lambda asd:asd[1], reverse=False)[0][0]
#print(index)
for i in range(index):
    result_train.drop(drop_dict[i],axis=1,inplace=True)
    result_test.drop(drop_dict[i],axis=1,inplace=True)
print(result_train.shape)
print(result_test.shape)
# Save validation predictions for stacking/ensembling
result_train.to_csv('../mixIlr_train.csv', float_format='%.6f', index=False)

# Create submission file
result_test.to_csv('../mixIlr_test.csv', float_format='%.6f', index=False)

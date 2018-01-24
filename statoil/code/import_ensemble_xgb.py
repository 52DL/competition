import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import gc
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
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

feat_names = []
#feat_names.append("orig+TLvgg16+Adam")
#feat_names.append("orig+TLres50")
feat_names.append("orig1+TLvgg16+Adam")
feat_names.append("orig1+TLres50")
#feat_names.append("origf1+TLdense121")

feat_names.append("all+TLvgg16+Adam")

#feat_names.append("trans+TLvgg16+Adam")
#feat_names.append("trans+TLres50")
feat_names.append("trans1+TLvgg16+Adam")
feat_names.append("trans1+TLres50")

#feat_names.append("norm+TLvgg16")
#feat_names.append("norm+TLres50")
feat_names.append("norm1+TLvgg16")
feat_names.append("norm1+TLres50")
feat_names.append("norm1+TLdense121")

feat_names.append("normf1+TLvgg16")
feat_names.append("norm+lenet")

feat_names.append("ctt+xgb")
feat_names.append("picf+xgb")
feat_names.append("pca+lgb")

feat_names.append("mix+lr")
feat_names.append("mix+xgb")
feat_names.append("mix+lgb")
feat_names.append("mix+svm")

valids = [pd.read_csv('../subm/'+f+'_valid.csv') for f in feat_names]
valid = pd.merge(valids[0],valids[1] , how='left',on='id')
for i in range(2,len(feat_names)):
    valid = pd.merge(valid,valids[i],how='left',on='id')
cols = ['id']+feat_names
valid.columns = cols
train=valid
#valid['is_iceberg'] = y
tests = [pd.read_csv('../subm/'+f+'_submit.csv') for f in feat_names]
test = pd.merge(tests[0],tests[1] , how='left',on='id')
for i in range(2,len(feat_names)):
    test = pd.merge(test,tests[i],how='left',on='id')
cols = ['id']+feat_names
test.columns = cols

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
log_model = XGBClassifier(
                        n_estimators=4000,
                        max_depth=5,
                        objective="binary:logistic",
                        learning_rate=0.01,
                        subsample=1,
                        min_child_weight=1,
                        colsample_bytree=1,
                        scale_pos_weight=1,
                        gamma=0,
                        reg_alpha=0,
                        reg_lambda=1,
                     ) 
np.random.seed(1108)
global cv
cv = [[],1]
def get_cv(cols,first):
    y_valid_pred = 0.0*y
    X1 = X.loc[:,cols].values
    gain_dict = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):

        # Create data for this fold
        y_train, y_valid = y[train_index].copy(), y[test_index]
        X_train, X_valid = X1[train_index,:].copy(), X1[test_index,:].copy()
        #X_test = test_df.copy()
        #print( "\nFold ", i)

        # Run model for this fold
        eval_set=[(X_valid,y_valid)]
        fit_model = log_model.fit( X_train, y_train,
                               eval_set=eval_set,
                               eval_metric='logloss',
                               early_stopping_rounds=200,
                               #verbose=1000
                               verbose=False
                             )
        if first == True:
            imp_vals = fit_model.booster().get_fscore()
            imp_dict = {cols[i]:float(imp_vals.get('f'+str(i),0.)) for i in range(len(cols))}
            gain_dict.append(imp_dict) 
        # Generate validation predictions for this fold
        pred = fit_model.predict_proba(X_valid, ntree_limit=log_model.best_ntree_limit)[:,1]
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
    if first == True:
        for i in cols:
            gain.append(gain_dict[0][i]+gain_dict[1][i]+gain_dict[2][i]+gain_dict[3][i]+gain_dict[4][i])
    #print("feature_", col_name," gain: ",gain)
    return [gain,logloss]#,accuracy_score(y,np.round(y_valid_pred)))


cols = X.columns
cv = get_cv(cols,True)
print('orig cv :',cv[1], X.shape)
gain = cv[0]
cols = np.array(cols)[:,np.newaxis]
gain = np.array(gain)[:,np.newaxis]
temp = pd.DataFrame(np.concatenate([cols,gain],axis=1))
temp = temp.sort_values(by=1,ascending=False)

num_use = 8
dis = 2
while(True):
    loss = []
    ids = [3,2,1,0,-1,-2,-3]
    print('\nuse nums: ', [num_use-i*dis for i in ids])
    for use in [num_use-i*dis for i in ids]:
        if use<=0:
            loss.append(1)
            continue
        usecols = temp.values[:use,0]
        loss.append(get_cv(usecols,False)[1])
    print('losses:   ', loss)
    a = loss.index(min(loss))
    num_use = num_use-ids[a]*dis
    if dis == 1:
        break
    dis = int(dis/2)
print('use: ',num_use,'loss: ',min(loss))    
usecols = temp.values[:num_use,0]
usecols = usecols.tolist()
print('cols: ',usecols)
'''
usecols.append('id')
result_train = result_train.loc[:,usecols]
result_test = result_test.loc[:,usecols]
print(result_train.shape)
print(result_test.shape)
# Save validation predictions for stacking/ensembling
result_train.to_csv('../subm/ensemble/Ixgb_train.csv', float_format='%.6f', index=False)

# Create submission file
result_test.to_csv('../subm/ensemble/Ixgb_test.csv', float_format='%.6f', index=False)
'''

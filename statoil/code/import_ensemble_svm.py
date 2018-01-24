import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import gc
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
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

col = [c for c in train.columns if c not in ['id']]
#print(col)
X = train.loc[:,col]
#print(X.info())
print(X.shape)
test_df = test.loc[:,col]
print(test_df.shape)
from sklearn.preprocessing import MinMaxScaler
#blend = np.concatenate([X,test_df])
#scaling = MinMaxScaler(feature_range=(-1,1)).fit(blend)
#blend = scaling.transform(blend)
#X = pd.DataFrame(blend[:X.shape[0],:],columns=X.columns)
#test_df = pd.DataFrame(blend[X.shape[0]:,:],columns=test_df.columns)
scaling = MinMaxScaler(feature_range=(-1,1)).fit(X)
X = pd.DataFrame(scaling.transform(X),columns=X.columns)
test_df = pd.DataFrame(scaling.transform(test_df),columns=test_df.columns)
#print(X.values)
#print(test_df.values)
# Set up folds
K = 5
kf = KFold(n_splits = K, random_state = 1108, shuffle = True)
np.random.seed(1108)
for m in [.0025,.003,.0035]:
    print('m: ',m)
    #log_model = LinearSVC(C=1,penalty='l1',loss='hinge')
    lsvc = LinearSVC(C=.2,random_state=1)
    rbfsvc = SVC(C=m,probability=True,random_state=1)
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
            if first == True:
                fit_model = lsvc.fit( X_train, y_train)
                imp_vals = fit_model.coef_[0]
                gain_dict.append(imp_vals) 
            # Generate validation predictions for this fold
            else:
                fit_model = rbfsvc.fit( X_train, y_train)
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
        if first == True:
            for i in range(len(cols)):
                gain.append(abs(gain_dict[0][i])+abs(gain_dict[1][i])+abs(gain_dict[2][i])+abs(gain_dict[3][i])+abs(gain_dict[4][i]))
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
        #print('\nuse nums: ', [num_use-i*dis for i in ids])
        for use in [num_use-i*dis for i in ids]:
            if use<=0:
                loss.append(1)
                continue
            usecols = temp.values[:use,0]
            loss.append(get_cv(usecols,False)[1])
        #print('losses:   ', loss)
        a = loss.index(min(loss))
        num_use = num_use-ids[a]*dis
        if dis == 1:
            break
        dis = int(dis/2)
    print('use: ',num_use,'loss: ',min(loss))    
    #num_use = 111
    usecols = temp.values[:num_use,0]
    usecols = usecols.tolist()
    print('cols: ',usecols)
    '''
    print(get_cv(usecols,False)[1]) 
    result_train = pd.DataFrame(X.loc[:,usecols])
    result_train['id'] = id_train
    # Save validation predictions for stacking/ensembling
    result_train.to_csv('../mixIsvm_train.csv', float_format='%.6f', index=False)

    # Create submission file
    result_test = pd.DataFrame(test_df.loc[:,usecols])
    result_test['id'] = id_test
    result_test.to_csv('../mixIsvm_test.csv', float_format='%.6f', index=False)
    '''

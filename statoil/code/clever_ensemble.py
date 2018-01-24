import numpy as np
import pandas as pd
import scipy as sp
from sklearn.metrics import log_loss
from scipy.optimize import minimize


def get_minmax(df):
    a=np.where(np.sum((df.iloc[:,:6]>.5).astype(int),axis=1)>3, 
                                    1,#df['is_iceberg_max'], 
                                    1e-4)#df['is_iceberg_min'])
    
    return a

def get_max(df):
    a=df.iloc[:,:6].max(axis=1) 
    return a
                                
def get_mean(df):
    a=df.iloc[:,:6].mean(axis=1) 
    return a
                                
    
def get_median(df):
    a=df.iloc[:,:6].median(axis=1) 
    return a

def mae_func(weights,Y_values,predictions):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction
    #print(final_prediction.shape)
    #print(prediction.shape)
    #print(Y_values.shape)
    return log_loss(Y_values, final_prediction)

def get_mini(df,bestWght):
    predictions=[]
    #print(Y_values.shape) 
    lls = []
    wghts = []

    for i in range(6):
        predictions.append(np.array(df.iloc[:,i]))
  
    if bestWght==[]:
        Y_values = df['is_iceberg'].values
        for i in range(100):
            starting_values = np.random.uniform(size=6)
            cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
            bounds = [(0,1)]*len(predictions)
       
            res = minimize(fun=mae_func, x0=starting_values,args=(Y_values, predictions), method='L-BFGS-B', bounds=bounds, options={'disp': False, 'maxiter': 100000})

            lls.append(res['fun'])
            wghts.append(res['x'])
        # Uncomment the next line if you want to see the weights and scores calculated in real time
        #    print('Weights: {weights}  Score: {score}'.format(weights=res['x'], score=res['fun']))

        bestSC = np.min(lls)
        bestWght = wghts[np.argmin(lls)]
    #print(bestSC)
    print(bestWght)
    a = 0.0*predictions[0]
    for weight, prediction in zip(bestWght, predictions):
        a += weight*prediction
    return a,bestWght

def gen_type(df):
    hig = .90
    mid = .5
    low = .10
 
    if df[0]>hig and df[1]>hig and df[2]>hig and df[3]>hig and df[4]>hig and df[5]>hig:
        return 1
    if df[0]<low and df[1]<low and df[2]<low and df[3]<low and df[4]<low and df[5]<low:
        return 1

    if np.abs(df['is_iceberg_max']-df['is_iceberg_min'])>.91:
        return 2

    return 0

def gen_test(df,type,sub):
    print('\nprocessing ',sub)
    df['new_pred'] = df[sub]
    #a = df#[df.type==3]
    a = df[df.type==type]
    index = a.index
    print(a.shape)
    print('orig logloss is: ',log_loss(df['is_iceberg'], df[sub]))
    pred = get_minmax(a)
    df.loc[index,'new_pred'] = pred.clip(0.001,0.999)
    print('minmax logloss is: ',log_loss(df['is_iceberg'],df['new_pred']))
    df['new_pred'] = df[sub]
    #print('minmax logloss is: ',log_loss(df['is_iceberg'],df['new_pred']))

    pred = get_mean(a)
    df.loc[index,'new_pred'] = pred.clip(0.001,0.999)
    print('mean logloss is: ',log_loss(df['is_iceberg'],df['new_pred']))
    df['new_pred'] = df[sub]
    #print('mean logloss is: ',log_loss(df['is_iceberg'],df['new_pred']))

    pred = get_median(a)
    df.loc[index,'new_pred'] = pred.clip(0.001,0.999)
    print('median logloss is: ',log_loss(df['is_iceberg'],df['new_pred']))
    df['new_pred'] = df[sub]
    #print('median logloss is: ',log_loss(df['is_iceberg'],df['new_pred']))

    #pred,weights = get_mini(a,[])
    ##df.loc[index,'new_pred'] = pred.clip(0.001,0.999)
    #print('mini logloss is: ',log_loss(df['is_iceberg'],df['new_pred']))
    #df['new_pred'] = df[sub]
    #print('mini logloss is: ',log_loss(df['is_iceberg'],df['new_pred']))

train = pd.read_csv('note/myvalid.csv')
test = pd.read_csv('note/mytest.csv')
train['is_iceberg_max'] = train.iloc[:, :6].max(axis=1)
train['is_iceberg_min'] = train.iloc[:, :6].min(axis=1)
train['is_iceberg_mean'] = train.iloc[:, :6].mean(axis=1)
train['is_iceberg_median'] = train.iloc[:, :6].median(axis=1)
test['is_iceberg_max'] = test.iloc[:, :6].max(axis=1)
test['is_iceberg_min'] = test.iloc[:, :6].min(axis=1)
test['is_iceberg_mean'] = test.iloc[:, :6].mean(axis=1)
test['is_iceberg_median'] = test.iloc[:, :6].median(axis=1)

train['type'] = train.apply(gen_type,1)
test['type'] = test.apply(gen_type,1)

################################
################################
gen_test(train,1,'lr9')
gen_test(train,1,'svm')
gen_test(train,1,'xgb')
gen_test(train,1,'knn')
gen_test(train,1,'avg2')
gen_test(train,1,'mini2')
#gen_test(train,2)



train['new_pred'] = train['mini2']
a = train[train.type==1]
index = a.index
print(a.shape)
pred = get_median(a)
train.loc[index,'new_pred'] = pred.clip(0.001,0.999)
print('max logloss is: ',log_loss(train['is_iceberg'],train['new_pred']))
'''
a = train[train.type==2]
print(a.shape)
#print(a)
index = a.index
#pred = a['mix+lr']
pred,weights2 = get_mini(a,[])
train.loc[index,'new_pred'] = pred.clip(0.001,0.999)
'''

test['new_pred'] = test['mini2']
a = test[test.type==1]
index = a.index
print(a.shape)
pred = get_median(a)
test.loc[index,'new_pred'] = pred.clip(0.001,0.999)
'''
a = test[test.type==2]
print(a.shape)
index = a.index
#pred = a['mix+lr']
pred,_ = get_mini(a,weights2)
test.loc[index,'new_pred'] = pred.clip(0.001,0.999)
'''

sub = test.loc[:,['id','new_pred']]
sub.columns = ['id','is_iceberg']
sub.to_csv('../subm/clever_ensemble/91type+meadian_sub.csv', float_format='%.6f', index=False)


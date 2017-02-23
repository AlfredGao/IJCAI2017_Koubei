# -*- coding: utf-8 -*-
# encoding = uft-8
"""
@Author: Alfred Gao
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.metrics import make_scorer
import sys
sys.path.append('../')
import time
from __utils import cal_offline_score
from __utils import get_result


def calculate_score(pre,real):
    if(len(pre.shape)==1):
        pre = pd.DataFrame(pre,columns=[0])
        real = pd.DataFrame(real,columns=[0])
    else:
        pre = pd.DataFrame(pre,columns=[i for i in range(pre.shape[1])])
        real = pd.DataFrame(real,columns=[i for i in range(real.shape[1])])

    if(len(pre)!=len(real)):
        print 'len(pre)!=len(real)','\n'
    if(len(pre.columns)!=len(real.columns)):
        print 'len(pre.columns)!=len(real.columns)','\n'
    N = len(pre)
    T = len(pre.columns)
    print 'N:',N,'\t','T:',T,'\n'

    n = 0
    t = 0
    L=0

    while(t<T):
        n=0
        while(n<N):
            c_it = round(pre.ix[n,t])
            c_git = round(real.ix[n,t])


            if((c_it==0 and c_git==0) or (c_it+c_git)==0 ):
                c_it=1
                c_git=1

            L = L+abs((float(c_it)-c_git)/(c_it+c_git))
            n=n+1
        t=t+1
    #print L
    return L/(N*T)


ExtraTree = ExtraTreesRegressor()
parameters = {'n_estimators':[1200],'n_jobs':[-1],'random_state':[1],'min_samples_split':[2],\
                'min_samples_leaf':[2],'max_depth':[25],'criterion':['mse'],'max_features':[244]}


train_feature = pd.read_csv('../../train/train_feature_0222.csv')
train_label = pd.read_csv('../../train/train_label.csv')
del train_feature['Unnamed: 0']
del train_label['Unnamed: 0']

train_feature_val = train_feature.values
train_label_val = train_label.values

kf = KFold(len(train_feature_val),n_folds=5,shuffle=True,random_state=1)
loss = make_scorer(calculate_score,greater_is_better=False)
#Grid Search for the best parameters
GridSearchModel = GridSearchCV(ExtraTree,param_grid=parameters,cv=kf,scoring=loss)
GridSearchModel.fit(train_feature_val,train_label_val)
best_p = GridSearchModel.best_params_
print best_p

test_feature = pd.read_csv('../../train/test_feature_0222.csv')
#Del the 'Unnamed:0' col in three tables.
del test_feature['Unnamed: 0']

ExtraTree = ExtraTreesRegressor(n_estimators=best_p['n_estimators'],
                                random_state=best_p['random_state'],
                                n_jobs=best_p['n_jobs'],
                                min_samples_split=best_p['min_samples_split'],
                                min_samples_leaf=best_p['min_samples_leaf'],
                                max_depth=best_p['max_depth'],
                                max_features=best_p['max_features'])

ExtraTree.fit(train_feature, train_label)
prediction = (ExtraTree.predict(test_feature)).round()
result = get_result(prediction)
result = result.astype(int)


timestamp = time.strftime("%y%m%d")
save_submit_path = '../../output/submit_' + timestamp + '.csv'
result.to_csv(save_submit_path,index=False,header=False)

# pre = pd.read_csv('../../output/submit_170221.csv',names=['id','d1','d2','d3','d4','d5','d6','d7','d8','d9','d10','d11','d12','d13','d14'])
# true = pd.read_csv('../../output/submit_170222.csv',names=['id','d1','d2','d3','d4','d5','d6','d7','d8','d9','d10','d11','d12','d13','d14'])
# true['d11'] = (true['d11']*1.2).round()
# true = true.astype(int)
# true.to_csv('submit_170222_double11.csv',index=False, header=False)
# del pre['id']
# del true['id']
#
# print cal_offline_score(np.array(pre), np.array(true))

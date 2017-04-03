# -*- coding: utf-8 -*-
# encoding = uft-8
"""
@Author: Alfred Gao
@Review: Allan
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


class myTreesRegressor():
    def __init__(self):
        self.n_estimators=0
        self.random_state=0
        self.n_jobs=0
        self.min_samples_split=0
        self.min_samples_leaf=0
        self.max_depth=0
        self.max_features=0

    def gridSearch(self, train, train_label, para, isPath=False):
        ExtraTree = ExtraTreesRegressor()
        parameters = para
        import os
        if isPath:
            if os.path.isfile(train) and os.path.isfile(train_label):
                train_feature = pd.read_csv(train)
                train_label = pd.read_csv(train_label)
                del train_feature['Unnamed: 0']
                del train_label['Unnamed: 0']
        else:
            train_feature = train
            train_label = train_label
            del train_feature['Unnamed: 0']
            del train_label['id']


        train_feature_val = train_feature.values
        train_label_val = train_label.values

        kf = KFold(len(train_feature_val),n_folds=5,shuffle=True,random_state=1)
        loss = make_scorer(cal_offline_score,greater_is_better=False)
        #Grid Search for the best parameters
        self.GridSearchModel = GridSearchCV(ExtraTree,param_grid=parameters,cv=kf,scoring=loss)
        self.GridSearchModel.fit(train_feature_val,train_label_val)
        return self.GridSearchModel.best_params_


    def predict(self, train_feature, train_label, predict, best_p, singleOutput = False, isPath=False):
        import os
        if isPath:
            if os.path.isfile(data):
                predict_feature = pd.read_csv(data)
        else:
            predict_feature = predict
# test_feature = pd.read_csv('../../train/test_feature_0222.csv')
#Del the 'Unnamed:0' col in three tables.

        ExtraTree = ExtraTreesRegressor(n_estimators=best_p['n_estimators'],
                                        random_state=best_p['random_state'],
                                        n_jobs=best_p['n_jobs'],
                                        min_samples_split=best_p['min_samples_split'],
                                        min_samples_leaf=best_p['min_samples_leaf'],
                                        max_depth=best_p['max_depth'],
                                        max_features=best_p['max_features'])

        ExtraTree.fit(train_feature, train_label)
        result = (ExtraTree.predict(test_feature)).round()
        if not singleOutput:
            result = get_result(result)
            timestamp = time.strftime("%y%m%d")
            save_submit_path = '../../output/submit_' + timestamp + '.csv'
            result.to_csv(save_submit_path,index=False,header=False)
        result = result.astype(int)
        return result

# pre = pd.read_csv('../../output/submit_170221.csv',names=['id','d1','d2','d3','d4','d5','d6','d7','d8','d9','d10','d11','d12','d13','d14'])
# true = pd.read_csv('../../output/submit_170222.csv',names=['id','d1','d2','d3','d4','d5','d6','d7','d8','d9','d10','d11','d12','d13','d14'])
# true['d11'] = (true['d11']*1.2).round()
# true = true.astype(int)
# true.to_csv('submit_170222_double11.csv',index=False, header=False)
# del pre['id']
# del true['id']
#
# print cal_offline_score(np.array(pre), np.array(true))

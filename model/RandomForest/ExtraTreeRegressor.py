import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.metrics import make_scorer
import sys
sys.path.append('../')
import time
from __utils import cal_offline_score
from __utils import get_result


ExtraTree = ExtraTreesRegressor()
parameters = {'n_estimators':[1200],'n_jobs':[-1],'random_state':[1],'min_samples_split':[2],\
                'min_samples_leaf':[2],'max_depth':[25],'criterion':['mse'],'max_features':[237]}


train_feature = pd.read_csv('../../train/train_feature.csv')
train_label = pd.read_csv('../../train/train_label.csv')


train_feature_val = train_feature.values
train_label_val = train_label.values


kf = KFold(len(train_feature_val),n_folds=5,shuffle=True,random_state=1)
loss = make_scorer(cal_offline_score,greater_is_better=False)
#Grid Search for the best parameters
GridSearchModel = GridSearchCV(ExtraTree,param_grid=parameters,cv=kf,scoring=loss)
GridSearchModel.fit(train_feature_val,train_label_val)
best_p = GridSearchModel.best_params_
print best_p

test_feature = pd.read_csv('../../train/test_feature.csv')
ExtraTree = ExtraTreesRegressor(n_estimators=int(best_p['n_estimators']),\
random_state=(best_p['random_state']),n_jobs=int(best_p['n_jobs']),min_samples_split=int(best_p['min_samples_split']),\
min_samples_leaf=int(best_p['min_samples_leaf']),max_depth=int(best_p['max_depth']),\
max_features=int(best_p['max_features']))
ExtraTree.fit(train_feature, train_label)
prediction = ExtraTree.predict(test_feature).round()
result = get_result(prediction)
result = result.astype(int)


timestamp = time.strftime("%y%m%d")
save_submit_path = '../../output/submit_' + timestamp + '.csv'
result.to_csv(save_submit_path,index=False,header=False)

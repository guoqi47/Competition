import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn import utils

data = pd.read_csv('PINGAN-2018-train_demo.csv')
train = data.copy()
test = data.copy()[68000:]
features = ['LONGITUDE','LATITUDE','DIRECTION','HEIGHT','SPEED','CALLSTATE']
target = ['Y']
train['Y'] = utils.multiclass.type_of_target(train['Y'].astype('int'))

clf = lgb.LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=80, n_jobs=20)
clf.fit(train[features], train[target], feature_name=features)
test['lgb_predict'] = clf.predict_proba(test[features],)[:, 1]
#print(log_loss(test[target], test['lgb_predict']))
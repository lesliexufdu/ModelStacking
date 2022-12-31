from sklearn.model_selection import StratifiedKFold
from sklearn.base import TransformerMixin
from copy import deepcopy
import numpy as np; import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestNeighbors
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

class classifierForStacking():
    '''
    将分类器用K折交叉验证包装用以Stacking
    初始参数:
        clf--基础分类器,需包含fit和predict_proba方法(返回2维矩阵)
        nFolds--交叉验证参数
    方法:
        fit_transform:交叉验证的拟合-转化过程,用于后续的fit
        transform:单纯的多次拟合取平均,用于最终stacking的predict
    '''
    def __init__(self, clf, nFolds = 5):
        self.__basicClassifier = [deepcopy(clf) for _ in range(nFolds)]
        self.nFolds = nFolds

    def fit_transform(self, X, y):
        if isinstance(X, pd.DataFrame):
            X_copy = X.values
        else:
            X_copy = X.copy()
        if self.nFolds==1:
            self.__basicClassifier[0].fit(X_copy, y)
            outArray = self.__basicClassifier[0].predict_proba(X_copy)[:,1]
        else:
            skf = StratifiedKFold(n_splits=self.nFolds, shuffle=True)
            outArray = np.zeros(y.shape[0])
            for i, (train_idx, test_idx) in enumerate(skf.split(X_copy, y)):
                self.__basicClassifier[i].fit(X_copy[train_idx], y[train_idx])
                outArray[test_idx] = self.__basicClassifier[i].\
                                     predict_proba(X_copy[test_idx])[:,1]
        return outArray

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_copy = X.values
        else:
            X_copy = X.copy()
        outArray = np.zeros(X.shape[0])
        for i in range(self.nFolds):
            outArray += self.__basicClassifier[i].predict_proba(X_copy)[:,1]
        return outArray/self.nFolds

class lsvcSimple(TransformerMixin, LinearSVC):
    '''
    简化的线性SVM模型
    '''
    def predict_proba(self, X):
        return super().predict(X).reshape([-1,1]).repeat(2,1)

class knnSimple(NearestNeighbors):
    '''
    简化的最近邻,返回n个近邻中0/1的占比
    注意:fit方法的y必须是np数组而非Series
    '''
    def fit(self, X, y):
        super().fit(X)
        self.yDistribution = y
        return self
    def predict_proba(self, X):
        out = np.zeros([X.shape[0],2])
        for i in range(X.shape[0]):
            nnRes = self.kneighbors(X[i:i+1])[1]
            nnRes = self.yDistribution[nnRes]
            out[i,1] = nnRes.mean()
            out[i,0] = 1 - out[i,1]
        return out

class lgbSimple:
    def __init__(self, n_jobs = 10, categoricalFeatures = [], metric = 'auc', test_size = 0.3,
                 stoppingRounds = 20, numBoosting = 300, objective = 'binary',
                 paramsSearch = [{'num_leaves':np.arange(8,25,2),
                                 'min_child_samples':np.arange(700,1600,100),
                                 'learning_rate':np.arange(0.6,1.4,0.1)},
                                 {'reg_alpha':np.logspace(-6,6,13),
                                 'reg_lambda':np.logspace(-6,6,13)}],
                 ):
        '''
        集成参数自搜索的LGB
        '''
        self.n_jobs = n_jobs
        self.catFeatures = categoricalFeatures
        self.metric = metric
        self.numBoosting = numBoosting
        self.objective = objective
        self.test_size = test_size
        self.stoppingRounds = stoppingRounds
        self.paramsSearch = paramsSearch

    def fit(self, X, y):
        lgb_estimator = LGBMClassifier(objective = self.objective, n_jobs=self.n_jobs,
                                       n_estimators=self.numBoosting)
        features_train,features_test,labels_train,labels_test = train_test_split(X,y)
        clf = GridSearchCV(lgb_estimator, self.paramsSearch)
        clf.fit(features_train, labels_train, eval_set=[(features_test, labels_test)],
                eval_names=['test'],eval_metric=self.metric, verbose = False,
                categorical_feature=self.catFeatures,early_stopping_rounds=self.stoppingRounds)
        self.estimator = clf.best_estimator_

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

class xgbSimple:
    def __init__(self, n_jobs = 10, metric = 'auc', test_size = 0.3,
                 stoppingRounds = 20, numBoosting = 300, objective = 'binary:logistic',
                 paramsSearch = [{'max_depth':np.arange(3,7,1),
                                 'min_child_weight':np.arange(1,6,1),
                                 'learning_rate':np.arange(0.6,1.4,0.1)},
                                 {'gamma':np.linspace(0.0,0.5,6),
                                 'subsample':np.linspace(0.6,1.0,6),
                                 'colsample_bytree':np.linspace(0.6,1.0,5)}],
                 ):
        '''
        集成参数自搜索的XGB
        '''
        self.n_jobs = n_jobs
        self.metric = metric
        self.numBoosting = numBoosting
        self.objective = objective
        self.test_size = test_size
        self.stoppingRounds = stoppingRounds
        self.paramsSearch = paramsSearch

    def fit(self, X, y):
        lgb_estimator = XGBClassifier(objective = self.objective, n_jobs=self.n_jobs,
                                       n_estimators=self.numBoosting)
        features_train,features_test,labels_train,labels_test = train_test_split(X,y)
        clf = GridSearchCV(lgb_estimator, self.paramsSearch)
        clf.fit(features_train, labels_train, eval_set=[(features_test, labels_test)],
                eval_metric=self.metric, early_stopping_rounds=self.stoppingRounds,
                verbose = False)
        self.estimator = clf.best_estimator_

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
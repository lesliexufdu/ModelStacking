import numpy as np; import pandas as pd
from copy import deepcopy

class pipeForStacking():
    '''
    stacking平行结构,每个子转换器需要有fit_transform方法和transform方法
    调用fit和predict方法意味着此pipe是根Stacking框架,此时末端分类器要有fit和predict_proba方法
    '''
    def __init__(self, *classifiers):
        self.__basicClassifiers = classifiers

    def fit(self, X, y):
        out = X.copy()
        for clf in self.__basicClassifiers[:-1]:
            out = clf.fit_transform(out, y)
        self.__basicClassifiers[-1].fit(out, y)

    def fit_transform(self, X, y):
        out = X.copy()
        for clf in self.__basicClassifiers:
            out = clf.fit_transform(out, y)
        return out

    def transform(self, X):
        out = X.copy()
        for clf in self.__basicClassifiers:
            out = clf.transform(out)
        return out

    def predict(self, X):
        out = X.copy()
        for clf in self.__basicClassifiers[:-1]:
            out = clf.transform(out)
        return self.__basicClassifiers[-1].predict_proba(out)[:,1]

class unionForStacking():
    '''
    stacking并行结构,每个子转换器需要有fit_transform方法和transform方法
    '''
    def __init__(self, *transformers):
        self.__basicTransformers = transformers

    def fit_transform(self, X, y):
        out = []
        for clf in self.__basicTransformers:
            out.append( clf.fit_transform(X, y).reshape([X.shape[0],-1]) )
        return np.hstack(out)

    def transform(self, X):
        out = []
        for clf in self.__basicTransformers:
            out.append( clf.transform(X).reshape([X.shape[0],-1]) )
        return np.hstack(out)

class colSamplingForStacking:
    '''
    stacking中提供列抽样
    初始化参数: 基础分类器(有fit_transform方法和transform方法), 采样比例, 采样次数, 是否放回
    注意: 无放回情况下采样比例*采样次数应当小于1
    '''
    def __init__(self, classifier, colSamplingRate = 0.8, cycles = 10, replace = True):
        self.samplingRate = colSamplingRate
        self.estNum = cycles
        self.replace = replace
        self.__basicClassifier = [deepcopy(classifier) for _ in range(cycles)]

    def fit_transform(self, X, y):
        X_copy = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols = X_copy.columns
        featuresNum = int(self.samplingRate*len(cols))
        colSampled = np.random.choice(cols, replace=self.replace,
                     size = featuresNum * self.estNum)
        self.colSampled = [list(set(colSampled[i*featuresNum:(i+1)*featuresNum]))\
                           for i in range(self.estNum)]
        outArray = []
        for i, col in enumerate(self.colSampled):
            outArray.append(self.__basicClassifier[i].fit_transform(X_copy[col], y).\
                            reshape([X_copy.shape[0],-1]))
        return np.hstack(outArray)

    def transform(self, X):
        X_copy = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        outArray = []
        for i, col in enumerate(self.colSampled):
            outArray.append(self.__basicClassifier[i].transform(X_copy[col]).\
                            reshape([X_copy.shape[0],-1]))
        return np.hstack(outArray)

class observationSamplingForStacking:
    '''
    stacking中提供观测抽样
    初始化参数: 基础分类器(有fit_transform方法和transform方法), 采样比例, 采样次数, 是否放回
    '''
    def __init__(self, classifier, SamplingRate = 0.8, cycles = 10):
        self.samplingRate = SamplingRate
        self.estNum = cycles
        self.__basicClassifier = [deepcopy(classifier) for _ in range(cycles)]

    def fit_transform(self, X, y):
        X_copy = X if isinstance(X,np.ndarray) else X.values
        y_copy = y if isinstance(y,np.ndarray) else y.values
        samplesNumTot = X_copy.shape[0]
        observationNum = int(self.samplingRate * samplesNumTot)
        observationSampled = np.random.choice(range(samplesNumTot),
                                              size = observationNum * self.estNum)
        self.observationSampled=[np.unique(observationSampled[i*observationNum:(i+1)*observationNum])\
                                   for i in range(self.estNum)]
        outArray = []
        for i, idx in enumerate(self.observationSampled):
            self.__basicClassifier[i].fit_transform(X_copy[idx], y_copy[idx])
            outArray.append(self.__basicClassifier[i].transform(X_copy).reshape([samplesNumTot,-1]))
        return np.hstack(outArray)

    def transform(self, X):
        X_copy = X if isinstance(X,np.ndarray) else X.values
        samplesNumTot = X_copy.shape[0]; outArray = []
        for i, idx in enumerate(self.observationSampled):
            outArray.append(self.__basicClassifier[i].transform(X_copy).reshape([samplesNumTot,-1]))
        return np.hstack(outArray)
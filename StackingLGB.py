import numpy as np
from ModelFitting.models import classifierForStacking, lgbSimple, xgbSimple
from ModelFitting.pipeAndUnion import pipeForStacking, unionForStacking, colSamplingForStacking
from sklearn.linear_model import LogisticRegressionCV
from DataPreProcessing.BinEncoding import BinEncodingProb, SpecialValueEncoding, scoreSplitBins
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier

if __name__=="__main__":
    #%%第一层
    ###树模型;catboost不能代入,目前原因不知
    lgb_1 = classifierForStacking(lgbSimple(paramsSearch = [{'num_leaves':np.arange(6,15,3),
                                 'min_child_samples':np.arange(700,1600,200),
                                 'learning_rate':[0.1]}]))
    lgbColSampling = colSamplingForStacking(lgb_1,
                     colSamplingRate=0.8, cycles=3)

    xgb_1 = classifierForStacking(xgbSimple(paramsSearch = [{'max_depth':np.arange(3,7,1),
                                 'min_child_weight':np.arange(1,6,2),'learning_rate':[0.1],
                                 'gamma':np.linspace(0.0,0.5,4)}]))
    xgbColSampling = colSamplingForStacking(xgb_1,
                     colSamplingRate=0.8, cycles=3)
    #mlp_1:分组转换-20|50|20感知机
    mlp_1 = pipeForStacking(SpecialValueEncoding(specialValueList=[-9999999,-9999998,
                                -9999997,-8888888, -8888887, -8888885, -7777777]),
                            MinMaxScaler(feature_range=(0.3,0.7)),
                            classifierForStacking(MLPClassifier(
                                activation='relu', hidden_layer_sizes=(10,10)))
                            )
    mlp_1 = colSamplingForStacking(mlp_1, colSamplingRate=0.7,
            cycles=3)
    #%%第二层:汇总LR
    level1_tot = unionForStacking(lgbColSampling, xgbColSampling, mlp_1)
    final_lr = pipeForStacking(level1_tot,
                               scoreSplitBins(),
                               LogisticRegressionCV()
                               )

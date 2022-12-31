from DataPreProcessing.BinEncoding import BinEncodingProb, SpecialValueEncoding
from DataPreProcessing.BinEncoding import sqrtTransform, logTransform
from ModelFitting.models import classifierForStacking, lsvcSimple, knnSimple
from ModelFitting.pipeAndUnion import pipeForStacking, unionForStacking
from sklearn.preprocessing import MinMaxScaler,StandardScaler,QuantileTransformer
from sklearn.decomposition import PCA, NMF
from sklearn.linear_model import LogisticRegressionCV
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier

if __name__=="__main__":
    #%%第一层
    ###逻辑回归相关
    #lr_1:特殊值均值转换-逻辑回归
    lr_1 = pipeForStacking(SpecialValueEncoding(specialValueList=[-9999999,-9999998,
                           -9999997,-8888888, -8888887, -8888885, -7777777]),
                           classifierForStacking(LogisticRegressionCV()))
    #lr_3:特殊值均值转换-最大绝对值归一化-log坐标转换-逻辑回归
    lr_3 = pipeForStacking(SpecialValueEncoding(specialValueList=[-9999999,-9999998,
                            -9999997, -8888888, -8888887, -8888885, -7777777]),
                         MinMaxScaler(feature_range=(0.3,0.7)),
                         logTransform(),
                         classifierForStacking(LogisticRegressionCV()))
    #lr_4:分组概率转换-逻辑回归
    lr_4 = pipeForStacking(BinEncodingProb(),
                           classifierForStacking(LogisticRegressionCV()))
    #lr_5:分组概率转换-sqrt坐标转换-逻辑回归
    lr_5 = pipeForStacking(BinEncodingProb(),
                           sqrtTransform(),
                           classifierForStacking(LogisticRegressionCV()))

    ###树模型;catboost不能代入,目前原因不知
    lgb_1 = classifierForStacking(LGBMClassifier(num_leaves=10, n_estimators=100,
                                      min_child_samples=1000, n_jobs=5))
    xgb_1 = classifierForStacking(XGBClassifier(max_depth=3, learning_rate=0.1,
                                      n_estimators=100, n_jobs=5))

    ###简单神经网络
    #mlp_1:特殊值均值转换-标准化-20|50|20感知机
    mlp_1 = pipeForStacking(SpecialValueEncoding(specialValueList=[-9999999,
                                      -9999998,-9999997, -8888888, -8888887,
                                      -8888885, -7777777]),
                                  StandardScaler(),
                                  classifierForStacking(MLPClassifier(
                                      activation='relu', hidden_layer_sizes=(20,50,20)))
                                  )
    #mlp_2:概率映射-20|20感知机
    mlp_2 = pipeForStacking(BinEncodingProb(),
                          classifierForStacking(MLPClassifier(activation='tanh',
                                                    hidden_layer_sizes=(20,20)))
                          )

    ###极端树/随机森林
    rf = classifierForStacking(RandomForestClassifier(max_depth=3, n_estimators=10,
                                   min_weight_fraction_leaf=0.05))
    et = classifierForStacking(ExtraTreesClassifier(max_depth=3, n_estimators=10,
                                   min_weight_fraction_leaf=0.05))

    ###KNN
    #knn_1:概率映射-KNN
    knn_1 = pipeForStacking(BinEncodingProb(),
                            classifierForStacking(knnSimple(13)))
    #knn_2:特殊值均值转换-最大最小值归一化-NMF(8)-KNN
    knn_2 = pipeForStacking(SpecialValueEncoding(specialValueList=[-9999999,-9999998,
                                -9999997,-8888888, -8888887, -8888885, -7777777]),
                            MinMaxScaler(feature_range=(0.3,0.7)),
                            NMF(n_components=8),
                            classifierForStacking(knnSimple(13)))

    ###线性SVM
    #svc_2:特殊值均值转换-最大最小值归一化-svc
    svc_2 = pipeForStacking(SpecialValueEncoding(specialValueList=[-9999999,-9999998,
                                -9999997,-8888888,-8888887,-8888885,-7777777]),
                             MinMaxScaler(feature_range=(0.3,0.7)),
                             classifierForStacking(lsvcSimple()))

    #%%第二层:汇总LR
    level1_tot = unionForStacking(lr_1, lr_4, lgb_1, xgb_1, mlp_1, mlp_2,
                                  rf, et, knn_1, knn_2, svc_2,lr_3,lr_5)
    from sklearn.linear_model import LogisticRegressionCV
    final_lr = pipeForStacking(level1_tot,
                               classifierForStacking(LGBMClassifier(num_leaves=10,
                                    n_estimators=100,min_child_samples=1000,n_jobs=5))
                               )

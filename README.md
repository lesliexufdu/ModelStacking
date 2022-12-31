# model-stacking、

模型Stacking的Pipeline，实现多个模型的并行和串行。

## 使用

`DataPreProcessing/BinEncoding.py`中是常见的数据预处理方式，包括：

- SpecialValueEncoding：将数据集中的特殊值转化成对应的值，方法是将非特殊值分段，找出与特殊值表现最接近的分段，将特殊值替换为该分段的均值或中位数或众数
- BinEncodingProb：将自变量分组后映射为对应分组因变量的均值，适用于二分类
- logTransform：按$f(x) = ln(1 + x)$转化原数值, 要求原值满足$x>=-1$
- sqrtTransform：按$f(x) = \sqrt{1 + x}$转化原数值, 要求原值满足$x>=-1$
- scoreSplitBins：按分位数分段后转换成分段的序号

以上类都有fit和transform方法。

`ModelFitting/models.py`中是用于Stacking的基础模型，主要包括一个包装器：

- classifierForStacking：将分类器用K折交叉验证包装用以Stacking，包含fit_transform和transform两个方法，前者在训练过程使用，交叉训练并以验证集结果作为模型结果进入后续的Stacking，后者用在预测时，是单纯的多次拟合取平均

classifierForStacking是适用于二分类的Stacking包装类，被包装的类必须有fit和predict_proba方法。此外还有lsvcSimple类(简化的线性SVM模型)、knnSimple类(简化的最近邻)、lgbSimple类(包含GridSearch的LightGBM)、xgbSimple类(包含GridSearch的Xgboost)。

`ModelFitting/pipeAndUnion.py`主要包含pipeline的封装：

- pipeForStacking：Stacking的串行，每个子转换器需要有fit_transform方法和transform方法。如果用于根Stacking(根Stacking一般都是串行的)，可调用fit和predict方法,此时末端分类器必须有fit和predict_proba方法。(末端分类器不需要交叉验证取验证数据)
- unionForStacking：Stacking的并行，每个子转换器需要有fit_transform方法和transform方法
- colSamplingForStacking：为基础类提供的列抽样包装器，基础分类器须有fit_transform方法和transform方法
- observationSamplingForStacking：为基础类提供的观测抽样包装器，基础分类器须有fit_transform方法和transform方法

`StackingNormal.py`中是具体的一个使用案例。

## 维护者

[@lesliexufdu](https://github.com/lesliexufdu)

## License

[MIT](LICENSE) © Leslie Xu
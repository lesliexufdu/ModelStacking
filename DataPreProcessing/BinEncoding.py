import pandas as pd; import numpy as np
from sklearn.base import TransformerMixin
from sklearn import tree
import re

class SpecialValueEncoding(TransformerMixin):
    '''
    针对规定的特殊值,变量分组后转化成对应的均值或中位数或众数
    '''
    def __init__(self, depth=None, min_pct=0.05, cri='gini',
                 specialValueList = [-9999999], replaceMode = 'mean'):
        '''
        depth:决策树分组的最大深度
        min_pct:叶子节点的最少样本占比
        specialValueList:特殊值列表
        replaceMode:均值、中位数或者众数('mean'/'median'/'mode')
        '''
        self.depth = depth
        self.min_pct = min_pct
        self.cri = cri
        self.spVL = specialValueList
        self.replaceMode = replaceMode
    
    def __tree_parser(self, clf):#解析决策树节点,按顺序排列
        tr = tree.export_graphviz(clf,out_file=None)
        tr = tr.split('\n'); sp_value = []
        pattern = re.compile(r'X\[0\] <= (-?\d+\.?\d*)')
        for i in tr:
            mm = pattern.search(i)
            if mm:
                sp_value.append(float(mm.groups()[0]))
        sp_value.sort()
        return sp_value
        
    def fit(self, data, target):
        #按输入数据决定特殊值归类
        def sp_value_select(sp_list):
            nonlocal dataNotSpecial, targetNotSpecial, self
            tmp_data = np.zeros(dataNotSpecial.shape[0])
            for j in range(len(sp_list) - 1):
                idx = dataNotSpecial.values <= sp_list[j]
                if self.replaceMode=='mean':
                    replaceValue = dataNotSpecial[idx].mean()
                elif self.replaceMode=='median':
                    replaceValue = dataNotSpecial[idx].median()
                elif self.replaceMode=='mode':
                    replaceValue = dataNotSpecial[idx].mode()
                else:
                    raise TypeError('replaceMode should be one in [\'mean\',\'median\',\'mode\']')
                tmp_data[idx] = replaceValue
            tmp = pd.DataFrame({'data':tmp_data,'label':targetNotSpecial},dtype='int')
            tmpGroup = tmp.groupby('data')
            return tmpGroup.sum()/tmpGroup.count()
    
        min_data = int(self.min_pct * data.shape[0])
        self.valueDct = {}
        data_copy = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        for col in data_copy.columns:
            dataSpecial = data_copy[col].isin(self.spVL)#是否为特殊值的bool型
            if dataSpecial.any():#存在特殊值时先计算菲特树值分组概率
                dataNotSpecial = data_copy[col][~dataSpecial]
                targetNotSpecial = target[~dataSpecial]
                clf = tree.DecisionTreeClassifier(max_depth = self.depth,
                                              min_samples_leaf=min_data,
                                              criterion=self.cri)
                clf.fit(dataNotSpecial.to_frame(), targetNotSpecial)
                tmp_list = self.__tree_parser(clf)
                notSpecialGroup = sp_value_select(tmp_list)
                self.valueDct[col] = {}#初始化特殊值转化字典
                #遍历特殊值寻找最近的值
                for value in self.spVL:
                    idx = data_copy[col]==value
                    if idx.any():
                        specialValuePercent = target[idx].sum()/target[idx].shape[0]
                        valueToReplace = np.abs(specialValuePercent - notSpecialGroup).idxmin()[0]
                        self.valueDct[col][value] = valueToReplace
        return self

    def transform(self, data):
        #为新数据集做转换
        dataCopy = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        for col, replaceDict in self.valueDct.items():
            dataCopy[col].replace(replaceDict, inplace=True)                
        return dataCopy.values

class BinEncodingProb(TransformerMixin):
    '''
    变量分组后转化为对应的概率值
    '''
    def __init__(self, depth=None, min_pct=0.03, cri='gini', woe_thre = 0.1, only_group = False):
        '''
        初始化参数
            depth:决策树分组的最大深度
            min_pct:叶子节点的最少样本占比
        '''
        self.depth = depth
        self.min_pct = min_pct
        self.cri = 'gini'
        self.woe_thre = woe_thre
        self.only_group = only_group
        
    def __prob_trans(self, X, y=None, repl = None):
        #输入分类变量与target,转换成每一类中1的概率
        WOE_tr = np.zeros(X.shape)
        if repl:
            for i,repl_col in enumerate(repl):
                WOE_tr[:,i] = pd.Series(X[:,i]).replace(repl_col)
            return WOE_tr
        else:
            repl = []
            for i in range(X.shape[1]):
                col = pd.DataFrame({'data':X[:,i],'label':y})
                repl_col = col.groupby('data')['label'].mean().to_dict()
                repl.append(repl_col)
                WOE_tr[:,i] = pd.Series(X[:,i]).replace(repl_col)
            return WOE_tr, repl
    
    def __tree_parser(self, clf):
        tr = tree.export_graphviz(clf,out_file=None)
        tr = tr.split('\n'); sp_value = set()
        pattern = re.compile(r'X\[0\] <= (-?\d+\.?\d*)')
        for i in tr:
            mm = pattern.search(i)
            if mm:
                sp_value.add(float(mm.groups()[0]))
        sp_value = list(sp_value)
        sp_value.sort()
        return sp_value
        
    def __tree_split(self, data, target):
        '''
        将数据中数值变量按决策树寻找分裂点,
        返回字典在属性split_dct里
        '''
        def sp_value_select(sp_list, col):
            #删除相邻区间WOE差值<0.1的分割点
            nonlocal data, target, bad_tot, good_tot, self
            tmp_data = np.zeros(data.shape[0])
            for j in sp_list:
                tmp_data += (data[col] > j).astype(int)
            tmp = pd.DataFrame({'data':tmp_data,'label':target},dtype='int')
            bad_cnt = tmp.groupby('data').sum()
            per = np.log( (tmp.groupby('data').count()-bad_cnt)*bad_tot/\
                         (bad_cnt * good_tot) )
            woe_diff = np.abs(per['label'].values[:-1] - per['label'].values[1:])
            self.split_dct[col] = [sp_list[i] for i in np.where(
                np.bitwise_and(woe_diff>self.woe_thre,woe_diff!=np.inf))[0]]
    
        min_data = int(self.min_pct * data.shape[0])
        self.split_dct = {}
        bad_tot = target.sum(); good_tot = target.shape[0]-bad_tot
        for i in data.columns:
            if i not in self.cat_f:
                clf = tree.DecisionTreeClassifier(max_depth = self.depth,
                                                  min_samples_leaf=min_data,
                                                  criterion=self.cri)
                clf.fit(data[[i]],target)
                tmp_list = self.__tree_parser(clf)
                sp_value_select(tmp_list, i)
            else:
                self.split_dct[i] = [ [i] for i in data[i].unique() ]
    
    def __data_to_group(self, data):
        group_data = np.zeros(data.shape)
        for i,col in enumerate(data.columns):
            if col in self.cat_f:                
                group_data[:,i] = data[col].replace({si:idx for idx,\
                    item in enumerate(self.split_dct[col]) for si in item})
            else:
                group_data[:,i] = pd.cut(data[col],
                          bins=[-np.inf]+self.split_dct[col]+[np.inf],
                          labels = range(len(self.split_dct[col])+1) )
        return group_data

    def fit(self, data, target, str_col=[]):
        '''
        数据处理与LR拟合,data需DataFrame,分类变量已转化成数字表示
        属性cat_f存储分类变量名
        属性repl_list储存分组-概率值转换字典
        '''
        dataCopy = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        self.cat_f = str_col
        self.__tree_split(dataCopy, target)
        group_data = self.__data_to_group(dataCopy)
        if not self.only_group:
            WOE_data, self.repl_list = self.__prob_trans(group_data, y=target)
        return self
    
    def transform(self, data):
        dataCopy = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        group_data = self.__data_to_group(dataCopy)
        if self.only_group:
            return group_data
        WOE_tr = self.__prob_trans(group_data, repl=self.repl_list)
        return WOE_tr

class logTransform(TransformerMixin):
    '''
    ln(1 + x)转化, 要求原值全部>=-1
    '''
    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        return np.log(1.00000001 + X)
    
class sqrtTransform(TransformerMixin):
    '''
    sqrt(1 + x)转化, 要求原值全部>=-1
    '''
    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        return np.sqrt(1.00000001 + X)
		
class scoreSplitBins(TransformerMixin):
	'''等样本分割分数,q为最多分段数'''
	def __init__(self, q = 20):
		self.binNum = q
		
	def qcut(self, dataScore, q):
		try:
			scoreBins,bins = pd.qcut(dataScore,q,labels=range(1,self.binNum+1),retbins=True)
		except:
			scoreBins,bins = self.qcut(dataScore,q-1)
		bins[0] = -np.inf;bins[-1] = np.inf
		return bins
		
	def fit(self, data, y=None):
		if isinstance(data, pd.DataFrame):
			data = data.values
		self.colBinsDict = {}
		for col in range(data.shape[1]):
			binTemp = self.qcut(data[:,col],self.binNum)
			self.colBinsDict[col] = binTemp
		return self
	
	def transform(self, data):
		if isinstance(data, pd.DataFrame):
			data = data.values
		dataRes = np.zeros(data.shape)
		for col in range(data.shape[1]):
			binTemp = self.colBinsDict[col]
			dataRes[:,col] = pd.cut(data[:,col],bins=binTemp,labels=range(1,len(binTemp)))
		return dataRes
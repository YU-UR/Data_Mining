import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

warnings.filterwarnings('ignore')
pd.options.display.max_rows = None
pd.options.display.max_columns = None

# 设置字段名，原始数据重没有每一维数据代表的含义
column_names = ['Sample code number', 'clump thickness', 'uniformity of cell size', \
                'uniformity of cell shape', 'marginal adhesion', \
                'single epithelial cell size', 'bare nuclei', 'bland chromatin', \
                'normal nucleoli', 'mitnoses', 'class']
data = pd.read_csv("breast-cancer-wisconsin.data", header=None, names=column_names)
# 显示前四条记录
# print(data.head(4))
# 显示数据维度（几行几列）
# print(data.shape)
# 查看数据信息(列名，count 类型以及有缺省值)
# print(data.info())
# # 显示数据统计描述（count,mean,std,min,25%,50%,75%,max）
print(data.describe())
# # 显示数据分布情况(类别数，各类别内数据量)
# print(data.groupby('class').size())
#
# # 缺失值处理，在bare nuclei列中有缺失值，以？表示的
# '''
# mean_value = data[data["bare nuclei"]!="?"]["bare nuclei"].astype(np.int).mean()# 计算异常值列的平均值
# data = data.replace('?',mean_value)
# data["bare nuclei"] = data["bare nuclei"].astype(np.int64)
# '''
# data = data.replace(to_replace='?', value=np.nan)  # 非法字符的替代
# data = data.dropna(how='any')
# print(data.head(25))
#
# 拆分数据集
# 将样本按照3:1划分训练集和测试集,参数解释：
# train_data：所要划分的样本特征集 1~9为特征值
# train_target：所要划分的样本结果 10表示label
# test_size：样本占比，如果是整数的话就是样本的数量
# random_state：是随机数的种子。
X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]], test_size=0.20,
                                                    random_state=5)
# 查看训练样本的数量和类别分布
print(y_train.value_counts())
print(y_test.value_counts())
# print(set(y_train))

# # 标准化数据，保证每一个维度的特征数据方差为1，均值为0.使得预测结果不会被某些维度过大的特征值所主导
# ss = StandardScaler()
# X_train = ss.fit_transform(X_train)
# X_test = ss.transform(X_test)
#
# # 模型一：线性分类模型逻辑回归
# # LR = LogisticRegression()
# LR = LogisticRegression(C=1.0, penalty='l2', tol=0.1)
# LR.fit(X_train, y_train)
# LR_y_predict = LR.predict(X_test)
# # 使用线性分类模型中的classification_report模块对肿瘤预测模型的性能进行分析
# print('预测结果为:{}'.format(LR.score(X_test, y_test)))
# print('预测结果为:{}'.format(accuracy_score(y_test, LR_y_predict)))
# # 利用classification_report模块获得LR的其他三个指标（pression,recall,f1 score）
# print(classification_report(y_test, LR_y_predict, target_names=['Bebign', 'Malignat']))
#
# # 模型二：线性分类模型SGD
# SGDC = SGDClassifier()
# SGDC.fit(X_train, y_train)
# SGDC_y_predict = SGDC.predict(X_test)
# print('Accuracy Of SGD:', SGDC.score(X_test, y_test))
# print(classification_report(y_test, SGDC_y_predict, target_names=['Bebign', 'Malignat']))
#
# # 多种模型比较
# models = {}
# models['LR'] = LogisticRegression()
# models['LDA'] = LinearDiscriminantAnalysis()
# models['KNN'] = KNeighborsClassifier()
# models['CART'] = DecisionTreeClassifier()
# models['NB'] = GaussianNB()
# models['SVM'] = SVC()
#
# num_folds = 10
# seed = 7
# kfold = KFold(n_splits=num_folds, random_state=seed,shuffle=True)
# # 评估算法
# results = []
# for name in models:
#     result = cross_val_score(models[name], X_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(result)
#     msg = '%s: %.3f (%.3f)' % (name, result.mean(), result.std())
#     print(msg)
#
# # 图表显示
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(models.keys())
# plt.show()

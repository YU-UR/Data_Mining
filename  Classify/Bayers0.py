import numpy
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, KFold, cross_val_score

warnings.filterwarnings('ignore')
pd.options.display.max_rows = None
pd.options.display.max_columns = None

# 加载数据集
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
# column_names = [
#     'ID', 'Clump_Thickness', 'Uniformity_Of_Cell_Size', 'Uniformity_Of_Cell_Shape',
#     'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin',
#     'Normal_Nuclei', 'Mitoses', 'Class'
# ]
# data = pd.read_csv(url, header=None, names=column_names)
#
# # 分离特征和标签
# X = data.iloc[:, 1:]  # 所有特征
# y = data['Class']  # 目标变量
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
column_names = ['Sample code number', 'clump thickness', 'uniformity of cell size', 'uniformity of cell shape',
                'marginal adhesion',
                'single epithelial cell size', 'bare nuclei', 'bland chromatin', 'normal nucleoli', 'mitnoses', 'class']
# data = pd.read_csv(url, header=None, names=column_names)
data = pd.read_csv("breast-cancer-wisconsin.data", header=None, names=column_names)
# print(data.head(3))
# print(data['bare nuclei'])
mean_value = data[data["bare nuclei"] != "?"]["bare nuclei"].astype(np.int64).mean()  # 计算异常值列的平均值
data = data.replace('?', mean_value)
data["bare nuclei"] = data["bare nuclei"].astype(np.int64)
print('---------------------------')
# 拆分数据集
# 将样本按照3:1划分训练集和测试集,参数解释：
# train_data：所要划分的样本特征集 1~9为特征值
# train_target：所要划分的样本结果 10表示label
# test_size：样本占比，如果是整数的话就是样本的数量
# random_state：是随机数的种子。
X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]], test_size=0.2,
                                                    random_state=5)
# X_train = data[column_names[1:10]]
# X_train = data.drop(['Sample code number', 'class'], axis=1)
# y_train = data['class']
print('-------------------------')
# X_train = np.array(X_train)
print(X_test.values[0])
print(y_test.values[0])
print(X_test.values[6])
print(y_test.values[6])
# print(X_test.values[5])
# print(y_test.values[5])
# 查看训练样本的数量和类别分布
# print(y_train.value_counts())
# print(y_test.value_counts())
# print(type(X_train))
# print(type(y_test))
X_train = numpy.array(X_train)
# X_test = numpy.array(X_test)
y_train = numpy.array(y_train)
# y_test = numpy.array(y_test)
# print(type(X_train))
# print(type(X_test))

# 计算先验概率
prior_prob = {label: (np.sum(y_train == label) + 1) / (len(y_train) + len(set(y_train)))
              for label in set(y_train)}

# 计算条件概率
cond_prob = {}
for label in set(y_train):
    label_data = X_train[y_train == label]
    for i in range(X_train.shape[1]):
        unique_values = np.unique(X_train[:, i])
        prob = {}
        for value in unique_values:
            prob[value] = (np.sum(X_train[:, i] == value) + 1) / (len(label_data) + len(unique_values))
        cond_prob[i] = prob


# 分类函数
def classify(instance):
    posterior_prob = {}
    for label in set(y_train):
        posterior = prior_prob[label]
        for i, value in enumerate(instance):
            if value in cond_prob[i]:
                posterior *= cond_prob[i][value]
            else:
                posterior *= 1 / (len(X_train[y_train == label]) + 1)
        posterior_prob[label] = posterior
    return max(posterior_prob, key=posterior_prob.get)


# 使用分类函数进行预测
# new_instance = np.array([..., ..., ...])  # 新实例的特征
# print(X_test.values[0])
# print(y_test.values[0])
# print(X_test.values[5])
# print(y_test.values[5])
# print(type(X_test.values))
# print(X_test.values)
# print(X_test.values.shape[0])
# print(X_test.values)
# ls = []
# for i in X_test.values[0]:
#     ls.append(i)
# print(ls)
# [7 2 4 1 3 4 3 3 1]
# 4
# [1 3 3 2 2 1 7 2 1]
# 2
predicted_label = classify(X_test.values[0])
print(f"The predicted label for the new instance is: {predicted_label}")
# for ii in X_test.values:
#     ls = []
#     for i in ii:
#         ls.append(i)
#     print(ls)
# predicted_label = classify(ls)
# print(f"The predicted label for the new instance is: {predicted_label}")
# i = np.array(i)
# print(type(i))
# print(i)

# //////////////////////////////////////////////////////////////////////////////////
# ////////////////////Another ONE///////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////
# 计算每个类别的先验概率。
# 对于每个特征，计算给定类别下该特征的条件概率。
# 使用贝叶斯定理计算后验概率。
# 选择具有最大后验概率的类别作为预测类别。

# #
# # 计算先验概率
# def calculate_prior(y):
#     prior = {c: (y == c).sum() / len(y) for c in np.unique(y)}
#     return prior
#
#
# # 计算条件概率
# def calculate_likelihood(X, y):
#     likelihood = {}
#     for c in np.unique(y):
#         X_c = X[y == c]
#         mean = X_c.mean()
#         var = X_c.var()
#         likelihood[c] = {
#             i: 1 / (np.sqrt(2 * np.pi * var[i])) * np.exp(-((X_c.iloc[:, i] - mean[i]) ** 2) / (2 * var[i])) for i in
#             range(X.shape[1])}
#     return likelihood
#
#
# # 朴素贝叶斯分类器
# class GaussianNaiveBayes:
#     def __init__(self):
#         self.prior = None
#         self.likelihood = None
#
#     def fit(self, X, y):
#         self.prior = calculate_prior(y)
#         self.likelihood = calculate_likelihood(X, y)
#
#     def predict(self, X):
#         predictions = []
#         for i in range(X.shape[0]):
#             p = self.prior['M']
#             for j, x in enumerate(X.iloc[i]):
#                 for c in self.prior:
#                     if c == 'M':
#                         continue
#                     p *= self.likelihood[c][j][x]
#             predictions.append(max(p, self.prior['B'] * (1 - p)))
#         return predictions
#
#
# # 实例化分类器
# gnb = GaussianNaiveBayes()
#
# # 训练模型
# gnb.fit(X_train, y_train)
#
# # 进行预测
# predictions = gnb.predict(X_test)
#
# # 评估模型
# from sklearn.metrics import accuracy_score
#
# print("Accuracy:", accuracy_score(y_test, predictions))

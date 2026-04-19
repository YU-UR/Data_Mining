import pandas as pd
import numpy as np

# 加载数据集
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
column_names = [
    'ID', 'Clump_Thickness', 'Uniformity_Of_Cell_Size', 'Uniformity_Of_Cell_Shape',
    'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin',
    'Normal_Nuclei', 'Mitoses', 'Class'
]
data = pd.read_csv(url, header=None, names=column_names)

# 分离特征和标签
X = data.iloc[:, 1:].values  # 所有特征
y = data.iloc[:, -1].values  # 目标变量


# 计算欧氏距离
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


# 投票机制确定类别
def vote(neighbors):
    counts = np.bincount(neighbors)
    return np.argmax(counts)


# KNN分类器
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        # 无需拟合模型，直接保存数据
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        # 预测单个样本
        predictions = []
        for sample in X_test:
            distances = []
            for i, train_sample in enumerate(self.X_train):
                distance = euclidean_distance(sample, train_sample)
                distances.append((distance, self.y_train[i]))
            distances.sort(key=lambda x: x[0])
            neighbors = distances[:self.k]
            predictions.append(vote(neighbors))
        return predictions


# 实例化KNN分类器
knn = KNN(k=3)

# 训练模型
knn.fit(X, y)

# 进行预测
X_test = X[:5]  # 假设我们只对前5个样本进行预测
predictions = knn.predict(X_test)

# 打印预测结果
print(predictions)

# 我们首先定义了一个euclidean_distance函数来计算两个样本之间的欧氏距离。
# 然后，我们创建了一个KNN类，它有一个fit方法来保存训练数据，
# 和一个predict方法来对新的测试样本进行分类。

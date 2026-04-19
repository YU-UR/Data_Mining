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
X = data.iloc[:, 1:]  # 所有特征
y = data['Class']  # 目标变量


# 计算信息增益
def information_gain(X, y, feature_index, threshold):
    left, right = split_data(X, y, feature_index, threshold)
    p_left = len(left) / (len(X))
    p_right = len(right) / (len(X))
    p_total = p_left * entropy(y) + p_right * entropy(right)
    return entropy(y) - p_total


# 计算熵
def entropy(y):
    counts = np.bincount(y)
    p = counts / len(y)
    return -np.sum(p * np.log2(p))


# 根据特征和阈值分割数据集
def split_data(X, y, feature_index, threshold):
    X_left = X[y < threshold]
    X_right = X[y >= threshold]
    y_left = y[y < threshold]
    y_right = y[y >= threshold]
    return X_left, X_right, y_left, y_right


# 决策树节点
class Node:
    def __init__(self, is_leaf=False, prediction=None, feature_index=None, threshold=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = None
        self.right = None


# 构建决策树
def build_tree(X, y):
    # 递归终止条件：所有样本都属于同一类别
    if len(np.unique(y)) == 1:
        return Node(is_leaf=True, prediction=y[0])

    # 选择最佳特征和阈值
    best_feature_index = None
    max_info_gain = 0
    for i in range(X.shape[1]):
        for value in np.unique(X[:, i]):
            info_gain = information_gain(X, y, i, value)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_feature_index = i
                best_threshold = value

    # 创建当前节点
    current_node = Node(feature_index=best_feature_index, threshold=best_threshold)

    # 递归构建左右子树
    X_left, X_right, y_left, y_right = split_data(X, y, best_feature_index, best_threshold)
    current_node.left = build_tree(X_left, y_left) if len(X_left) > 0 else None
    current_node.right = build_tree(X_right, y_right) if len(X_right) > 0 else None

    return current_node


# 应用决策树进行预测
def predict(node, sample):
    if node.is_leaf:
        return node.prediction
    else:
        if sample[node.feature_index] < node.threshold:
            return predict(node.left, sample)
        else:
            return predict(node.right, sample)


# 训练模型
tree = build_tree(X,y)

# 进行预测
predictions = [predict(tree, sample) for sample in X]

# 评估模型
from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(y, predictions))

# 它使用信息增益作为特征选择的准则

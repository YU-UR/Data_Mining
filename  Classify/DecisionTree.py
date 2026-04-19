import math
import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score

start = time.time()
# 首先定义了一个 DecisionTreeNode 类来表示决策树的节点。
# 然后，我们定义了几个辅助函数来计算熵、信息增益、获取最重要的属性，并根据这些信息构建决策树。
# build_tree 函数递归地构建决策树，predict 函数递归地对新实例进行分类预测。


class DecisionTreeNode:
    def __init__(self, attribute=None, value=None, left=None, right=None, class_label=None):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.class_label = class_label


def entropy(df):
    total = len(df)
    counts = df.iloc[:, -1].value_counts()
    sum_entropy = 0
    for count in counts.values:
        p_count = count / total
        sum_entropy += -p_count * math.log2(p_count)
    return sum_entropy


def information_gain(df, attribute, values):
    total_entropy = entropy(df)
    split_entropy = 0
    for value in values:
        subset = df[df[attribute] == value]
        split_entropy += (len(subset) / len(df)) * entropy(subset)
    return total_entropy - split_entropy


def get_most_important_attribute(df):
    best_attribute = None
    max_info_gain = 0
    for attribute in df.columns[:-1]:  # Exclude the last column (class label)
        unique_values = df[attribute].unique()
        info_gain = information_gain(df, attribute, unique_values)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_attribute = (attribute, unique_values)
    return best_attribute


def build_tree(df, min_samples_split=2):
    if len(df) <= min_samples_split or len(df.iloc[:, -1].unique()) == 1:
        class_label = df.iloc[:, -1].mode()[0]
        return DecisionTreeNode(class_label=class_label)
    best_attribute, unique_values = get_most_important_attribute(df)
    left_indices = df[df[best_attribute] == unique_values[0]].index
    right_indices = df[df[best_attribute] != unique_values[0]].index
    left_tree = build_tree(df.reindex(left_indices), min_samples_split)
    right_tree = build_tree(df.reindex(right_indices), min_samples_split)
    return DecisionTreeNode(attribute=best_attribute, value=unique_values[0], left=left_tree, right=right_tree)


def predict(node, instance):
    if node.class_label is not None:
        return node.class_label
    if instance[node.attribute] == node.value:
        return predict(node.left, instance)
    else:
        return predict(node.right, instance)


# 给数据集加上列名
column_names = ['Sample code number', 'clump thickness', 'uniformity of cell size', 'uniformity of cell shape',
                'marginal adhesion',
                'single epithelial cell size', 'bare nuclei', 'bland chromatin', 'normal nucleoli', 'mitnoses', 'class']
# data = pd.read_csv(url, header=None, names=column_names)
data = pd.read_csv("breast-cancer-wisconsin.data", header=None, names=column_names)
mean_value = data[data["bare nuclei"] != "?"]["bare nuclei"].astype(np.int64).mean()  # 计算异常值列的平均值
data = data.replace('?', mean_value)
data["bare nuclei"] = data["bare nuclei"].astype(np.int64)
print(f"--------------------------------------------------")

df = data.iloc[:, 1:11]
# 切分数据集为训练集和测试集
train_df = df.sample(frac=0.7, random_state=42)
test_df = df.drop(train_df.index)

tree = build_tree(train_df)
predictions = []
predicted_class = test_df.apply(lambda row: predict(tree, row.iloc[:-1]), axis=1)
for row, key in zip(test_df.itertuples(index=False), predicted_class):
    row = list(row)
    print(f"The record is: {row}")
    print(f"The real class is: {row[-1]}")
    predictions.append(key)
    print(f"The predicted class for the record is: {key}")
    print(f"--------------------------------------------------")

end = time.time()
print(f"Total time is: {round(end-start,5)}")

test_class = test_df.iloc[:, 9].values.tolist()
print(f"--------------------------------------------------")
print("Accuracy:", round(accuracy_score(test_class, predictions), 5))


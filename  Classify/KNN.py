# 首先定义了一个euclidean_distance函数来计算两个实例之间的欧几里得距离。
# 然后，我们定义了find_nearest_neighbors函数来找到最近的K个邻居，vote函数来进行投票并确定预测的类别。
# split_data函数用于将原始数据集切分为训练集和测试集。
# predict函数是主要的预测函数，它接受训练数据、测试实例和K值作为输入，并返回预测的类别。
# 使用np.random.rand来生成一个随机的测试实例，这在实际应用中会被替换为实际的测试数据。
import time

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')
start = time.time()


# 计算两个点之间的欧几里得距离
def euclidean_distance(point1, point2):
    return sum((x - y) ** 2 for x, y in zip(point1, point2))


# 找到最近的K个邻居，使用位置信息而不是直接使用索引
def find_nearest_neighbors(train_data, test_instance, k):
    distances = []
    for index, train_instance in train_data.iterrows():
        distance = euclidean_distance(train_instance[:-1], test_instance[:-1])
        distances.append((index, distance))  # 保存位置信息和距离
    nearest_neighbors = sorted(distances, key=lambda x: x[1])[:k]
    return nearest_neighbors


# 进行投票，返回预测的类别
def vote(neighbors, train_data):
    class_votes = {}
    for index, _ in neighbors:
        try:
            class_label = train_data.loc[index, -1]  # 类别标签位于最后一列
            class_votes[class_label] = class_votes.get(class_label, 0) + 1
        except KeyError:
            # 如果索引无效，可以选择记录错误或跳过该邻居
            print(f"Warning: Key {index} not found in training data.")
            continue

    # 如果类投票字典为空，返回默认类别或错误信息
    if not class_votes:
        return "No class found"  # 或者你可以选择一个默认的类别标签

    # 返回票数最多的类别
    return max(class_votes, key=class_votes.get)


# 切分数据集为训练集和测试集
def split_data(df, test_ratio=0.2):
    # 使用 train_test_split 切分数据，并重置索引
    train_df, test_df = train_test_split(df, test_size=test_ratio, random_state=42, shuffle=False)
    train_df.reset_index(drop=True, inplace=True)  # 重置训练集的索引
    test_df.reset_index(drop=True, inplace=True)  # 重置测试集的索引
    return train_df, test_df


# 预测函数
def predict(train_data, test_instance, k):
    # 使用欧几里得距离找到最近的K个邻居
    distances = []
    for index, train_instance in train_data.iterrows():
        distance = euclidean_distance(train_instance[:-1], test_instance[:-1])
        distances.append((index, distance))
    nearest_neighbors = sorted(distances, key=lambda x: x[1])[:k]

    # 收集邻居的类别标签并进行投票
    class_votes = {}
    for index, _ in nearest_neighbors:
        try:
            class_label = train_data.iloc[index][-1]  # 使用 .iloc 来访问类别标签
            class_votes[class_label] = class_votes.get(class_label, 0) + 1
        except KeyError:
            # 如果索引无效，可以选择记录错误或跳过该邻居
            print(f"Warning: Key {index} not found in training data.")
            continue

    # 如果类投票字典为空，返回默认类别或错误信息
    if not class_votes:
        return "No class found"

    # 返回票数最多的类别
    return max(class_votes, key=class_votes.get)


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
# 切分数据集
train_df, test_df = split_data(df)
predictions = []
# K值
k = 1
print(f"--------------------------------------------------")
for row in test_df.itertuples(index=False):
    row = list(row)
    print(f"The record is: {row}")
    print(f"The real class is: {row[-1]}")
    predicted_class = predict(train_df, row, k)
    predictions.append(predicted_class)
    print(f"The predicted class for the record is: {predicted_class}")
    print(f"--------------------------------------------------")

end = time.time()
print(f"Total time is: {round(end - start, 5)}")

test_class = test_df.iloc[:, 9].values.tolist()
print(f"--------------------------------------------------")
print("Accuracy:", round(accuracy_score(test_class, predictions), 5))

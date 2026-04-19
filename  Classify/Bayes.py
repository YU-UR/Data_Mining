import pandas as pd
import numpy as np
import warnings
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')
pd.options.display.max_rows = None
pd.options.display.max_columns = None
start = time.time()

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
df = data.iloc[:, 1:11].values.tolist()
# data是 DataFrame类型
# 选取第2列到第10列（注意 Python 索引从 0 开始）
X = data.iloc[:, 1:10].values.tolist()
# 选取第11列（索引为10的列）
y = data.iloc[:, 10].values.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
features = X_train
labels = y_train

# # 计算类别的数量
num_classes = len(set(labels))
# 初始化类别概率和条件概率字典
class_probabilities = {}
feature_probabilities = {class_label: {} for class_label in set(labels)}

# 计算每个类别的概率
for label in set(labels):
    class_probabilities[label] = (labels.count(label) + 1) / (len(labels) + num_classes)

# 计算每个类别下每个特征的条件概率
for feature_index in range(9):  # 假设前9个属性为特征，最后一个为类别标签
    feature_values = [record[feature_index] for record in df]
    unique_values = set(feature_values)  # 获取所有不同的特征值
    for class_label in set(labels):
        # 获取当前类别下的所有记录
        class_records = [record for record in df if record[-1] == class_label]
        # 计算当前类别下特征的条件概率
        feature_probabilities[class_label][feature_index] = {}
        for value in unique_values:
            # 计算当前特征值在类别中出现的频率
            frequency = sum(1 for record in class_records if record[feature_index] == value)
            # 计算条件概率，这里使用Laplace平滑避免零概率问题
            total_count = len(class_records)
            probability = (frequency + 1) / (total_count + len(unique_values))
            feature_probabilities[class_label][feature_index][value] = probability


def predict(record, class_probabilities, feature_probabilities):
    prediction = None
    max_probability = 0
    for class_label in set(labels):
        probability = class_probabilities[class_label]
        for feature, value in enumerate(record):
            probability *= feature_probabilities[class_label][feature].get(value, 1 / sum(
                feature_probabilities[class_label][feature].values()))
        if probability > max_probability:
            max_probability = probability
            prediction = class_label
    return prediction


# 测试输出结果
test_records = X_test
test_classes = y_test
print(f"//////////////////////////////////////////")
# print(type(y_test))
print(f"输入的特征测试集list如下：")
print(X_test)
print(f"输入的类别测试集list如下：")
print(y_test)
print(f"/////////////////////////////////////////")
predictions = []
for ii, jj in zip(test_records, test_classes):
    print(f"Features are {ii}")
    print(f"The real class is: {jj}")
    # 进行预测
    predicted_class = predict(ii, class_probabilities, feature_probabilities)
    predictions.append(predicted_class)
    print(f"The predicted class for these features is: {predicted_class}")
    print(f"--------------------------------------------------")

end = time.time()
print(f"Total time is: {round(end-start,5)}")

print(f"--------------------------------------------------")
print("Accuracy:", round(accuracy_score(y_test, predictions), 5))

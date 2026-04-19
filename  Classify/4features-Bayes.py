import random

# 示例数据集，每一行包含9个属性和一个类别标签
data = [
    [5.1, 3.5, 1.4, 0.2, 'Setosa'],
    [6.1, 3.0, 4.9, 1.8, 'Versicolor'],
    [5.9, 3.0, 5.1, 1.8, 'Versicolor'],
    [5.5, 2.3, 4.0, 1.3, 'Setosa'],
    # ... 更多记录
]

# 分离特征和标签
features = [record[:-1] for record in data]
labels = [record[-1] for record in data]
# 计算类别的数量
num_classes = len(set(labels))

# 初始化类别概率和条件概率字典
class_probabilities = {}
feature_probabilities = {class_label: {} for class_label in set(labels)}

# 计算每个类别的概率
for label in set(labels):
    class_probabilities[label] = (labels.count(label) + 1) / (len(labels) + num_classes)

# 初始化特征条件概率字典
feature_probabilities = {class_label: {} for class_label in set(labels)}

# 计算每个类别下每个特征的条件概率
for class_label in set(labels):
    # 获取当前类别下的所有记录
    class_records = [record for record in data if record[-1] == class_label]

    # 遍历每个特征
    for feature_index in range(4):  # 假设前8个属性为特征
        feature_values = [record[feature_index] for record in class_records]
        unique_values = set(feature_values)  # 获取所有不同的特征值

        # 初始化当前类别特征的条件概率字典
        feature_probabilities[class_label][feature_index] = {}

        # 计算每个特征值的条件概率
        for value in unique_values:
            # 计算当前特征值在类别中出现的频率
            frequency = feature_values.count(value)
            # 计算条件概率，这里使用Laplace平滑避免零概率问题
            total_count = len(feature_values)
            probability = (frequency + 1) / (total_count + len(unique_values))
            feature_probabilities[class_label][feature_index][value] = probability

# 计算类别的先验概率
class_probabilities = {class_label: (labels.count(class_label) + 1) / (len(labels) + num_classes) for class_label in
                       set(labels)}


# 现在我们有了特征条件概率和类别先验概率，可以进行预测
def predict(record, class_probabilities, feature_probabilities):
    prediction = None
    max_probability = 0
    for class_label in set(labels):
        probability = class_probabilities[class_label]
        for feature, value in enumerate(record):
            probability *= feature_probabilities[class_label][feature].get(value, 1/sum(feature_probabilities[class_label][feature].values()))
        if probability > max_probability:
            max_probability = probability
            prediction = class_label
    return prediction

# 测试记录
test_record = [5.0, 3.2, 1.2, 0.2]

# 进行预测
predicted_class = predict(test_record, class_probabilities, feature_probabilities)
print(f"The predicted class for the test record is: {predicted_class}")
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min


# 读取CSV文件并提取特征
def read_csv_features(file_path, feature_indices):
    data = []
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # 提取特征列的值，并转换为浮点数
            features = [float(row[index]) for index in feature_indices]
            data.append(features)
    return np.array(data)


# K-means算法实现
def k_means(X, n_clusters):
    # 随机选择K个点作为初始质心
    centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]

    while True:
        # 计算每个点到各个质心的距离，并分配到最近的质心
        distances, closest_centroid_indices = pairwise_distances_argmin_min(X, centroids)

        # 更新质心
        new_centroids = np.array([X[closest_centroid_indices == i].mean(axis=0) for i in range(n_clusters)])

        # 如果质心没有变化，结束循环
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, closest_centroid_indices


# 主程序
def main():
    # 读取数据
    file_path = '../Cluster/data.csv'  # 假设已经处理过的数据保存在这个文件中
    feature_indices = ['A', 'C', 'E', 'H', 'N', 'R', 'SP', 'SH', 'B', 'L']

    # 读取特征数据
    X = read_csv_features(file_path, feature_indices)

    # 标准化特征数据
    from sklearn.impute import SimpleImputer

    # imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')  # 或者 'median', 'most_frequent'
    # X_filled = imputer.fit_transform(X)
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X_filled)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # print(X)


    # 聚类数目
    n_clusters = 3  # 你可以根据需要设置K的值
    #
    # # 执行K-means算法
    centroids, labels = k_means(X, n_clusters)
    #
    # # 输出结果
    # print("Centroids:")
    # print(centroids)
    # print("Labels:")
    # print(labels)
    #
    # # 可视化聚类结果
    # plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', marker='.')
    # plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
    # plt.title('K-means Clustering of Tourist Destinations')
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    # plt.show()


if __name__ == '__main__':
    main()
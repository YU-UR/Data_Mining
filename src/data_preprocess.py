import numpy as np
import pandas as pd
import matplotlib.pyplot as plt2
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')
from data_explore import car_df

# 获取数据
# car_df = pd.read_csv('../results/explore/car_df.csv')

# 数据预处理
# 剔除carlength
features = car_df.drop(['carlength'], axis=1)

# 将取值具有大小意义的类别型变量数据转变为数值型映射
features1 = features.copy()
# features1.to_csv('../results/preprocess/car_df_orig.csv')
# print(features1.head())
# 使用LabelEncoder对不具实体数值数据编码
carSize1 = LabelEncoder().fit_transform(features1['carSize'])
features1['carSize'] = carSize1

# 对于类别离散型特征，取值间没有大小意义的，可采用one-hot编码
cate = features1.select_dtypes(include='object').columns
# print(cate)
features1 = features1.join(pd.get_dummies(features1[cate], dtype=int)).drop(cate, axis=1)
# features1.to_csv('../results/preprocess/car_df_change.csv')
# print(features1.head())

# 对数值型数据进行归一化
features1 = preprocessing.MinMaxScaler().fit_transform(features1)
features1 = pd.DataFrame(features1)
# features1.to_csv('../results/preprocess/car_df_normalisation.csv')
# print(features1.head())


# # 对数据集进行PCA降维（信息保留为99.99%）
# pca = PCA(n_components=0.9999)  # 保证降维后的数据保持90%的信息，则填0.9
# features2 = pca.fit_transform(features1)
# # 降维后，每个主要成分的解释方差占比（解释PC携带的信息多少）
# ratio = pca.explained_variance_ratio_
# print('各主成分的解释方差占比：', ratio)
# # 降维后有几个成分
# print('降维后有几个成分：', len(ratio))
# # 累计解释方差占比
# cum_ratio = np.cumsum(ratio)
# print('累计解释方差占比：', cum_ratio)
#
# # 绘制PCA降维后各成分方差占比的直方图和累计方差占比折线图
# plt2.figure(figsize=(8, 6))
# X = range(1, len(ratio) + 1)
# Y = ratio
# plt2.bar(X, Y, edgecolor='black')
# plt2.plot(X, Y, 'r.-')
# plt2.plot(X, cum_ratio, 'b.-')
# plt2.ylabel('explained_variance_ratio')
# plt2.xlabel('PCA')
# plt2.savefig('../results/preprocess/pca_plot.png', bbox_inches='tight')
# # plt2.show()



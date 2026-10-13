import numpy as np
import pandas as pd
import matplotlib.pyplot as plt3
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings('ignore')
from data_preprocess import features1
from data_explore import car_price

# car_price = pd.read_csv('../results/explore/car_price.csv')
# features1 = pd.read_csv('../results/preprocess/car_df_normalisation.csv')
# PCA选择降维保留8个主要成分
pca = PCA(n_components=8)
features3 = pca.fit_transform(features1)
# 降维后的累计各成分方差占比和（即解释PC携带的信息多少）81.3%
# print(sum(pca.explained_variance_ratio_))

# K-Means聚类
# 肘方法看k值，簇内离差平方和
# 对每一个k值进行聚类并且记下对于的SSE，然后画出k和SSE的关系图
# sse = []
# for i in range(1, 15):
#     km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
#     km.fit(features3)
#     sse.append(km.inertia_)
#
# plt3.plot(range(1, 15), sse, marker='*')
# plt3.xlabel('n_clusters')
# plt3.ylabel('distortions')
# plt3.title("The Elbow Method")
# plt3.savefig('../results/clustering/Elbow_K_plot.png', bbox_inches='tight')
# # plt3.show()

# 进行K-Means聚类分析
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=0)
kmeans.fit(features3)
lab = kmeans.predict(features3)
# print(lab)

# 绘制聚类结果2维的散点图，效果不是很好
# plt3.figure(figsize=(8, 8))
# plt3.scatter(features3[:, 0], features3[:, 1], c=lab)
# for ii in np.arange(205):
#     plt3.text(features3[ii, 0], features3[ii, 1], s=car_price.car_ID[ii])
# plt3.xlabel('PC1')
# plt3.ylabel('PC2')
# plt3.title('K-Means PCA')
# plt3.savefig('../results/clustering/Kmeans5_pac2_plot.png', bbox_inches='tight')
#

# # 绘制聚类结果后3d散点图
# plt3.figure(figsize=(8, 8))
# ax = plt3.subplot(111, projection='3d')
# ax.scatter(features3[:, 0], features3[:, 1], features3[:, 2], c=lab)
# # 视角转换，转换后更易看出簇群
# ax.view_init(30, 45)
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
# plt3.savefig('../results/clustering/Kmeans5_pac3_plot.png', bbox_inches='tight')

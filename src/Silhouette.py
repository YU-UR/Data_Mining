import numpy as np
import pandas as pd
import matplotlib.pyplot as plt4
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings('ignore')
# 绘制轮廓图和3d散点图
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from clustering import features3
# from data_explore import car_price
car_price = pd.read_csv('../results/explore/car_price.csv')
# for n_clusters in range(2, 9):
#     fig = plt4.figure(figsize=(12, 6))
#     ax1 = fig.add_subplot(121)
#     ax2 = fig.add_subplot(122, projection='3d')
#
#     ax1.set_xlim([-0.1, 1])
#     ax1.set_ylim([0, len(features3) + (n_clusters + 1) * 10])
#     km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=0)
#     y_km = km.fit_predict(features3)
#     silhouette_avg = silhouette_score(features3, y_km)
#     print('n_cluster=', n_clusters, 'The average silhouette_score is :', silhouette_avg)
#
#     cluster_labels = np.unique(y_km)
#     silhouette_vals = silhouette_samples(features3, y_km, metric='euclidean')
#     y_ax_lower = 10
#     for i in range(n_clusters):
#         c_silhouette_vals = silhouette_vals[y_km == i]
#         c_silhouette_vals.sort()
#         cluster_i = c_silhouette_vals.shape[0]
#         y_ax_upper = y_ax_lower + cluster_i
#         color = cm.nipy_spectral(float(i) / n_clusters)
#         ax1.fill_betweenx(range(y_ax_lower, y_ax_upper), 0, c_silhouette_vals, edgecolor='none', color=color)
#         ax1.text(-0.05, y_ax_lower + 0.5 * cluster_i, str(i))
#         y_ax_lower = y_ax_upper + 10
#
#     ax1.set_title('The silhouette plot for the various clusters')
#     ax1.set_xlabel('The silhouette coefficient values')
#     ax1.set_ylabel('Cluster label')
#
#     ax1.axvline(x=silhouette_avg, color='red', linestyle='--')
#
#     ax1.set_yticks([])
#     ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
#
#     colors = cm.nipy_spectral(y_km.astype(float) / n_clusters)
#     ax2.scatter(features3[:, 0], features3[:, 1], features3[:, 2], marker='.', s=30, lw=0, alpha=0.7, c=colors,
#                 edgecolor='k')
#
#     centers = km.cluster_centers_
#     ax2.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='o', c='white', alpha=1, s=200, edgecolor='k')
#
#     for i, c in enumerate(centers):
#         ax2.scatter(c[0], c[1], c[2], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')
#
#     ax2.set_title("The visualization of the clustered data.")
#     ax2.set_xlabel("Feature space for the 1st feature")
#     ax2.set_ylabel("Feature space for the 2nd feature")
#     ax2.view_init(30, 45)
#
#     plt4.suptitle(("Silhouette analysis for KMeans clustering on sample data "
#                    "with n_clusters = %d" % n_clusters),
#                 fontsize=14, fontweight='bold')
#     plt4.savefig('../results/silhouette/silhouette_plot{}.png'.format(n_clusters), bbox_inches='tight')
# # plt4.show()


# 调整选择k=6进行聚类
kmeans = KMeans(n_clusters=6, init='k-means++', n_init=10, max_iter=300, random_state=0)
y_pred = kmeans.fit_predict(features3)
# print(y_pred)

# 将聚类后的类目放入原特征数据中
car_df_km = car_price.copy()
car_df_km['km_result'] = y_pred

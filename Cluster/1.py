import numpy as np
import pandas as pd
import matplotlib.pyplot as plt1
import seaborn as sns
import warnings

from matplotlib import pyplot as plt
from openpyxl.workbook import Workbook
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.inf)
# pd.set_option('display.width', 300)  # Setting the character display width
pd.set_option('display.max_rows', None)  # Setting the maximum lines to be displayed
pd.set_option('display.max_columns', None)  # Set to show maximum columns, None to show all columns.

# 获取数据
data = pd.read_csv('../Cluster/output.csv', encoding='utf-8')
# print(data.head())
# 提取类别变量的列名
# cate_columns = ['Latitude', 'Longitude', 'A', 'C', 'E', 'H', 'N', 'R', 'SP', 'SH', 'B', 'L']
# # 打印类别变量每个分类的取值情况
# for i in cate_columns:
#     # print(i)
#     print(set(data[i]))
data1 = data.drop(['ID', 'City', 'Place_names1', 'Place_names2'], axis=1)
# print(data1.head())
scaler = StandardScaler()
X = scaler.fit_transform(data1)
# print(X_scaled)
# K-Means聚类 肘方法看k值，簇内离差平方和 对每一个k值进行聚类并且记下对于的SSE，然后画出k和SSE的关系图 /////////// n_clusters=i: 这个参数指定了聚类的数量，即你想要将数据分成多少个簇。i
# 应该是一个整数，表示簇的数目。在实际应用中，你可能需要根据数据的特性和需求来选择合适的 i 值。 init='k-means++': 这个参数指定了初始化质心（centroids）的方法。'k-means++' 是 K-means
# 算法的一种改进的初始化策略，它可以比随机初始化质心更有可能选择到较好的初始质心，从而提高聚类的质量。k-means++ 算法会根据数据点之间的距离来选择质心，使得初始质心尽可能分散。 n_init=10:
# 这个参数指定了算法运行的次数，即 KMeans 算法将随机选择质心并运行 n_init 次。每次运行都会产生一组质心，算法会保留使得数据点到其质心的总距离（within-cluster sum of squares,
# WCSS）最小的那一组质心。设置 n_init 的一个较大值可以增加找到全局最优解的概率。 max_iter=30: 这个参数指定了每次算法运行时的最大迭代次数。在每次运行 K-means
# 算法时，算法会迭代地更新质心和簇分配，直到满足停止条件（例如质心位置不再变化或总内平方和的变化小于某个阈值）。max_iter 限制了单次运行的最大迭代次数，以避免无限循环。 random_state=0:
# 这个参数设置了随机数生成器的种子。在 K-means 算法中，初始质心的选择是随机的，设置 random_state 可以确保每次运行代码时都能得到相同的初始质心，从而使结果可复现。 综上所述，这行代码初始化了一个 K-means
# 聚类模型，设置了聚类数目为 i，使用 k-means++ 方法初始化质心，算法将运行 n_init=10 次，每次运行的最大迭代次数为 max_iter=30，并且通过设置随机数种子为 0 来确保结果的可复现性。
sse = []
for i in range(1, 15):
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=30, random_state=0)
    km.fit(X)
    sse.append(km.inertia_)

plt.plot(range(1, 15), sse, marker='*')
plt.xlabel('n_clusters')
plt.ylabel('distortions')
plt.title("The Elbow Method")
plt.savefig('../Cluster/Elbow_K5.png', bbox_inches='tight')
# ///cluster=5
# plt.show()

# 进行K-Means聚类分析
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=30, random_state=0)
kmeans.fit(X)
lab = kmeans.predict(X)
# print(lab)
#
# 绘制聚类结果2维的散点图，效果不是很好
# plt.figure(figsize=(8, 8))
# plt.scatter(X[:, 0], X[:, 1], c=lab)
# for i in np.arange(232):
#     plt.text(X[i, 0], X[i, 1], s=data.ID[i])
# plt.xlabel('Latitude')
# plt.ylabel('Longitude')
# plt.title('K-Means6')
# plt.savefig('../Cluster/Kmeans5.png', bbox_inches='tight')

# 绘制聚类结果后3d散点图
# plt.figure(figsize=(8, 8))
# ax = plt.subplot(111, projection='3d')
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=lab)
# # 视角转换，转换后更易看出簇群
# ax.view_init(30, 45)
# ax.set_xlabel('Latitude')
# ax.set_ylabel('Longitude')
# ax.set_zlabel('A')
# plt.savefig('../Cluster/Kmeans5_3d.png', bbox_inches='tight')
# plt.show()


# 将聚类后的类目放入原特征数据中
data2 = data.copy()
data2['km_result'] = lab
# 统计聚类后每个集群的数量
# df = data2.groupby('km_result')['ID'].count()
# print(df)
# km_result
# 0     31
# 1    102
# 2     46
# 3     51
# 4      2
# Name: ID, dtype: int64
# df = data2.loc[:, ['ID', 'City', 'Place_names1', 'Place_names2']]
# 分类1的所有城市信息
# df1 = df.loc[data2['km_result'] == 1]
# df1.to_csv('../Cluster/df1.csv')
# df = df.groupby('City')['City'].count()
# print(df1.to_string(index=False))
# workbook = Workbook()
# worksheet = workbook.active
# for index, row in df1.iterrows():
#     worksheet.append(row.tolist())
# workbook.save('../Cluster/df1.xlsx')
# City
# Anbar            2
# Arbil            2
# Babylon         14
# Baghdad         20
# Basra            2
# Dhi Qar          7
# Diyala           5
# Dohuk            2
# Karbala          5
# Kirkuk           4
# Mosul           13
# Muthanna         3
# Najaf           11
# Qadisiyah        2
# Salahaddin       5
# Sulaymaniyah     3
# Wasit            2
# Name: City, dtype: int64

df = data2.loc[:, ['City', 'Place_names1']]
df2 = df.loc[((data2['km_result'] == 1) & (
            (data2['City'] == 'Babylon') | (data2['City'] == 'Baghdad') | (data2['City'] == 'Mosul') | (
                data2['City'] == 'Najaf')))]

# print(df.count())
# City            58
# Place_names1    58
# dtype: int64
print(df2)
# workbook = Workbook()
# worksheet = workbook.active
# for index, row in df2.iterrows():
#     worksheet.append(row.tolist())
# workbook.save('../Cluster/df2.xlsx')


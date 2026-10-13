import numpy as np
import pandas as pd
import matplotlib.pyplot as plt5
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from openpyxl import Workbook

warnings.filterwarnings('ignore')
# 绘制轮廓图和3d散点图
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from Silhouette import car_df_km

pd.options.display.max_rows = 100
pd.options.display.max_columns = 100
# 统计聚类后每个集群的车型数
car_df_km.groupby('km_result')['car_ID'].count()
# 统计每个集群各个品牌的车型数
car_df_km.groupby(by=['km_result', 'carBrand'])['car_ID'].count()
# 统计每个品牌所属各个集群的车型数
car_df_km.groupby(by=['carBrand', 'km_result'])['km_result'].count()

# 查看特指车名‘vokswagen’车型的聚类集群
df = car_df_km.loc[:, ['car_ID', 'CarName', 'carBrand', 'km_result', 'carSize']]
# print(df.loc[df['CarName'].str.contains("vokswagen")])
# 查看特指车名为‘vokswagen’车型的竞品车型（分类0的所有车型）
# print(df.loc[df['km_result'] == 0])

# 提取分类为0的所有车型特征数据
df0 = car_df_km.loc[car_df_km['km_result'] == 0]
# df0.head()
# print(df0.head())

# df0_1 = df0.drop(['car_ID', 'CarName', 'km_result'], axis=1)
# # 查看集群0的车型所有特征分布
# fig = plt5.figure(figsize=(20, 20))
# i = 1
# for c in df0_1.columns:
#     ax = fig.add_subplot(7, 4, i)
#     if df0_1[c].dtypes == 'int' or df0_1[c].dtypes == 'float':
#         sns.histplot(df0_1[c], ax=ax)
#     else:
#         sns.barplot(x=df0_1[c].value_counts().index, y=df0_1[c].value_counts(), ax=ax)
#     i = i + 1
#     plt5.xlabel('')
#     plt5.title(c)
# plt5.subplots_adjust(top=1.2)
# plt5.savefig('../results/analyse/Kmeans5_pac2_plot.png', bbox_inches='tight')

# 对不同车型级别、品牌、车身等类型特征进行数据透视
# df2 = df0.pivot_table(index=['carSize', 'carbody', 'carBrand', 'CarName'])

# 提取集群0中的A级车
df0_A = df0.loc[df0['carSize'] == 'A']
# df0_A.to_csv('../results/analyse/df0_A.csv')
# workbook = Workbook()
# worksheet = workbook.active
# for index, row in df0_A.iterrows():
#     worksheet.append(row.tolist())
# workbook.save('../results/analyse/df0_A.xlsx')
# print(df0_A)

# 查看集群0中A级车型的类别型变量的分类情况//上述df0_A已经包含所有
# ate_col = df0_A.select_dtypes(include='object').columns
# df3 = df0_A[ate_col]
# print(df3)

# 对集群0中A级车的特征进行数据透视
# df4 = df0_A.pivot_table(index=['carBrand', 'CarName', 'doornumber', 'aspiration', 'drivewheel'])
# print(df4)


# 对油耗的分析('citympg','highwaympg')
# lab = df0_A['CarName']
# fig, ax = plt5.subplots(figsize=(10, 8))
# ax.barh(range(len(lab)), df0_A['highwaympg'], tick_label=lab, color='red')
# ax.barh(range(len(lab)), df0_A['citympg'], tick_label=lab, color='blue')
# for i, (highway, city) in enumerate(zip(df0_A['highwaympg'], df0_A['citympg'])):
#     ax.text(highway, i, highway, ha='right')
#     ax.text(city, i, city, ha='right')
#
# plt5.legend(('highwaympg', 'citympg'), loc='upper right')
# plt5.title('miles per gallon')
# plt5.savefig('../results/analyse/HighCityMap_plot.png', bbox_inches='tight')
# plt5.show()


# 其他6个特征分析
# colors = ['yellow', 'blue', 'green', 'red', 'gray', 'tan', 'darkviolet']
# col2 = ['symboling', 'wheelbase', 'enginesize', 'horsepower', 'curbweight', 'price']
# data = df0_A[col2]
# fig = plt5.figure(figsize=(10, 8))
# i = 1
# for c in data.columns:
#     ax = fig.add_subplot(3, 2, i)
#     plt5.barh(range(len(lab)), data[c], tick_label=lab, color=colors)
#     for y, x in enumerate(data[c].values):
#         plt5.text(x, y, "%s" % x)
#     i = i + 1
#     plt5.xlabel('')
#     plt5.title(c)
# plt5.subplots_adjust(top=1.2, wspace=0.7)
# plt5.savefig('../results/analyse/Others_plot.png', bbox_inches='tight')
# plt5.show()

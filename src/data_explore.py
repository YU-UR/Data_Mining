import numpy as np
import pandas as pd
import matplotlib.pyplot as plt1
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


warnings.filterwarnings('ignore')

# 保证可以显示中文字体
plt1.rcParams['font.sans-serif'] = ['SimHei']
# 正常显示负号
plt1.rcParams['axes.unicode_minus'] = False

# 获取数据
car_price = pd.read_csv('../data/car_price.csv')
# print(car_price.head())

# 查看数据类型和非空、重复值
# car_price.info()
# print(car_price.duplicated().sum())

# 提取类别变量的列名
cate_columns = ['symboling', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation',
                'enginetype', 'fuelsystem', 'cylindernumber']

# 打印类别变量每个分类的取值情况
# for i in cate_columns:
#     print(i)
#     print(set(car_price[i]))

# 气缸数可使用具体的数值替换分类
car_price['cylindernumber'] = car_price.cylindernumber.replace(
    {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'eight': 8, 'twelve': 12})
# 提取变量特征数据(除了'car_ID'和'CarName')
car_df = car_price.drop(['car_ID', 'CarName'], axis=1)
# 查看连续数值型情况，并是检查否有异常值
# 对数据进行描述性统计
# print(car_df.describe())



# 还可以描绘数据集的箱线图，查看异常值
#
# # 提取连续数值型数据的列名
# num_cols = car_df.columns.drop(cate_columns)
# # print(num_cols)
# # 绘制连续数值型数据的箱线图，检查异常值
# fig = plt1.figure(figsize=(12, 8))
# i = 1
# for col in num_cols:
#     ax = fig.add_subplot(3, 5, i)
#     sns.boxplot(data=car_df[col], ax=ax)
#     i = i + 1
#     plt1.title(col)
# plt1.subplots_adjust(wspace=0.4, hspace=0.3)
# plt1.savefig('../results/explore/box_plot.png', bbox_inches='tight')

# plt1.show()


# 去重查看CarName
# print(car_price['CarName'].drop_duplicates())  # 验证是否object全部改为数值类型

# 利用split，由CarName拆出品牌信息
carBrand = car_price['CarName'].str.split(expand=True)[0]
# 查看汽车品牌名称（过滤重复）
# print(set(carBrand))

#
# 修改品牌名称的不规则命名
carBrand = carBrand.replace(
    {'porcshce': 'porsche', 'vokswagen': 'volkswagen', 'Nissan': 'nissan', 'maxda': 'mazda', 'vw': 'volkswagen',
     'toyouta': 'toyota'})
# print(set(carBrand))
# 将carBrand放入原数据集中
car_price['carBrand'] = carBrand
#
# 由上面描述性统计可知，车身长范围为141.1~208.1英寸之间，可划分为6类
bins = [min(car_df.carlength) - 0.01, 145.67, 169.29, 181.10, 192.91, 200.79, max(car_df.carlength) + 0.01]
label = ['A00', 'A0', 'A', 'B', 'C', 'D']
carSize = pd.cut(car_df.carlength, bins, labels=label)
# print(carSize)

# 将车型大小分类放入数据集中
car_price['carSize'] = carSize
car_df['carSize'] = carSize
# car_df.to_csv('../results/explore/car_df.csv')
# car_price.to_csv('../results/explore/car_price.csv')
# # # 查看数值型特征的相关系数
df_corr = car_df.corr(method='pearson', numeric_only=True)
# print(df_corr)

# 绘制相关性热力图
# mask = np.zeros_like(df_corr)
# mask[np.triu_indices_from(mask)] = True
# plt1.figure(figsize=(10, 10))
# with sns.axes_style("white"):
#     ax = sns.heatmap(df_corr, mask=mask, square=True, annot=True, cmap='bwr')
# ax.set_title("df_corr Variables Relation")
# plt1.savefig('../results/explore/heatmap_plot.png', bbox_inches='tight')
# plt1.show()




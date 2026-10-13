import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
# 获取数据
car_price = pd.read_csv('../data/car_price.csv')
# 分离数值变量与分类变量
Nu_feature = list(car_price.select_dtypes(exclude=['object']).columns)  # 数值变量
Ca_feature = list(car_price.select_dtypes(include=['object']).columns)
# print(len(Nu_feature), Nu_feature)
# print(len(Ca_feature), Ca_feature)
car_price.describe().T

plt.figure(figsize=(30, 25))
i = 1
for col in Nu_feature:
    ax = plt.subplot(4, 4, i)
    ax = sns.distplot(car_price[col], color='red', hist=False)
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    i += 1

plt.savefig('../results/explore/distributions_plot.png', bbox_inches='tight')
plt.show()
#  大部分的数据都服从正态分布

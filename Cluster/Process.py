import pandas as pd
import re
#
# # 读取CSV文件
# df = pd.read_csv('../Cluster/processed.csv', sep=';', encoding='ISO-8859-1')
#
#
# # 定义一个函数，用于替换连续的两个分号
# def replace_double_semicolon(value):
#     return value.replace(';', ' ')
#
#
# # 应用函数替换所有相关列中的连续分号
# for column in df.columns[7:15]:  # 从第8列开始（索引为7），因为前7列没有分号分隔的数据
#     df[column] = replace_double_semicolon(df[column])
# # 保存修改后的DataFrame到新的CSV文件
# df.to_csv('../Cluster/processed.csv', index=False)

# ///////////////////////////////////////////////////////////
import csv
#
#
# # 定义一个函数来转换数据
# def transform_data(row):
#     # 将每一项中的分号（;;）替换为;0;
#     transformed_row = [cell.replace(';;', ';0;') for cell in row]
#     return transformed_row
#
#
# # 打开CSV文件
# with open('../Cluster/transformed_Iraqi_tourism_places.csv', mode='r', encoding='ISO-8859-1') as csvfile:
#     # 创建CSV阅读器
#     reader = csv.reader(csvfile)
#
#     # 读取CSV文件的标题行
#     header = next(reader)
#
#     # 打开一个新的CSV文件用于写入转换后的数据
#     with open('transformed_places.csv', mode='w', encoding='ISO-8859-1', newline='') as transformed_csvfile:
#         # 创建CSV写入器
#         writer = csv.writer(transformed_csvfile)
#
#         # 写入标题行
#         writer.writerow(header)
#
#         # 遍历CSV文件中的每一行
#         for row in reader:
#             # 转换数据并写入新的CSV文件
#             writer.writerow(transform_data(row))
#
# print("CSV文件已转换并保存为transformed_places.csv")
# ///////////////////////////////////////////////////////////////////////////////
# import csv


# 定义一个函数来转换数据
# def transform_data(row):
    # 检查每项是否以分号（;）结尾，如果是，则在末尾添加0
    # transformed_row = [cell + '0' if cell.endswith(';') else cell for cell in row]
    # 将;变为空格
    # transformed_row = [cell.replace(';', ',') for cell in transformed_row]
    # 去掉两端引号
    # transformed_row = [cell.strip('"') for cell in row]
    # 使用正则表达式去除每个单元格中的引号
    # transformed_row = [re.sub(r'"(.*?)"', r'\1', cell) for cell in row]
    # 去除单元格中的所有引号
    # return [cell.replace('"', '') for cell in row]
    # return transformed_row


# 打开CSV文件
# with open('../Cluster/transformed.csv', mode='r', encoding='ISO-8859-1') as csvfile:
#     # 创建CSV阅读器
#     reader = csv.reader(csvfile)
#
#     # 读取CSV文件的标题行
#     header = next(reader)
#
#     # 打开一个新的CSV文件用于写入转换后的数据
#     with open('transformed1.csv', mode='w', encoding='ISO-8859-1', newline='') as transformed_csvfile:
#         # 创建CSV写入器
#         writer = csv.writer(transformed_csvfile)
#
#         # 写入标题行
#         writer.writerow(header)
#
#         # 遍历CSV文件中的每一行
#         for row in reader:
#             # 转换数据并写入新的CSV文件
#             writer.writerow(transform_data(row))

# print("CSV文件已转换并保存为transformed1.csv")
# ///////////////////////////////
# import csv
# #
# # 定义一个函数来去除字符串中的非打印字符，包括引号
# def clean_cell(row):
#     return row.replace('"', '')
#
#
# # 打开原始CSV文件进行读取
# with open('../Cluster/2.csv', mode='r', encoding='utf-8') as infile:
#     reader = csv.reader(infile)
#
#     # 打开一个新的CSV文件用于写入转换后的数据
#     with open('output.csv', mode='w', encoding='utf-8', newline='') as outfile:
#         # 创建CSV写入器
#         writer = csv.writer(outfile)
#
#         # 逐行读取并写入转换后的数据
#         for row in reader:
#             # 对每个单元格进行清洗
#             cleaned_row = [clean_cell(cell) for cell in row]
#             writer.writerow(cleaned_row)
#
# print("CSV文件已转换并保存为output.csv")
# //////////////////////////////////////////////////////////
import pandas as pd

# 读取CSV文件
df = pd.read_csv('../Cluster/output.csv')

# 指定要替换的列范围，这里假设列A到L分别对应位置0到11
columns_to_replace = ['A', 'C', 'E', 'H', 'N', 'R', 'SP', 'SH', 'B', 'L']
# 遍历指定的列，将值为0的替换为2
for column in columns_to_replace:
    df[column] = df[column].replace(0, 2)

# 保存修改后的DataFrame到新的CSV文件
df.to_csv('data.csv', index=False)
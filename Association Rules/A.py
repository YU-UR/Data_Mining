# Apriori算法的主要思想是利用频繁项集的性质，通过逐层搜索的方式发现频繁项集。算法的过程如下：
# 1.初始扫描：计算单个项的支持度，筛选出满足最小支持度阈值的项集作为一阶频繁项集。
# 2.迭代生成：根据当前的k - 1阶频繁项集，生成候选k阶项集。
# 3.连接：将k - 1阶频繁项集两两连接生成候选k阶项集。
# 4.剪枝：对候选k阶项集进行剪枝操作，删除包含非频繁子项集的项集。
# 5.频繁项集计数：对候选k阶项集进行扫描，计算每个项集的支持度。
# 6.停止条件：如果没有发现新的频繁项集，或者已达到预定的阶数，则停止迭代。

# 数据集为一组购物篮数据
# dataset = {
#     'T1': {'牛奶', '面包'},
#     'T2': {'面包', '尿布', '啤酒', '鸡蛋'},
#     'T3': {'牛奶', '尿布', '啤酒', '可乐'},
#     'T4': {'面包', '牛奶', '尿布', '啤酒'},
#     'T5': {'面包', '牛奶', '尿布', '可乐'}
# }
# 打开文件并读取所有行
with open('../Association Rules/data.txt', 'r', encoding='UTF-8') as file:
    lines = file.readlines()
# print(lines)
# 将每一行转换为键值对
dataset = {}
for line in lines:
    line = line.strip()
    k = line.split(' ')[0]
    v = line.split(' ')[1:]
    dataset[k] = set(v)  # 列表['牛奶', '面包']转换成集合{'牛奶', '面包'}
# 依旧是关闭文件
file.close()
# 输出结果
# print(dataset)

support = 2  # 设置最小支持度计数 = 4
k = 0  # k表示当前为频繁k项集
goods = set()  # 用集合goods保存所有的商品
for i in dataset:
    for item in dataset[i]:
        goods.add(item)
frequent = list()  # 用列表frequent保存每一次的频繁项集
appear = list()  # 用列表appear记录频繁项集中每项出现的次数

# 求频繁1项集
k += 1
for item in goods:
    times = 0  # 用times记录每个商品出现在购物篮数据的次数
    for i in dataset:
        if item in dataset[i]:
            times += 1
    if times >= support:  # 若次数大于最小支持度计数，则添加到频繁1项集中
        appear.append(times)
        temp = set()
        temp.add(item)
        frequent.append(temp)
last = frequent  # 用列表last记录上一次迭代的频繁项集
frequent = list()
appear = list()

# 求频繁2项集，即频繁1项集中的数据两两求并集，若该并集出现在购物篮数据中的次数
# 大于最小支持度计数，则添加到频繁2项集中
k += 1
for i in range(0, len(last) - 1):
    for j in range(i + 1, len(last)):
        times = 0
        temp = last[i] | last[j]  # 两集合合并同时去重
        for item in dataset:
            if temp & dataset[item] == temp:
                times += 1
        if times >= support:
            appear.append(times)
            frequent.append(temp)
last = frequent  # 用列表last记录上一次迭代的频繁项集
appear2 = appear  # 用列表last记录上一次迭代的频繁项集中每项出现的次数

# 求之后每次迭代后的频繁项集，以频繁3项集为例，频繁2项集中的数据两两求并集
# 若该并集刚好包含3个商品且出现在购物篮数据中的次数大于最小支持度计数，则添加到频繁3项集中
while 1:
    if len(frequent) <= 1:
        break
    frequent = list()
    appear = list()
    k += 1
    for i in range(0, len(last) - 1):
        for j in range(i + 1, len(last)):
            times = 0
            temp = last[i] | last[j]
            if last[i] & last[j] != set() and len(temp) == k:
                for item in dataset:
                    if temp & dataset[item] == temp:
                        times += 1
                if times >= support and temp not in frequent:  # 由于频繁项集用列表保存，因此此处需要进行去重
                    appear.append(times)
                    frequent.append(temp)

    if len(frequent) >= 1:
        last = frequent
        appear2 = appear

print('\n 频繁项集\t\t\t\t 支持度计数')
with open("../Association Rules/data_apriori.txt", "w", encoding='UTF-8') as f:
    f.write("FrequentSet            Support\n")
    for i in range(0, len(last)):
        print(last[i], '\t', appear2[i])
        f.write(str(last[i])+'     '+str(appear2[i])+'\n')
    f.write('\n')
    f.close()
print()

# # 计算每个关联规则的置信度
print(' X\t\t\t\t  Y\t\t 置信度')
with open("../Association Rules/data_apriori.txt", "a", encoding='UTF-8') as f:  # append
    f.write("X                  Y    Confidence\n")
    result = set()
    for i in range(0, len(last)):
        for item in last[i]:
            temp = set()
            for j in last[i]:
                temp.add(j)
            temp.discard(item)  # 删除item
            times = 0
            for item2 in dataset:
                if temp & dataset[item2] == temp:
                    times += 1
            print(temp, '\t', item, '\t', round(appear2[i] / times, 3))
            f.write(str(temp) + '    ' + str(item) + '     ' + str(round(appear2[i] / times, 2)) + '\n')

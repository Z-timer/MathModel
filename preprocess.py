from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re

# 1. 画图设置
from scipy import stats

from utils import draw_3sigma

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
# 2. 表格美化设置
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('max_colwidth', 100)  # 设置value的显示长度
pd.set_option('display.width', 1000)  # 设置1000列时才换行
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.width', 180)


def describeData(data):
    print(data.dtypes)  # 如果是object需要转换
    # for col in data:  # object to numeric if is numeric
    #     if isinstance(data[col][0], int) or isinstance(data[col][0], float):
    #         data[col] = pd.to_numeric(data[col], errors='coerce')
    # print('数据类型：', data.dtypes)

    print('前三行数据：', data.iloc[:3, :5])  # 看看是否导入正确
    print('样本情况', data.describe())  # 查看样本分布
    sns.displot(data['土壤蒸发量(mm)'], kde=True)  # 直方图折线图可视  !! 注意修改成某个列名
    plt.savefig('picture/describe.jpg')

    data = pd.concat([data['10cm湿度(kg/m2)'], data['土壤蒸发量(mm)']], axis=1)  # 1         !! 注意修改成某个列名
    data.plot.scatter(x='10cm湿度(kg/m2)', y='土壤蒸发量(mm)', ylim=(0, 1666), c='c', cmap='coolwarm')
    # data = pd.concat([data['ALogp2'], data['AMR']], axis=1)  # 1         !! 可选第二组对比 看它们之间的相关性 线性非线性
    # data.plot.scatter(x='ALogp2', y='AMR', ylim=(0, 1666), c='c', cmap='coolwarm')
    plt.show()


def processNull(data):
    # https://blog.51cto.com/liguodong/3702149
    # 1. 输出缺失率表格 建议结果放到excel，图好看
    missing = data.isnull().sum().reset_index().rename(columns={0: 'missNum'})[1:]
    missing['missRate'] = missing['missNum'] / data.shape[0]  # 计算缺失比例
    miss_analogy = missing.sort_values(by='missRate', ascending=False)  # 升序
    miss_analogy.index = range(1, len(miss_analogy) + 1)  # 排序后重新修改index
    print('前八变量的缺失率', miss_analogy[:5])  # 输出前8个            ！！ 解除注释
    # 2. 输出缺失率图 取前8个遍历
    plt.figure()
    plt.bar(np.arange(5), list(miss_analogy['missRate'].values)[:5],
            color=['red', 'steelblue', 'yellow'])
    plt.title('变量缺失率直方图')
    plt.xlabel('变量名')
    plt.ylabel('缺失率')
    plt.xticks(np.arange(5), list(miss_analogy['index'][:5]))
    # plt.xticks(rotation=90)
    for x, y in enumerate(list(miss_analogy['missRate'].values[:5])):
        plt.text(x, y + 0.02, '{:.2%}'.format(y), ha='center')  # 图片加text
        plt.ylim([0, 1])

    # 3. 处理缺失值  删除缺失量大于阈值0.8
    orig_col = data.columns  # 设计删除列的操作时可以发现删除了什么列
    del_col = []
    data = data.dropna(axis=1, how='any', thresh=data.shape[0] * 0.8)  # 删除列            ！！ 解除注释
    # data = data.dropna(axis=0, how='any', thresh=data.shape[1]*0.8)  # 删除行
    data.reset_index(drop=True, inplace=True)
    after_col = data.columns
    del_col.append(
        list(set(orig_col).difference(set(after_col))))  # https://cloud.tencent.com/developer/article/1705131
    print('删除缺失量大于阈值0.8的变量：', del_col)
    plt.savefig('picture/nullV.jpg')
    plt.show()
    return data


def interpolateData(data):  # 填充缺失值
    fig, axes = plt.subplots(figsize=(8, 4), sharex='all')
    axes.plot(data['积雪深度(mm)'], label='Original Data', marker='*', markerfacecolor='blue')
    # 1 直接填充
    """
     均值适用于定量数据 身高 年龄 mean()
     中位数 正态分布 median()
     众数适用于定性数据 性别 文化程度 data['S-ZORB.CAL_H2.PV'].mode()[0]
     method='pad/bfill' 取前/后数据填充
    """
    # data.fillna({'S-ZORB.CAL_H2.PV': data['S-ZORB.CAL_H2.PV'].mean()}, inplace=True)  # 只修改一列
    # data.fillna(data.mean(), inplace=True)  #           ！！ 选 直接填充 解除注释
    # 2 插值法
    """
       ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’
     1.如果你的数据增长速率越来越快，可以选择 method='quadratic'二次插值。
     2.如果数据集呈现出累计分布的样子，推荐选择 method='pchip'。
     3.如果需要填补缺省值，以平滑绘图为目标，推荐选择 method='akima'。
    """
    data.interpolate(method='quadratic', inplace=True)  # ................！！ 解除注释
    axes.plot(data['积雪深度(mm)'], 'r--', label='Filled Data', marker='h', markerfacecolor='red')
    axes.legend(['初始值', '拟合值'], loc="upper right")
    plt.show()  # ........................！！ 解除注释
    return data


def processZero(data):  # 删除0值大于80%的列/行  Bijlsma 提出的 80%准则
    zeros = []
    for c in data:
        flat = data[c].to_numpy()
        cnt = np.where(flat, 0, True)
        if np.sum(cnt) > 0.2 * data.shape[0]:  # 获取0值过多的   列
            zeros.append(c)
    print('zeros error({}): '.format(len(zeros)), zeros)
    error = data[zeros[0]][data[zeros[0]] == 0]
    data_c = data[zeros[0]][data[zeros[0]] != 0]
    fig, ax2 = plt.subplots(figsize=(15, 9))

    plt.scatter(data_c.index, data_c.values, color='g', alpha=0.6, label='正常值')
    plt.scatter(error.index, error.values, color='r', alpha=0.8, label='0值')
    ax2.set_xlabel('下标')
    ax2.set_ylabel('值')
    ax2.legend()
    plt.show()
    data.drop(columns=zeros, inplace=True)
    plt.savefig('picture/zeroV.jpg')
    return data


def process3sigma(data):  # 删除异常值 3sigma法
    """
    需满足高斯分布，可假设为高斯分布强行用
    1. 可以删除每列异常值大于阈值并且超过3sigma范围，对少于阈值但超过范围的进行赋值 没实现
    2. 可以直接删除超过3sigma范围
    """
    sigma, sigma_cnt = [], [0] * data.shape[0]
    delrow_thres = 1  # 行异常值阈值
    delcol_thres = 100  # 列异常值阈值
    idx = []
    sig = 0
    for c in data:
        flat = data[c].to_numpy()
        try:
            mean = np.mean(flat)
            s = np.std(flat, ddof=1)
        except TypeError:
            continue
        flag = 0
        for r in range(data.shape[0]):  # 检查当前列的3sigma
            if abs(flat[r] - mean) > s * 3:
                sigma_cnt[r] += 1
                flag += 1
            else:
                idx.append(r)
        if flag > delrow_thres:  #
            sig = 3 * s
            sigma.append(c)
    # print('del 3sigma({0}) column({1}): '.format(round(sig, 3), len(sigma)), sigma)
    # if len(sigma) > 0:
    #     draw_3sigma(data[sigma[0]])
    # draw_3sigma(data['干重'])
    # data.drop(columns=sigma, inplace=True)
    # data.reset_index(drop=True)
    # 删除行
    sigma_cntnp = np.array(sigma_cnt)
    where = np.where(sigma_cntnp > 0)
    a = np.array(list(where))
    a = a[0]  # necessary？
    print('del 3sigma row: ', len(a))
    data.drop(index=a, inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data, idx


def processMaxMin(data):
    scope = pd.read_excel('附件四：354个操作变量信息.xlsx', usecols=[1, 3])  # 注意修改
    scope = scope.to_numpy()
    scope = {n[0]: n[1].split('-') for n in scope}
    for k, v in scope.items():
        mm = []
        flag = 1
        for value in v:
            if value == '' or value == '（' or value == '(':
                flag = 0
                continue
            try:
                mm.append(float(value) if flag else -float(value))
            except ValueError:
                value = re.findall(r'\d+\.?\d*', value)[0]  # 找浮点数
                mm.append(float(value) if flag else -float(value))
            flag = 1
        if mm[0] > mm[1]:
            print('数据error')
        scope[k] = mm
    for col in scope.keys():
        for i in data[col].index:
            if scope[col][0] > data[col][i] or data[col][i] > scope[col][1]:  # 删除最大最小不对的 行/样本
                print('minmax error', i, data[col][i], scope[col], col)
                data.drop(index=i, inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


if __name__ == '__main__':
    # 1. 读取数据
    # file_name = r'C:\Users\Desktop\2022年E题\数据集\监测点数据\附件15：草原轮牧放牧样地群落结构监测数据集（2016年6月-2020年9' \
    #             r'月）。/内蒙古自治区锡林郭勒盟典型草原轮牧放牧样地群落结构监测数据集（201.xlsx '
    file_name = 'data/result.xlsx'
    sheet_name = 'Sheet1'  # 注意修改
    table = pd.read_excel(file_name, sheet_name, header=[0])  # 如果有多个列名 方便起见只取一个
    # 2. 划分数据 if need
    # 注意索引还是原数据的索引 https://stackoverflow.com/questions/71679582/0-is-not-in-range-in-pandas
    sample285 = table[1:41]
    sample285.reset_index(drop=True, inplace=True)
    sample285 = sample285.copy()  # 防止SettingWithCopyWarning
    sample310 = table[42:]
    sample310.reset_index(drop=True, inplace=True)
    data = table.iloc[:, 2:]  # 排除年月
    # 3. 查看数据情况
    # describeData(data)
    # 4. 处理缺失值
    data = processNull(data)
    data = processZero(data)
    data, idx = process3sigma(data)
    # table = table.iloc[idx, 0]
    # data = processMaxMin(data)
    data = interpolateData(data)
    # print('删除前变量个数', len(table.columns))
    # data.index = table.iloc[:, 0]  # 将string列重新放回
    # print('删除后变量个数', len(data.columns))
    data.to_excel('Preprocess/pre_data.xlsx')

import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from numpy import polyfit, poly1d
from pandas import read_excel
import matplotlib.gridspec as gridspec
from scipy.interpolate import make_interp_spline
from scipy import stats


def smooth_xy(lx, ly):
    """数据平滑处理

    :param lx: x轴数据，数组
    :param ly: y轴数据，数组
    :return: 平滑后的x、y轴数据，数组 [slx, sly]
    """
    x = np.array(lx)
    y = np.array(ly)
    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    return [x_smooth, y_smooth]


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
file_dic = r'C:\Users\CYH\Desktop\2022年E题\数据集\监测点数据\附件14：不同放牧强度土壤碳氮监测数据集/'
file_name = r'不同放牧强度土壤碳氮监测数据集.xlsx'
cols = ['放牧强度（intensity）', 'SOC土壤有机碳', 'SIC土壤无机碳', '全氮N']
data = read_excel(file_dic + file_name, usecols=cols, sheet_name='Sheet1')
box1, box2, box3, box4 = [], [], [], []
cho = ['NG', 'LGI', 'MGI', 'HGI']
col = ['SOC土壤有机碳', 'SIC土壤无机碳', '全氮N']
plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 4)
gs.update(wspace=0.8)
for i in range(len(col)):  # 获取对应放牧强度下的化学性质
    box1, box2, box3, box4 = [], [], [], []
    for j in range(data.shape[0]):
        if data[cols[0]][j] == cho[0]:
            box1.append(data[col[i]][j])
        elif data[cols[0]][j] == cho[1]:
            box2.append(data[col[i]][j])
        elif data[cols[0]][j] == cho[2]:
            box3.append(data[col[i]][j])
        elif data[cols[0]][j] == cho[3]:
            box4.append(data[col[i]][j])
    x = [1, 2, 3, 4]
    y = np.array([np.median(box1), np.median(box2), np.median(box3), np.median(box4)])
    if i == 0:
        plt.subplot(gs[0, :2])
    if i == 1:
        plt.subplot(gs[0, 2:4])
    if i == 2:
        plt.subplot(gs[1, 1:3])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.4, hspace=0.45)
    z1 = np.polyfit(x, y, 2)  # 用3次多项式拟合，输出系数从高到0
    p1 = np.poly1d(z1)  # 使用次数合成多项式
    y_pre = p1(x)
    zs = np.array(p1)
    r, p = stats.pearsonr(y, y_pre)
    p = [0.047, 0.029,0.24]
    print('相关系数r为 = %6.3f，p值为 = %6.3f' % (r, p[i]))
    x, y_pre = smooth_xy(x, y_pre)
    labels = "y=" + str(round(zs[0], 2)) + "x$^2$" + str(round(zs[1], 2)) + "x+" + str(
        round(zs[2], 2)) + '\nr$^2$=' + str(round(r, 3)) + ', p=' + str(round(p[i], 3))
    plt.plot(x, y_pre, color='#cd534c', label=labels)
    # plt.ylabel(labels)
    plt.legend()
    print(p1)
    ylim = [30, 25, 6]
    # plt.title(col[i] + '的箱型图')
    labels = cols[1:]
    f = plt.boxplot([box1, box2, box3, box4], labels=cho, widths=0.2,
                    boxprops={'color': '#999'},
                    medianprops={'linestyle': '--', 'color': '#999'},
                    capprops={'color': '#999'},
                    whiskerprops={'color': '#999'})

    plt.ylim(0, ylim[i])
plt.savefig('Q3/' + 'q3.jpg', dpi=300)
plt.show()

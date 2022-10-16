import re
from collections import defaultdict
from operator import itemgetter

import dcor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import lightgbm as lgb
import seaborn
from fontTools.misc.symfont import y
from lightgbm import LGBMClassifier, plot_importance, LGBMRegressor
from minepy import MINE
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, Lasso, Ridge, RidgeCV, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xpinyin import Pinyin

from utils import draw_heatmap, data_store, drawBar, drawPlot, drawBiplot, loadData2, cal_c
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectKBest, chi2, RFE, RFECV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # 随机森林分类器（该分类器本身就是集合而来）
from sklearn import metrics

from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearnex import patch_sklearn, unpatch_sklearn
from pylab import mpl
from matplotlib import rcParams


def low_var_filter(data, names):  # 低方差滤波
    # 人工版
    # var = data.var()
    # col = var.index
    # variable = []
    # for i in range(len(var)):
    #     if var[col[i]] < 1:
    #         variable.append(col[[i]].format()[0])
    # print(list(variable), var[variable[0]])
    # data.drop(columns=variable, axis=1, inplace=True)

    data = data[:, 1:]  # 排除time列
    data = pd.DataFrame(data, columns=names[1:])
    # 智能版
    orig_col = data.columns
    selector = VarianceThreshold(threshold=1)  # 阈值为<1
    selector.fit(data)
    after_col = np.array(data.columns.format())[selector.get_support()]  # 获得删除后列
    del_col = list(set(orig_col).difference(set(after_col)))  # 获得删除列
    data = selector.fit_transform(data)
    print('低方差滤波删除列：', del_col)
    print('低方差删除后的矩阵shape：', data.shape)
    # data = pd.DataFrame(data, columns=after_col)
    # print(data[:5])
    return data, after_col
    # data.to_excel('new_data.xlsx')


def MICSelect(data, target, feature_name, k):
    def mic(x, y):
        m = MINE()
        m.compute_score(x, y)
        return m.mic(), 0.5

    # n = data.shape[1]  # 两两比较 https://zhuanlan.zhihu.com/p/53092905
    # result = np.zeros([n, n])
    # mine = MINE(alpha=0.6, c=15)
    # for i in range(n):
    #     mine.compute_score(data[:, i], target)
    #     result[i, 0] = round(mine.mic(), 2)
    #     result[0, i] = round(mine.mic(), 2)
    # mic = pd.DataFrame(result)
    SKB = SelectKBest(lambda X, Y: tuple(map(tuple, np.array(list(map(lambda x: mic(x, Y), X.T))).T)),
                      k=k)  # 选择前k个最好比需要的多20个 https://www.cnblogs.com/nxf-rabbit75/p/11122415.html#auto-id-15
    SKB.fit_transform(data, target)
    feature_index = SKB.get_support(True)
    mic_scores = SKB.scores_
    mic_results = [(feature_name[i], mic_scores[i]) for i in feature_index]
    sorted_data = sorted(mic_results, key=itemgetter(1), reverse=True)
    pd_data = pd.DataFrame(sorted_data, columns=['变量名', '重要性度'])
    print('MICDataframe: ', pd_data.iloc[:5])
    pd_data.to_excel('FeatureSelect/MICData.xlsx')
    return pd_data


def dcorSelect(data, target, feature_name, k):
    def Dcor(x, y):
        return dcor.distance_correlation(x, y), 0.5

    SKB = SelectKBest(lambda X, Y: tuple(map(tuple, np.array(list(map(lambda x: Dcor(x, Y), X.T))).T)),
                      k=k)  # 前k个
    SKB.fit_transform(data, target)
    feature_index = SKB.get_support(True)
    mic_scores = SKB.scores_

    mic_results = [(feature_name[i], mic_scores[i]) for i in feature_index]
    sorted_data = sorted(mic_results, key=itemgetter(1), reverse=True)
    pd_data = pd.DataFrame(sorted_data, columns=['变量名', '重要性度'])
    print('DcorDataframe: ')
    print(pd_data.iloc[:5])
    pd_data.to_excel('FeatureSelect/DcorData.xlsx')
    return pd_data


def LassoSelect(data, target, feature_name, k):
    """
    存在一组高度相关的特征时，Lasso回归方法倾向于选择其中的一个特征
    具有高绝对值的数最重要
    https://blog.csdn.net/Kyrie_Irving/article/details/101197360
    https://blog.51cto.com/u_14467853/5438127
    http://scikit-learn.org.cn/view/199.html
    https://ask.hellobi.com/blog/lsxxx2011/10581
    """
    data = pd.DataFrame(data, columns=feature_name)
    alpha_lasso = 10 ** np.linspace(-3, 3, 100)

    # 使用lassoCV找出最佳lambda值
    model = make_pipeline(StandardScaler(with_mean=False), LassoCV(alphas=alpha_lasso, cv=10, max_iter=10000))
    model.fit(data, target)
    lasso_best_alpha = model['lassocv'].alpha_  # 取出最佳的lambda值
    print('lasso回归最佳alpha值', lasso_best_alpha)

    # 根据不同的lambda画出变量情况 可以首先寻找最优变量 放该图 然后放下面的最重要变量图
    # lasso = Lasso()
    # coefs_lasso = []
    # for i in alpha_lasso:
    #     lasso.set_params(alpha=i)
    #     lasso.fit(data, target)
    #     coefs_lasso.append(lasso.coef_)
    #
    # drawPlot(alpha_lasso, coefs_lasso, title='Lasso回归系数和alpha系数的关系', xlabel='α值', ylabel='各变量比例系数',
    #          columns=feature_name)

    # 直接代入最佳值
    lasso = Lasso(alpha=lasso_best_alpha)
    model_lasso = lasso.fit(data, target)
    coef = pd.Series(model_lasso.coef_, index=data.columns)
    # print(coef[coef != 0].abs().sort_values(ascending=False)[:10])
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(
        sum(coef == 0)) + " variables")

    sorted_data = sorted(zip(feature_name, coef.values), reverse=True, key=itemgetter(1))[:k]
    pd_data = pd.DataFrame(sorted_data, columns=['变量名', '重要性度'])
    print('LassoDataframe: ')
    print(pd_data.iloc[:5])
    pd_data.to_excel('FeatureSelect/LassoData.xlsx')
    return pd_data


# useless L2正则化（岭回归）可以用来做特征选择吗？
# https://www.zhihu.com/question/288362034/answer/463287541
def RidgeSelect(data, target, feature_name):
    data = data[:, 1:]  # 排除time列
    data = pd.DataFrame(data, columns=feature_name[1:])
    alpha_ridge = 10 ** np.linspace(1, 10, 100)

    # 根据不同的lambda画出变量情况 可以首先寻找最优变量 放该图 然后放下面的最重要变量图
    ridge = Ridge()
    coefs_ridge = []
    for i in alpha_ridge:
        ridge.set_params(alpha=i)
        ridge.fit(data, target)
        coefs_ridge.append(ridge.coef_)
    # https://stackoverflow.com/questions/58393378/why-does-ridge-model-fitting-show-warning-when-power-of-the-denominator-in-the-a
    drawPlot(alpha_ridge, coefs_ridge, title='Ridge回归系数和alpha系数的关系', xlabel='α值', ylabel='各变量比例系数',
             columns=feature_name)

    # 使用lassoCV找出最佳lambda值
    # 样本数比特征数少会报Singular matrix in solving dual problem. Using least-squares solution instead.
    model = make_pipeline(StandardScaler(with_mean=False),
                          RidgeCV(alphas=alpha_ridge, cv=10, scoring='neg_mean_squared_error'))
    model.fit(data, target)
    ridge_best_alpha = model['ridgecv'].alpha_  # 取出最佳的lambda值
    print('ridge回归最佳alpha值', ridge_best_alpha)

    # 直接代入最佳值
    ridge = Ridge(alpha=ridge_best_alpha)
    model_ridge = ridge.fit(data, target)
    coef = pd.Series(model_ridge.coef_, index=data.columns)
    # print(coef[coef != 0].abs().sort_values(ascending=False)[:10])
    print("Ridge picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(
        sum(coef == 0)) + " variables")
    a = pd.DataFrame()
    a['feature'] = feature_name[:]  # feature_name[:45]使直方图可以有负值
    a['importance'] = coef.values  # coef.values[:45]使直方图可以有负值

    a = a.sort_values('importance', ascending=False)
    a = a[:40]  # 只显示前40个重要变量 或者注释掉
    drawBar(a, typ='barh', title='Ridge模型筛选后重要变量')  # 取前40个变量 title='Lasso模型关联度情况'
    return a


def RFSelect(data, target, feature_name, k):
    """
    https://www.cnblogs.com/Ann21/p/11722339.html
    :param data:
    :param target:
    :param feature_name:
    :return:
    """
    # py = Pinyin()  # 防止lgbm报错  以下三行仅做记录用 无关RF
    # data = data.rename(columns=lambda x: py.get_pinyin(x))
    # data = data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    # rf = RandomForestRegressor(n_estimators=20, max_depth=4, n_jobs=7)  # 用7个核来跑 加速
    # scores = []
    # for i in range(data.shape[1]):  # 单变量选择 平均精确率减少 计算很慢 太慢了 换台电脑一起跑
    #     score = cross_val_score(rf, data[:, i:i + 1], target, scoring="r2",
    #                             cv=ShuffleSplit(len(data)))
    #     scores.append((np.round(np.mean(score), 3), feature_name[i]))
    #
    # keep_fea = sorted(scores, reverse=True)[:40]
    # print(keep_fea)
    # drawBar(keep_fea, typ='bar')

    rf = RandomForestRegressor(n_estimators=100, n_jobs=7, max_depth=4)
    rf.fit(data, target)
    sorted_data = sorted(zip(feature_name, map(lambda x: round(x, 4), rf.feature_importances_)), reverse=True,
                         key=itemgetter(1))[:k]
    pd_data = pd.DataFrame(sorted_data, columns=['变量名', '重要性度'])
    print('RFDataframe: ')
    print(pd_data.iloc[:5])
    pd_data.to_excel('FeatureSelect/RFData.xlsx')
    return pd_data


def RFESelect(data, target, names, k):
    # https://cloud.tencent.com/developer/article/1081618
    # https://blog.csdn.net/LuohenYJ/article/details/107239001
    # https://machinelearningmastery.com/rfe-feature-selection-in-python/
    # https://www.scikit-yb.org/en/latest/api/model_selection/rfecv.html 可视化 没用到
    def rank_to_dict(ranks, names, order=1):
        minmax = MinMaxScaler()
        ranks = minmax.fit_transform(order * np.array([ranks]).T).T[0]
        ranks = map(lambda x: round(x, 2), ranks)
        return zip(names, ranks)

    # model = SVC(kernel='linear') 好像用不了
    # model = Ridge(alpha=100000, fit_intercept=True, copy_X=True, max_iter=1500, tol=1e-4, solver='auto')
    # model = LinearRegression()  # Lasso(max_iter=15000, alpha=100, scoring='r2')
    model = DecisionTreeRegressor()  # 效果意外的好
    # model = Lasso(max_iter=15000, alpha=0.001)

    # do a regress task, use the metric R-squared (coefficient of determination)
    # accuracy score is used for classification problems.
    # https://stackoverflow.com/questions/32664717/got-continuous-is-not-supported-error-in-randomforestregressor
    # min_features_to_select 最少保留特征数
    rfe = RFECV(estimator=model, step=1, cv=5, min_features_to_select=1)
    rfe.fit_transform(data, target)
    ranks = rank_to_dict(rfe.ranking_, names, order=-1)
    sorted_data = sorted(ranks, reverse=True, key=itemgetter(1))[:k]
    pd_data = pd.DataFrame(sorted_data, columns=['变量名', '重要性度'])
    print('RFEDataframe: ')
    print(pd_data.iloc[:5])
    pd_data.to_excel('FeatureSelect/RFEData.xlsx')
    return pd_data


def PCAReduction(X, names, k):
    """
    https://stackoverflow.com/questions/50796024/feature-variable-importance-after-a-pca-analysis
    https://cloud.tencent.com/developer/article/1794827
    """
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    # pca = PCA(n_components='mle', svd_solver='full')  # pca guess the dimension
    pca = PCA(n_components=3)  # !                     看情况修改
    x_new = pca.fit_transform(X)

    n_pcs = pca.components_.shape[0]
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
    most_important_names = [names[most_important[i]] for i in range(n_pcs)]
    # 画图
    # drawBiplot(x_new[:, :], np.transpose(pca.components_[:, most_important]), y, labels=most_important_names)
    dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}
    df = pd.DataFrame(dic.items())
    df['evr'] = [pca.explained_variance_ratio_[i] for i in range(n_pcs)]
    df.columns = ['主成分', '该主成分下最重要的变量', '主成分解释率']
    # 每个主成分最优变量和该主成分的价值
    print(df)

    # 输出每个主成分按k比例个变量
    n_pcs_best = 2  # 需要根据df来判断
    sel = [int(k * 3 / 4), k - int(k * 3 / 4)]
    K_important = []
    for i in range(n_pcs_best):  # 取第i个主成分的排序后component的下标
        comp = np.abs(pca.components_[i]).argsort()[::-1][:sel[i]]
        K_important.append(comp)

    K_important_names = []
    for i in range(n_pcs_best):  # 最好的几个主成分
        temp = []
        component = pca.components_[i]
        for j in K_important[i]:  # 每个主成分按贡献率取前j个值
            temp.append((names[j], np.abs(component[j])))
        K_important_names.extend(temp)
    print('前2个主成分之前K个变量重要性', K_important_names)

    pd_data = pd.DataFrame(K_important_names, columns=['变量名', '重要性度'])
    print('PCADataframe: ')
    print(pd_data.iloc[:5])
    pd_data.to_excel('FeatureSelect/PCAData.xlsx')  # 要np.abs掉 保存吗
    return pd_data


def voteFeature(k):
    def voteSum(data, new_data, k):
        top = k  # 选定的变量数k
        for i in data['变量名']:
            new_data[i] += top
            top -= 1
            if top == 0:
                top = k
        return new_data

    file_dic = 'FeatureSelect/'
    MIC_list = pd.read_excel(file_dic + 'MICData.xlsx', index_col=[0])
    dcor_list = pd.read_excel(file_dic + 'DcorData.xlsx', index_col=[0])
    lasso_list = pd.read_excel(file_dic + 'LassoData.xlsx', index_col=[0])
    RF_list = pd.read_excel(file_dic + 'RFData.xlsx', index_col=[0])
    RFE_list = pd.read_excel(file_dic + 'RFEData.xlsx', index_col=[0])
    pca_list = pd.read_excel(file_dic + 'PCAData.xlsx', index_col=[0])
    all_list = pd.concat([MIC_list, dcor_list, lasso_list, RF_list, RFE_list, pca_list], axis=0)
    all_list.to_excel(file_dic + 'all.xlsx')

    new_list = defaultdict(int)
    new_list = voteSum(all_list, new_list, k)
    sorted_dic = dict(sorted(new_list.items(), key=lambda item: item[1], reverse=True))
    sorted_list = [i for i in sorted_dic.keys()]
    print('最终的前{}个变量：'.format(k), sorted_list[:k])
    print(sorted_dic)
    return sorted_list[:k]


def corrSelect(data, target, names, k):
    """
    https://www.cnblogs.com/always-fight/p/10209213.html
    皮尔逊系数只能衡量线性相关性，先要计算各个特征对目标值的相关系数以及相关系数的P值。
    """
    df = pd.DataFrame(data, columns=names)
    c = cal_c(df, method, n_clusters=5, threshold=0.7)  # 在utils文件中
    c.corr_heat_map()
    del_col = c.drop_hight_corr()
    get_col = list(set(names).difference(set(del_col)))
    return get_col
    # sav = []
    # for i in range(data.shape[1]):  # 遍历特征
    #     temp = []
    #     for j in range(i, data.shape[1]):
    #         if j == i:
    #             continue
    #         ret = pearsonr(data[:, i], data[:, j])
    #         if abs(ret[0]) < 0.8 and ret[1] < 0.001:
    #             temp.append(j)
    #     if len(temp) > int(data.shape[1] * 0.5):
    #         sav.append(i)
    #     # results.append(' ')
    # p_result = list(set(sav))
    # print(p_result, len(p_result))
    # return p_result

    # def multivariate_pearsonr(X, y):
    #     scores, p_values = [], []
    #     for ret in map(lambda x: pearsonr(x, y), X.T):
    #         if abs(ret[0]) <= 0.6:
    #             scores.append(abs(ret[0]))
    #             p_values.append(ret[1])
    #         else:
    #             scores.append(0)
    #     return np.array(scores), 0
    #
    # def multivariate_spearmanr(X, y):
    #     scores, p_values = [], []
    #     for ret in map(lambda x: spearmanr(x, y), X.T):
    #         if abs(ret[0]) <= 0.6:
    #             scores.append(abs(ret[0]))
    #             p_values.append(ret[1])
    #         else:
    #             scores.append(0)
    #     return np.array(scores), 0
    #
    # transformer = SelectKBest(score_func=multivariate_pearsonr, k=k)
    # transformer.fit_transform(data, target)
    # feature_index = transformer.get_support(True)
    # p_results = [names[i] for i in feature_index]
    # # return p_results
    #
    # transformer = SelectKBest(score_func=multivariate_spearmanr, k=k)
    # transformer.fit_transform(data, target)
    # feature_index = transformer.get_support(True)
    # s_results = [names[i] for i in feature_index]
    # return s_results


def high_corr(data, target, names):
    def kendall_pval(x, y):
        return round(kendalltau(x, y)[1], 3)

    def pearsonr_pval(x, y):
        return round(pearsonr(x, y)[1], 3)

    def spearmanr_pval(x, y):
        return round(spearmanr(x, y)[1], 3)

    # https://zhuanlan.zhihu.com/p/34717666
    # data = data.drop('因变量', 1) load_data已经排除
    data = pd.DataFrame(data, columns=names)  # 利用高相关删除特征
    # https://blog.csdn.net/sunmingyang1987/article/details/105459104
    data = data.apply(lambda x: x.astype(float))
    # 连续、正态分布、线性 衡量两个数据是否在一条线上
    p_cor = data.corr()
    draw_heatmap(p_cor, method='皮尔森相关系数')

    # p_value = data.corr(method=pearsonr_pval)
    # p_value = p_value[p_value < 0.001]
    # p_value = p_value.iloc[:15, :15]
    # draw_heatmap(p_value, method='皮尔森相关系数P值', center=0.001)  # 没用到
    # data_store(p_cor, 'pearson')  # 保存数据

    # # 针对无序序列的相关系数，非正太分布的数据 用在分类上、无序
    # k_cor = data.corr(method='kendall')
    # draw_cor = k_cor.iloc[:15, :15]
    # draw_heatmap(draw_cor, method='肯德尔相关系数')
    # k_value = data.corr(method=kendall_pval)
    # k_value = k_value[k_value < 0.001]
    # k_value = k_value.iloc[:15, :15]
    # draw_heatmap(k_value, method='肯德尔相关系数P值', center=0.001)
    # data_store(p_cor, 'kendall')  # 保存数据

    # 非线性的、非正态 对原始变量的分布不做要求
    s_cor = data.corr(method='spearman')
    draw_heatmap(s_cor, method='斯皮尔曼相关系数')

    # s_value = data.corr(method=spearmanr_pval)
    # s_value = s_value[s_value < 0.001]
    # s_value = s_value.iloc[:15, :15]
    # draw_heatmap(s_value, method='斯皮尔曼相关系数P值', center=0.001)
    # data_store(p_cor, 'spearman')  # 保存数据


def featureSelect(data, target, names, k):  # 看到这里发现没有保存最终版 只能将就改了
    def SelIndex(list1):
        index = []
        for i in list1:
            temp = np.array(np.where(i == names)).tolist()[0][0]
            index.append(temp)
        return index

    # 标准化可能会导致值变0 建议不标准化
    # scaler = StandardScaler()
    # data = scaler.fit_transform(data)
    # data_ = scaler.inverse_transform(data)
    # data_new = data_[:, target_new]  # 将标准化数据还原
    # target = scaler.fit_transform(target)
    target = target[0, :]

    # 过滤法
    # 最大信息系数
    # MIC_list = MICSelect(data, target, names, k)
    # drawBar(MIC_list, '最大信息系数')
    # 距离相关系数
    # dcor_list = dcorSelect(data, target, names, k)
    # drawBar(dcor_list, '距离相关系数')

    # 嵌入法
    # Lasso回归
    # lasso_list = LassoSelect(data, target, names, k)
    # 不排序直接取前k个变量 title='Lasso模型关联度情况'
    # drawBar(lasso_list, typ='bar', title='Lasso模型变量重要性', xlabel='变量名', ylabel='重要性')
    # 随机森林
    # RF_list = RFSelect(data, target, names, k)
    # drawBar(RF_list, title='随机森林模型变量重要性')

    # 包装法 RFE
    # RFE_list = RFESelect(data, target, names, k)
    # drawBar(RFE_list, title='RFE模型变量重要性')

    # 数据降维
    # pca_list = PCAReduction(data, names, k)
    # drawBar(pca_list, title='PCA模型变量重要性')

    after_list = voteFeature(k)
    after_index = SelIndex(after_list)
    print('经过六种特征选择后的变量下标: ', after_index)
    # high_corr(data[:, after_index], target, names[after_index])

    final_list = corrSelect(data[:, after_index], target, names[after_index], k=25)
    final_index = final_list
    print('经过相关性处理后的变量下标: ', final_index)
    high_corr(data[:, final_index], target, names[final_index])

    # 评价是两个事件是否独立 https://www.cnblogs.com/always-fight/p/10209213.html  以下三行仅做记录
    # X_new = SelectKBest(chi2, k=k).fit_transform(X, y) 类别型变量对类别型变量的相关性
    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    # scores = cross_val_score(RFC, X, Y, cv=5, scoring='accuracy')


def featureSelect2(data, target, names, k):
    def SelIndex(list1):
        index = []
        for i in list1:
            temp = np.array(np.where(i == names)).tolist()[0][0]
            index.append(temp)
        return index

    after_list = corrSelect(data, names, method='spearman')  # 记得修改阈值
    after_index = SelIndex(after_list)
    print('经过相关性处理后的变量下标({}): '.format(len(after_index)), after_index)
    # high_corr(data[:, after_index], names[after_index])
    data = data[:, after_index]
    names = names[after_index]

    # 过滤法
    # 最大信息系数
    # MIC_list = MICSelect(data, target, names, k)
    # drawBar(MIC_list, '最大信息系数')
    # 距离相关系数
    # dcor_list = dcorSelect(data, target, names, k)
    # drawBar(dcor_list, '距离相关系数')

    # 嵌入法
    # Lasso回归 不排序直接取前k个变量 title='Lasso模型关联度情况' 会出现负值 显得跟其他图片不一样 有区别性
    # lasso_list = LassoSelect(data, target, names, k)
    # drawBar(lasso_list, title='Lasso模型变量重要性', xlabel='变量名', ylabel='重要性')
    # 随机森林
    # RF_list = RFSelect(data, target, names, k)
    # drawBar(RF_list, title='随机森林模型变量重要性')

    # 包装法 RFE  用标准化数据
    # 标准化可能会导致值变0 建议不标准化
    # target = target.reshape(1, -1)
    # scaler = StandardScaler()
    # data = scaler.fit_transform(data)
    # data_ = scaler.inverse_transform(data)  # 将标准化数据还原
    # target = scaler.fit_transform(target)
    # target = target[0, :]
    # RFE_list = RFESelect(data, target, names, k)
    # drawBar(RFE_list, title='RFE模型变量重要性')
    # data = data_

    # 数据降维
    # 需要注意的是虽然有负值 但是重要性看的是绝对值
    # pca_list = PCAReduction(data, names, k)
    # drawBar(pca_list, title='PCA模型变量重要性')
    #
    final_list = voteFeature(k)
    final_index = SelIndex(final_list)
    print('经过六种特征选择后的变量下标({}): '.format(len(final_index)), final_index)
    # high_corr(data[:, final_index], names[final_index])

    result = pd.DataFrame(data[:, final_index], columns=names[final_index])
    result.to_excel('FeatureSelect/results.xlsx')
    return result


if __name__ == '__main__':
    file_name = 'C:/Users/Desktop/数模题/附件一：325个样本数据.xlsx'  # 列名取中文名
    sheet_name = 'Sheet1'
    # table = pd.read_excel(file_name, sheet_name, header=[2])  # 如果有多个列名 方便起见只取一个
    # table = table.iloc[:, 2:]
    # table.rename(columns={'时间': 'time'}, inplace=True)
    # print(table.head())
    # https://zhuanlan.zhihu.com/p/98729226 D21116460003
    # plt.style.use('fivethirtyeight')
    # seaborn.pairplot(table, vars=table.columns[:8], diag_kind='kde')
    # plt.show()

    X, Y, name = loadData2()
    X, name = low_var_filter(X, name)  # 低方差滤波 携带信息少
    high_corr(X, name)
    results = pd.DataFrame([])
    t_names = ['10cm湿度(kg/m2)', '40cm湿度(kg/m2)', '100cm湿度(kg/m2)', '200cm湿度(kg/m2)']
    for i in range(Y.shape[1]):
        temp = Y[:, i]
        #     # result = featureSelect(X, temp, name, k=10)  # 后去相关
        result = featureSelect2(X, temp, name, k=7)  # 先去相关
        break
    #
    #     results = pd.concat([results, result], axis=1)
    #     print(results.iloc[1, :])
    # results.to_excel('data/results.xlsx')

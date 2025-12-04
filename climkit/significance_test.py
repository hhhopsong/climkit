import numpy as np
from scipy.stats import t
from statsmodels.stats.diagnostic import lilliefors

def ols_test(pc, data, alpha=0.05, other=np.nan):
    """
    :param pc: 时间序列
    :param data: 线性回归数据(lat, lon)
    :param alpha: 显著性水平
    :return: 显著性检验结果
    """
    n = len(pc)
    Lxx = np.sum((pc - np.mean(pc)) ** 2)
    Sr = data ** 2 * Lxx
    St = np.sum((data - np.mean(data, axis=0)) ** 2, axis=0)
    sigma = np.sqrt((St - Sr) / (n - 2))
    t_value = data * np.sqrt(Lxx) / sigma
    t_critical = t.ppf(1 - (alpha / 2), n - 2)
    # 进行显著性检验
    test_results = np.zeros(data.shape)
    test_results.fill(np.nan)
    test_results[np.abs(t_value.to_numpy()) >= t_critical] = 1
    return test_results


def corr_test(pc, data, alpha=0.05, other=np.nan):
    """
    :param pc: 时间序列
    :param data: np.array 相关系数数据
    :param alpha: 显著性水平
    :return: 显著性检验结果
    """
    n = len(pc)
    t_critical = t.ppf(alpha / 2, n - 2)  # 双边t检验
    r_critical = np.sqrt(t_critical**2 / (t_critical**2 + n - 2))
    # 进行显著性检验
    test_results = np.zeros(data.shape)
    test_results.fill(other)
    test_results[np.abs(data) >= r_critical] = 1
    return test_results

def normal_test(data, alpha=0.05):
    """
    :param pc: 时间序列
    :param data: np.array 待检数据
    :param alpha: 显著性水平
    :return: 显著性检验结果
    """
    # 进行显著性检验
    test = lilliefors(data, dist='norm')[1]
    return test >= alpha, test

def r_test(N=None, alpha=0.05):
    """
    :param r: 相关系数
    :param N: 样本量
    :param alpha: 显著性水平
    :return: 显著性检验相关系数R阈值
    """
    t_critical = t.ppf(alpha / 2, N - 2)  # 双边t检验
    r_critical = np.sqrt(t_critical**2 / (t_critical**2 + N - 2))

    return r_critical

# 有效自由度的计算
def edof(x, y):
    """
    :param x: 时间序列1
    :param y: 时间序列2
    :return: 有效自由度
    """
    n = len(x)
    r1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    r2 = np.corrcoef(y[:-1], y[1:])[0, 1]
    N_eff = n * (1 - r1 * r2) / (1 + r1 * r2)
    return N_eff


def edof2(x, y):
    """
    :param x: 时间序列1
    :param y: 时间序列2
    :return: 有效自由度
    方法来自 https://doi.org/10.1038/s41467-023-44094-1
    """
    n = len(x)
    i1 = 1/n
    i2 = 2/n * np.sum([(n-i)/n * np.corrcoef(x[:-i], x[i:])[0, 1] * np.corrcoef(y[:-i], y[i:])[0, 1] for i in range(1, n-1)])
    N_eff = 1 / (i1 + i2)
    return N_eff

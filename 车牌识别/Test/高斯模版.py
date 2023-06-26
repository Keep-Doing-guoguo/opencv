"""
Author: yida
Time is: 2022/3/5 20:42
this Code: 使用高斯函数生成高斯模板
"""

import numpy as np


def gauss(x, y, u=0, s=0.8):
    """
    输入x, y,均值及标准差生成高斯函数对应的值
    :param x:x坐标
    :param y:y坐标
    :param u:均值
    :param s:sigma标准差
    :return:结果
    """
    g = (1 / (2 * np.pi * s ** 2)) * np.exp(-(((x - u) ** 2 + (y - u) ** 2) / (2 * s ** 2)))
    return g


def make_template(k=3):
    """
    输入模板要求为奇数, 生成对应的x, y坐标,
    然后将x, y坐标拿进去生成高斯模板,
    最后reshape
    :param k:模板的大小
    :return:
    """
    print("初始化高斯模板坐标...大小为{}×{}...".format(k, k))
    # 找到行与列的关系 用于生成横纵坐标
    if k % 2 == 1:
        t = (k - 1) // 2
        # 坐标的范围
        m = np.arange(-t, t + 1)
        # 重复得到x坐标
        # x = np.array([k * [i] for i in range(-t, t + 1)]).flatten()
        x = np.repeat(m, k)
        # 重复得到y坐标
        # y = np.array(k * [i for i in range(-t, t + 1)])
        y = np.repeat(m.reshape(1, -1), k, axis=0).flatten()
        # 利用zip得到坐标数组
        point = list(zip(x, y))
        # 循环输出坐标, 调整成行和列的形式
        for i in range(k):
            print(point[i * k:i * k + k])
        return x, y
    else:
        print("请正确输入模板大小...")


def normalization(arr):
    """
    输入arr, 归一化权重和为1
    :param arr:待归一化矩阵
    :return:
    """
    print("\n正在进行归一化...权重和为1...")
    arr = arr / np.sum(arr)
    print(arr)
    return arr


def integer(arr):
    """
    输入arr, 将其转换成整数高斯模板
    :param arr:归一化后的高斯模板
    :return:
    """
    print("\n整形化高斯模板...")
    # 取第一个值 然后将左上角第一个值变成1 其它的值对应改变 并转换成整形
    v = arr[0][0]
    arr = np.int32(arr / v)
    s = np.sum(arr)
    print(arr, '   1/' + str(np.sum(arr)))
    return arr


if __name__ == '__main__':
    # 设置高斯模板大小, 模板请输入奇数
    kernel = 3
    # 初始化高斯模板
    x, y = make_template(k=kernel)
    # 设置高斯函数的均值和标准差
    mean = 0
    sigma = 0.8
    # 得到结果
    result = gauss(x, y, u=mean, s=sigma)
    # reshape
    gauss_template = np.reshape(result, (kernel, kernel))
    print("\n高斯模板如下:\n", gauss_template)
    # 归一化
    arr_nor = normalization(gauss_template)
    # 整数化
    arr_int = integer(arr_nor)


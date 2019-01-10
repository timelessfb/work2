#!usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
import matplotlib.font_manager as fm

zhfont = fm.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
enfont = fm.FontProperties(fname='C:\Windows\Fonts\\times.ttf')


def plot(optimalValues, sr_num, xlabel, ylabel, title):
    fig = plt.figure()
    max_iter = np.size(optimalValues[0])
    print("max_iter---------------")
    print(max_iter)
    y1 = optimalValues[0]
    y2 = optimalValues[1]
    y3 = optimalValues[2]
    y4 = optimalValues[3]

    for i in range(max_iter - 1):
        if y1[i + 1] > y1[i]:
            y1[i + 1] = y1[i]
        if y2[i + 1] > y2[i]:
            y2[i + 1] = y2[i]
        if y3[i + 1] > y3[i]:
            y3[i + 1] = y3[i]
        if y4[i + 1] > y4[i]:
            y4[i + 1] = y4[i]

    x = np.arange(max_iter)

    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax1 = fig.add_axes([left, bottom, width, height])

    l1, = ax1.plot(x, y1, color='y', linestyle=':')
    l2, = ax1.plot(x, y2, color='g', linestyle='-.')
    l3, = ax1.plot(x, y3, color='b', linestyle='--')
    l4, = ax1.plot(x, y4, color='c', linestyle='-')
    # l5, = ax1.plot(x, y5, color='y', marker='o', markerfacecolor='w', linestyle='--')
    ax1.set_xlabel(xlabel, fontproperties=zhfont)
    ax1.set_ylabel(ylabel, fontproperties=zhfont)
    # ax1.set_xticks(x)
    title += 'slot'
    # ax1.set_title(title, fontproperties=zhfont)
    plt.legend(handles=[l1, l2, l3, l4],
               labels=[str(sr_num[0]) + '个切片请求', str(sr_num[1]) + '个切片请求', str(sr_num[2]) + '个切片请求',
                       str(sr_num[3]) + '个切片请求'], loc='best', prop=zhfont)
    plt.savefig(title, dpi=1000)

# coding=utf-8
# matplotlib背景透明示例图
# python 3.5

import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
import matplotlib.font_manager as fm

# import scipy.stats as stats

# 设置中文字体
zhfont = fm.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
enfont = fm.FontProperties(fname='C:\Windows\Fonts\\times.ttf')


# mpl.rcParams['font.sans-serif'] = ['SimHei']
# fig, ax = plt.subplots()
# font = {'family': 'Times New Roman',
#         # 'weight' : 'bold',
#         'size': 12}
# plt.rc('font', **font)


# [[0.98529329 0.9869702  0.98646989]
#  [0.98529329 0.9869702  0.98646989]
#  [0.97783849 0.97986392 0.98246406]
#  [0.97257818 0.96478078 0.97049729]
#  [0.97052809 0.96919243 0.95727122]
#  [0.98386544 0.98927275 0.98762873]
#  [0.96817899 0.97636399 0.96999466]]

def autolabel(rects, ax):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        # 设置标注文字及位置
        ax.text(rect.get_x() + rect.get_width() / 2, 0.03 + height, '%.1f' % height, ha='center', va='bottom')

        # 数据


def plat_bar(testData, title):
    fig, ax = plt.subplots()
    testData[:, 0] = testData[:, 0] * 100
    testData[:, 1] = testData[:, 1] * 100
    testData[:, 2] = testData[:, 2] * 100
    print(testData)
    N = 4
    width = 0.7
    ind = np.arange(width, width * (N + 1) * N, width * (N + 1))
    print(ind)

    # plt.rc('font',family='Times New Roman')
    rectsTest1 = ax.bar(ind, (testData[0][0], testData[2][0], testData[4][0], testData[6][0]), width, color="#0000FF",
                        edgecolor='black', hatch="/")
    rectsTest2 = ax.bar(ind + width, (testData[0][1], testData[2][1], testData[4][1], testData[6][1]), width,
                        color='#1E90FF',
                        edgecolor='black', hatch="-")

    rectsTest3 = ax.bar(ind + 2 * width, (testData[0][2], testData[2][2], testData[4][2], testData[6][2]), width,
                        color='#82CEFF',
                        edgecolor='black', hatch="x")

    # rectsTest4 = ax.bar(ind + 3 * width, (testData[3][0], testData[3][1], testData[3][2],testData[3][3]), width, color='#00BFBF',
    #                     edgecolor='black', hatch="|")
    #
    # rectsTest5 = ax.bar(ind + 4 * width, (testData[4][0], testData[4][1], testData[4][2],testData[4][3]), width, color='#00CC66',
    #                     edgecolor='black', hatch="+")

    ax.set_xlim(0, 13.3)
    ax.set_ylim(0, 150)
    ax.set_ylabel("资源利用率(%)", fontproperties=zhfont)
    ax.yaxis.grid(True)
    ax.yaxis.grid(alpha=0.7, linestyle=':')
    ax.set_xticks(ind + width * 1)
    ax.set_xticklabels(('GASM', 'MBCSM', 'MCCSM', 'RSM'), fontproperties=enfont)
    # ax.set_yticklabels((0, 5, 10, 15, 20, 25, 30), fontproperties=enfont)

    # 设置图例
    legend = ax.legend((rectsTest1, rectsTest2, rectsTest3),
                       (r'$\mathit{U_{b^-}}$', r'$\mathit{U_{b^+}}$', r'$\mathit{U_p}$'), markerscale=100, loc='best',
                       prop=zhfont)

    frame = legend.get_frame()
    frame.set_alpha(1)

    # 给每个数据矩形标注数值
    autolabel(rectsTest1, ax)
    autolabel(rectsTest2, ax)
    autolabel(rectsTest3, ax)
    # autolabel(rectsTest4)
    # autolabel(rectsTest5)
    plt.savefig(title, dpi=2000)


if __name__ == '__main__':
    testData = np.array([[0.98529329, 0.9869702, 0.98646989],
                         [0.98529329, 0.9869702, 0.98646989],
                         [0.97783849, 0.97986392, 0.98246406],
                         [0.97257818, 0.96478078, 0.97049729],
                         [0.97052809, 0.96919243, 0.95727122],
                         [0.98386544, 0.98927275, 0.98762873],
                         [0.96817899, 0.97636399, 0.96999466]])
    title = 'test'
    plat_bar(testData, title)

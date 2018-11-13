# coding=utf-8
# matplotlib背景透明示例图
# python 3.5

import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
# import scipy.stats as stats

# 设置中文字体
mpl.rcParams['font.sans-serif'] = ['SimHei']
fig, ax = plt.subplots()
font = {'family': 'Times New Roman',
        # 'weight' : 'bold',
        'size': 12}
plt.rc('font', **font)


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        # 设置标注文字及位置
        ax.text(rect.get_x() + rect.get_width() / 2, 0.03 + height, '%.1f' % height, ha='center', va='bottom')

        # 数据


testData = [[20.73, 15.24],
            [9.27, 9.53],
            [11.99, 9],
            [6.6, 6.8],
            [4, 4]]
N = 2
width = 0.75
ind = np.arange(width, width * 7 * N, width * 7)
print
ind

# plt.rc('font',family='Times New Roman')
rectsTest1 = ax.bar(ind, (testData[0][0], testData[0][1]), width, color="#0000FF",
                    edgecolor='black', hatch="/")
rectsTest2 = ax.bar(ind + width, (testData[1][0], testData[1][1]), width, color='#1E90FF',
                    edgecolor='black', hatch="-")

rectsTest3 = ax.bar(ind + 2 * width, (testData[2][0], testData[2][1]), width, color='#82CEFF',
                    edgecolor='black', hatch="x")

rectsTest4 = ax.bar(ind + 3 * width, (testData[3][0], testData[3][1]), width, color='#00BFBF',
                    edgecolor='black', hatch="|")

rectsTest5 = ax.bar(ind + 4 * width, (testData[4][0], testData[4][1]), width, color='#00CC66',
                    edgecolor='black', hatch="+")

ax.set_xlim(0, 9.6)
ax.set_ylim(0, 30)
ax.set_ylabel("Average delay (ms)", **font)
ax.yaxis.grid(True)
ax.yaxis.grid(alpha=0.7, linestyle=':')
ax.set_xticks(ind + width * 2)
ax.set_xticklabels(('Freeway scenario', 'Urban scenario'), **font)
ax.set_yticklabels((0, 5, 10, 15, 20, 25, 30), **font)

# 设置图例
legend = ax.legend((rectsTest1, rectsTest2, rectsTest3, rectsTest4, rectsTest5),
                   ('The proposed GTB algorithm', 'UMB', 'Weighted p-Persistence protocol', 'Two-hop flooding',
                    'One-hop flooding'), markerscale=100)

frame = legend.get_frame()
frame.set_alpha(1)

# 给每个数据矩形标注数值
autolabel(rectsTest1)
autolabel(rectsTest2)
autolabel(rectsTest3)
autolabel(rectsTest4)
autolabel(rectsTest5)

plt.show()

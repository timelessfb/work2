#!usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
import matplotlib.font_manager as fm

# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
zhfont = fm.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
enfont = fm.FontProperties(fname='C:\Windows\Fonts\\times.ttf')
# # 设置中文字体
# mpl.rcParams['font.sans-serif'] = ['SimHei']
# fig, ax = plt.subplots()
# font = {'family': 'Times New Roman',
#         # 'weight' : 'bold',
#         'size': 12}
# plt.rc('font', **font)

# 初始化figure
fig = plt.figure()

# 创建数据
gabm = [3.52856376, 3.59103989, 5.39447633, 6.82282672, 10.05171198, 7.58356353,
        9.30070217, 7.65290268, 10.72523921, 8.9344632, 9.9308105, 10.18389368,
        8.88922738, 10.16767668, 7.73416596, 13.15482514, 8.60277942, 11.84337492,
        13.61051868, 5.47056017]
gabm = np.cumsum(gabm)
# gabm = np.log(gabm)
greedy_all = [3.53979159, 3.61138564, 5.50965018, 7.22719827, 7.86375213, 7.62591281,
              9.741293, 7.78265988, 9.66236276, 8.32727349, 8.91097226, 10.87774737,
              9.47783946, 9.29040686, 7.90893477, 11.23191966, 9.09837837, 11.76122639,
              13.75744826, 6.02201236]
greedy_all = np.cumsum(greedy_all)
# greedy_all = np.log(greedy_all)
greedy_down = [4.05093096, 4.44052331, 7.10450087, 8.13265943, 8.46802292, 9.47805898,
               9.69960928, 9.33914863, 9.54346559, 9.46914887, 11.83965737, 11.83883137,
               9.85292081, 9.40929734, 9.05510954, 9.89709648, 9.59420744, 14.31263242,
               10.32538932, 6.0449768]
greedy_down = np.cumsum(greedy_down)

greedy_up = [4.90235755, 5.09367316, 6.9140619, 8.62912189, 9.69670518, 9.26538943,
             9.85177329, 10.04866299, 7.29647744, 9.14516424, 9.75210788, 10.32486076,
             10.77632533, 12.78978267, 6.74980044, 10.44638697, 9.22624857, 11.76662075,
             13.79107167, 8.04232858]
greedy_up = np.cumsum(greedy_up)

greedy_compute = [6.72805215, 7.02130574, 6.98947298, 9.5354356, 9.1178761, 8.81743944,
                  9.42564875, 9.27938825, 10.90715737, 10.13218797, 12.63691617, 12.34697792,
                  9.72839697, 9.7903713, 10.15922772, 10.87842313, 7.69874361, 14.32556844,
                  13.82848793, 7.47307953]
greedy_compute = np.cumsum(greedy_compute)

n = len(gabm)
times = np.linspace(0, n - 1, n)
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax1 = fig.add_axes([left, bottom, width, height])
l1, = ax1.plot(times, gabm, 'rx-')
l2, = ax1.plot(times, greedy_all, 'go-')
l3, = ax1.plot(times, greedy_down, 'b+-')
l4, = ax1.plot(times, greedy_up, 'y^-')
l5, = ax1.plot(times, greedy_compute, 'm+-.')
ax1.set_xlabel('时间窗口(小时)', fontproperties=zhfont)
ax1.set_ylabel('映射代价', fontproperties=zhfont)
ax1.set_title('切片映射代价', fontproperties=zhfont)
plt.legend(handles=[l1, l2, l3, l4, l5], labels=['gabm', 'greedy_all', 'greedy_down', 'greedy_up', 'greedy_compute'],
           loc='best', prop=enfont)

# # 子图
# left, bottom, width, height = 0.2, 0.6, 0.25, 0.25
# ax2 = fig.add_axes([left, bottom, width, height])
# n = int(n / 4)
# gabm = gabm[0:n]
# tcgm = tcgm[0:n]
# bcgm = bcgm[0:n]
# times = np.linspace(0, n - 1, n)
# l11, = ax2.plot(times, gabm, 'rx-')
# l22, = ax2.plot(times, tcgm, 'go-')
# l33, = ax2.plot(times, bcgm, 'b+-')
# ax2.set_xlabel('时间窗口(秒)')
# ax2.set_ylabel('映射代价')
# ax2.set_title('切片映射代价')

plt.show()

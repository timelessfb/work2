#!usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 初始化figure
fig = plt.figure()

# 创建数据
gabm = [1.00823439, 0.9610158, 1.25045668, 1.33958262, 1.45751599, 1.58363811,
        1.41388021, 1.66687035, 2.08751612, 2.3921823, 2.44190869, 2.61424426,
        2.9362249, 4.14652182, 6.2225874, 8.54061499, 0., 2.94779991,
        1.00660737, 0.]
gabm = np.cumsum(gabm)
np.log
gabm = np.log(gabm)
greedy_all = [1.00924825, 0.960878, 1.25177135, 1.34064616, 1.45564969, 1.58680046,
              1.41479477, 1.66471074, 2.08561519, 2.39472523, 2.44826214, 2.61199104,
              2.96666356, 4.15181798, 6.22447257, 8.97152326, 5.7140688, 2.68936312,
              1.47309538, 1.22828263]
greedy_all = np.cumsum(greedy_all)
np.log()
gabm = np.log(greedy_all)
greedy_down = [1.01947679, 0.96864916, 1.26324037, 1.32930063, 1.46912797, 1.61600157,
               1.44704322, 1.69412202, 2.09294877, 2.43039641, 2.71076908, 2.78567561,
               3.66903996, 5.120238, 7.58749897, 9.16366928, 2.47805071, 0.9595719,
               2.04171308, 0.]
greedy_down = np.cumsum(greedy_down)

greedy_up = [1.01517191, 0.96144468, 1.27303092, 1.36123645, 1.44895328, 1.63003436,
             1.45422979, 1.69798793, 2.11161964, 2.71668521, 2.91063359, 3.4796993,
             3.54461101, 4.4054204, 6.67298857, 9.18515957, 5.15757623, 1.01117428,
             0., 0.79395723]
greedy_up = np.cumsum(greedy_up)

greedy_compute = [1.01450804, 0.96313182, 1.25476558, 1.3675296, 1.49832976, 1.5969548,
                  1.44596227, 1.79687217, 2.16947789, 2.42785674, 2.51086613, 2.73295013,
                  3.50331329, 5.13448691, 6.90807942, 10.26296645, 2.17866555, 2.8081006,
                  1.70977004, 0.5575955]
greedy_compute = np.cumsum(greedy_compute)

n = len(gabm)
times = np.linspace(0, n - 1, n)
print(times)
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax1 = fig.add_axes([left, bottom, width, height])
l1, = ax1.plot(times, gabm, 'rx-')
l2, = ax1.plot(times, greedy_all, 'go-')
l3, = ax1.plot(times, greedy_down, 'b+-')
l4, = ax1.plot(times, greedy_up, 'y^-')
l5, = ax1.plot(times, greedy_compute, 'm+-.')
ax1.set_xlabel('时间窗口(小时)')
ax1.set_ylabel('映射代价')
ax1.set_title('切片映射代价')
plt.legend(handles=[l1, l2, l3, l4, l5], labels=['gabm', 'greedy_all', 'greedy_down', 'greedy_up', 'greedy_compute'],
           loc='best')

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

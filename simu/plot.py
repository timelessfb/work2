#!usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 初始化figure
fig = plt.figure()

# 创建数据
gabm = [2.07204304, 2.33653884, 2.69998267, 2.83530052, 3.39271451, 4.58455437, 6.05599077, 8.78535706, 5.75447412,
        4.92403443, 0., 0., 0., 0.82795699, 0., 0., 0.]
gabm = np.cumsum(gabm)
tcgm = [2.07255885, 2.33292944, 2.7009231, 2.84209435, 3.39643122, 4.57393984, 6.03099213, 8.93168132, 14.10671229,
        5.89874407, 2.05978259, 0., 0., 0., 0., 0., 0.]
tcgm = np.cumsum(tcgm)
bcgm = [2.08690724, 2.34195704, 2.72669597, 2.82575188, 3.45161083, 4.69153424, 6.36793109, 9.63165266, 12.79928909,
        6.70787931, 2.99999998, 0., 0., 0., 0., 0., 0.]
bcgm = np.cumsum(bcgm)

n = len(gabm)
times = np.linspace(0, n - 1, n)
print(times)
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax1 = fig.add_axes([left, bottom, width, height])
l1, = ax1.plot(times, gabm, 'rx-')
l2, = ax1.plot(times, tcgm, 'go-')
l3, = ax1.plot(times, bcgm, 'b+-')
ax1.set_xlabel('时间窗口(秒)')
ax1.set_ylabel('映射代价')
ax1.set_title('切片映射代价')
plt.legend(handles=[l1, l2, l3], labels=['gabm', 'tcgm', 'bcgm'], loc='best')

# 子图
left, bottom, width, height = 0.2, 0.6, 0.25, 0.25
ax2 = fig.add_axes([left, bottom, width, height])
n = int(n / 4)
gabm = gabm[0:n]
tcgm = tcgm[0:n]
bcgm = bcgm[0:n]
times = np.linspace(0, n - 1, n)
l11, = ax2.plot(times, gabm, 'rx-')
l22, = ax2.plot(times, tcgm, 'go-')
l33, = ax2.plot(times, bcgm, 'b+-')
ax2.set_xlabel('时间窗口(秒)')
ax2.set_ylabel('映射代价')
ax2.set_title('切片映射代价')

plt.show()

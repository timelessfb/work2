#!usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
import matplotlib.font_manager as fm

zhfont = fm.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
enfont = fm.FontProperties(fname='C:\Windows\Fonts\\times.ttf')


# # 初始化figure
# fig = plt.figure()
# l = 100
# # 创建数据
# gabm = [4.22376253, 5.3703148, 5.17386761, 5.25521634, 7.67417229, 7.05459317,
#         7.59404593, 7.73766855, 7.44188555, 9.43715357, 10.0959776, 10.41188574,
#         9.75291084, 11.22828324, 8.48545377, 11.90691788, 12.07710097, 11.44302175,
#         13.09057788, 10.32758472, 11.26889859, 8.97576866, 5.93877196, 5.86480821,
#         6.20759639]
# gabm_success_count = [0., 0., 0., 0., 0., 0., 3., 0., 0., 0., 0., 0., 0., 1., 3., 1., 1., 2., 0., 3., 3., 3., 6., 6.,
#                       6.]
# gabm_success_count = [10 - i for i in gabm_success_count]
# gabm = np.cumsum(gabm)
# gabm_success_count = np.cumsum(gabm_success_count)
# gabm = np.true_divide(np.array(gabm), np.array(gabm_success_count))
#
# greedy_all = [7.73081479, 7.24647715, 7.2602596, 6.3585619, 8.98771822, 8.70778852,
#               6.4408716, 9.85000021, 8.95887443, 9.48514484, 10.12162563, 11.54006665,
#               10.02009414, 11.11496997, 9.10086591, 8.76318381, 10.66266621, 11.44302175,
#               10.63395618, 9.50395515, 7.13731883, 9.31039455, 5.93877196, 5.91798793,
#               6.20759639]
# greedy_all_success_count = [0., 0., 0., 0., 1., 0., 4., 0., 0., 1., 1., 0., 1., 1., 3., 3., 2., 2., 2., 4., 5., 3., 6.,
#                             6., 6.]
# greedy_all_success_count = [10 - i for i in greedy_all_success_count]
#
# greedy_all = np.cumsum(greedy_all)
# greedy_all_success_count = np.cumsum(greedy_all_success_count)
#
# greedy_all = np.true_divide(np.array(greedy_all), np.array(greedy_all_success_count))
# greedy_all = list(greedy_all)
# print(greedy_all)
# print(greedy_all_success_count)
# # greedy_all = l * np.log(greedy_all)
# # greedy_all = l * np.exp(greedy_all)
# # greedy_all = greedy_all - 3
#
# greedy_down = [6.37044435, 7.41947583, 6.9196751, 6.35710915, 8.84859073, 8.87280419,
#                7.59404593, 9.36669104, 8.47354879, 9.12560179, 10.27969452, 11.72913369,
#                11.30225574, 10.44002316, 9.53014523, 11.90691788, 10.35916701, 9.17989073,
#                12.31465346, 9.50395515, 9.18128111, 9.53510995, 4.35682481, 6.00299295,
#                6.22275869]
# greedy_down_success_count = [0., 0., 0., 0., 0., 0., 3., 0., 0., 1., 1., 0., 0., 2., 3., 1., 2., 3., 1., 4., 4., 3., 7.,
#                              6., 6.]
# greedy_down_success_count = [10 - i for i in greedy_down_success_count]
# greedy_down = np.cumsum(greedy_down)
# greedy_down_success_count = np.cumsum(greedy_down_success_count)
#
# greedy_down = np.true_divide(np.array(greedy_down), np.array(greedy_down_success_count))
# greedy_down = list(greedy_down)
# # List3 = map(lambda (a,b):a*b,zip(List1,List2))
# # greedy_down = l * np.log(greedy_down)
# # greedy_down = l * np.exp(greedy_down)
# # greedy_down = greedy_down - 3
#
# greedy_up = [6.6839089, 6.83677598, 8.14911262, 5.90597554, 9.29674973, 8.83350333,
#              7.59404593, 9.36299775, 9.133511, 8.42993972, 11.32834087, 9.64735334,
#              11.93181005, 8.25940072, 8.48545377, 7.77673887, 10.57523083, 9.24504891,
#              13.22034517, 10.67078358, 9.30696727, 10.09503361, 4.35682481, 5.89885526,
#              6.20759639]
#
# greedy_up_success_count = [0., 0., 0., 0., 1., 0., 3., 0., 0., 1., 1., 1., 0., 3., 3., 4., 2., 3., 0., 3., 4., 3., 7.,
#                            6., 6.]
# greedy_up_success_count = [10 - i for i in greedy_up_success_count]
#
# greedy_up = np.cumsum(greedy_up)
# greedy_up_success_count = np.cumsum(greedy_up_success_count)
#
# greedy_up = np.true_divide(np.array(greedy_up), np.array(greedy_up_success_count))
# greedy_up = list(greedy_up)
# # greedy_up = l * np.log(greedy_up)
# # greedy_up = l * np.exp(greedy_up)
# # greedy_up = greedy_up - 3
#
# greedy_compute = [7.21683231, 6.02839214, 6.76649296, 7.1990146, 8.88821622, 8.86435679,
#                   6.42072094, 9.99034523, 10.21886117, 7.41814499, 9.62672711, 10.2251609,
#                   11.38077317, 9.48911471, 9.64141896, 8.86148815, 10.70678971, 11.44302175,
#                   13.32403076, 9.29693026, 7.6715707, 7.99566177, 5.93877196, 5.91798793,
#                   6.22275869]
# greedy_compute_success_count = [0., 0., 0., 0., 0., 1., 4., 0., 0., 3., 2., 1., 0., 2., 3., 3., 2., 2., 0., 4., 5., 4.,
#                                 6., 6., 6.]
# greedy_compute_success_count = [10 - i for i in greedy_compute_success_count]
# greedy_compute = np.cumsum(greedy_compute)
# greedy_compute_success_count = np.cumsum(greedy_compute_success_count)
#
# greedy_compute = np.true_divide(np.array(greedy_compute), np.array(greedy_compute_success_count))
# greedy_compute = list(greedy_compute)
# # greedy_compute = l * np.log(greedy_compute)
# # greedy_compute = l * np.exp(greedy_compute)
# # greedy_compute = greedy_compute - 3
#
# n = len(gabm)
# times = np.linspace(0, n - 1, n)
# times = [10 * (i + 1) for i in times]
# left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
# ax1 = fig.add_axes([left, bottom, width, height])
# l1, = ax1.plot(times, gabm, 'rx-')
# l2, = ax1.plot(times, greedy_all, 'go-')
# l3, = ax1.plot(times, greedy_down, 'b+-')
# l4, = ax1.plot(times, greedy_up, 'y^-')
# l5, = ax1.plot(times, greedy_compute, 'm+-.')
# ax1.set_xlabel('切片请求数量（个）', fontproperties=zhfont)
# ax1.set_ylabel('单位切片映射代价', fontproperties=zhfont)
# ax1.set_title('累计切片映射代价', fontproperties=zhfont)
# plt.legend(handles=[l1, l2, l3, l4, l5], labels=['gabm', 'random', 'greedy_down', 'greedy_up', 'greedy_compute'],
#            loc='best', prop=enfont)
# plt.show()


# def plot_fun(cost, fails, req_num_eachtime, xlabel, ylabel, title):
#     fig = plt.figure()
#     y1 = np.cumsum(cost[0, :])
#     y2 = np.cumsum(cost[2, :])
#     y3 = np.cumsum(cost[4, :])
#     y4 = np.cumsum(cost[6, :])
#     y5 = np.cumsum(cost[6, :])
#     e1 = np.cumsum(req_num_eachtime - fails[0, :])
#     e2 = np.cumsum(req_num_eachtime - fails[2, :])
#     e3 = np.cumsum(req_num_eachtime - fails[3, :])
#     e4 = np.cumsum(req_num_eachtime - fails[4, :])
#     e5 = np.cumsum(req_num_eachtime - fails[6, :])
#
#     y1 = y1 / e1
#     y2 = y2 / e2
#     y3 = y3 / e3
#     y4 = y4 / e4
#     y5 = y5 / e5
#
#     n = len(y1)
#     x = np.linspace(0, n - 1, n)
#     x = [req_num_eachtime * (i + 1) for i in x]
#     left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
#     ax1 = fig.add_axes([left, bottom, width, height])
#
#     l1, = ax1.plot(x, y1, 'rx-')
#     l2, = ax1.plot(x, y2, 'go-')
#     l3, = ax1.plot(x, y3, 'b+-')
#     l4, = ax1.plot(x, y4, 'y^-')
#     l5, = ax1.plot(x, y5, 'm+-.')
#     ax1.set_xlabel(xlabel, fontproperties=zhfont)
#     ax1.set_ylabel(ylabel, fontproperties=zhfont)
#     ax1.set_title(title, fontproperties=zhfont)
#     plt.legend(handles=[l1, l2, l3, l4, l5], labels=['gabm', 'greedy_down', 'greedy_up', 'greedy_compute', 'random'],
#                loc='best', prop=enfont)
#     plt.savefig(title)


def plot_fun(cost, req_num_eachtime, xlabel, ylabel, title):
    fig = plt.figure()
    y1 = np.cumsum(cost[0, :])
    y2 = np.cumsum(cost[2, :])
    y3 = np.cumsum(cost[4, :])
    y4 = np.cumsum(cost[6, :])

    n = len(y1)
    x = np.linspace(0, n - 1, n)
    x = [req_num_eachtime * (i + 1) for i in x]
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax1 = fig.add_axes([left, bottom, width, height])

    x = np.r_[0, x]
    y1 = np.r_[0, y1]
    y2 = np.r_[0, y2]
    y3 = np.r_[0, y3]
    y4 = np.r_[0, y4]

    l1, = ax1.plot(x, y1, color='y', marker='d', markerfacecolor='w', linestyle=':')
    l2, = ax1.plot(x, y2, color='g', marker='x', markerfacecolor='w', linestyle='-.')
    l3, = ax1.plot(x, y3, color='b', marker='^', markerfacecolor='w', linestyle='-.')
    l4, = ax1.plot(x, y4, color='c', marker='+', markerfacecolor='w', linestyle='-.')

    ax1.set_xlabel(xlabel, fontproperties=zhfont)
    ax1.set_ylabel(ylabel, fontproperties=zhfont)
    # ax1.set_title(title, fontproperties=zhfont)
    plt.legend(handles=[l1, l2, l3, l4], labels=['GASM', 'MBCSM', 'MCCSM', 'RSM'], loc='best', prop=enfont)
    plt.savefig(title)


def plot_fun_slot(cost, fails, req_num_deta_eachtime, xlabel, ylabel, title):
    fig = plt.figure()
    n = np.size(cost[0, :])
    success = np.zeros((n), dtype=np.int)
    for i in range(n):
        success[i] = req_num_deta_eachtime * (i + 1)

    y1 = cost[0, :]
    y2 = cost[2, :]
    y3 = cost[4, :]
    y4 = cost[6, :]
    # y5 = cost[6, :]
    e1 = success - fails[0, :]
    e2 = success - fails[2, :]
    e3 = success - fails[4, :]
    e4 = success - fails[6, :]
    # e5 = success - fails[6, :]

    y1 = y1 / e1
    y2 = y2 / e2
    y3 = y3 / e3
    y4 = y4 / e4
    # y5 = y5 / e5
    x = success

    x = np.r_[0, x]
    y1 = np.r_[0, y1]
    y2 = np.r_[0, y2]
    y3 = np.r_[0, y3]
    y4 = np.r_[0, y4]
    # y5 = np.r_[0, y5]

    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax1 = fig.add_axes([left, bottom, width, height])

    l1, = ax1.plot(x, y1, color='y', marker='d', markerfacecolor='w', linestyle=':')
    l2, = ax1.plot(x, y2, color='g', marker='x', markerfacecolor='w', linestyle='-.')
    l3, = ax1.plot(x, y3, color='b', marker='^', markerfacecolor='w', linestyle='-.')
    l4, = ax1.plot(x, y4, color='c', marker='+', markerfacecolor='w', linestyle='-.')
    # l5, = ax1.plot(x, y5, color='y', marker='o', markerfacecolor='w', linestyle='--')
    ax1.set_xlabel(xlabel, fontproperties=zhfont)
    if ylabel == 1:
        ylabel = r'$\mathit{\overline{\phi}_p}$'

    if ylabel == 2:
        ylabel = r'$\mathit{\overline{\phi}_b}$'

    if ylabel == 3:
        ylabel = r'$\mathit{\overline{\phi}_{total}}$'

    ax1.set_ylabel(ylabel, fontproperties=zhfont)
    # ax1.set_xticks(x)
    title += 'slot'
    # ax1.set_title(title, fontproperties=zhfont)
    plt.legend(handles=[l1, l2, l3, l4], labels=['GASM', 'MBCSM', 'MCCSM', 'RSM'], loc='best', prop=enfont)
    plt.savefig(title)


def plot_fun_fail_slot(fails, req_num_deta_eachtime, xlabel, ylabel, title):
    fig = plt.figure()
    n = np.size(fails[0, :])
    success = np.zeros((n), dtype=np.int)
    for i in range(n):
        success[i] = req_num_deta_eachtime * (i + 1)

    y1 = fails[0, :]
    y2 = fails[2, :]
    y3 = fails[4, :]
    y4 = fails[6, :]
    # y5 = fails[6, :]

    y1 = 100 * (y1 / success)
    y2 = 100 * (y2 / success)
    y3 = 100 * (y3 / success)
    y4 = 100 * (y4 / success)
    # y5 = 100 * (y5 / success)

    x = success

    x = np.r_[0, x]
    y1 = np.r_[0, y1]
    y2 = np.r_[0, y2]
    y3 = np.r_[0, y3]
    y4 = np.r_[0, y4]
    # y5 = np.r_[0, y5]

    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax1 = fig.add_axes([left, bottom, width, height])

    l1, = ax1.plot(x, y1, color='y', marker='d', markerfacecolor='w', linestyle=':')
    l2, = ax1.plot(x, y2, color='g', marker='x', markerfacecolor='w', linestyle='-.')
    l3, = ax1.plot(x, y3, color='b', marker='^', markerfacecolor='w', linestyle='-.')
    l4, = ax1.plot(x, y4, color='c', marker='+', markerfacecolor='w', linestyle='-.')
    # l5, = ax1.plot(x, y5, color='y', marker='o', markerfacecolor='w', linestyle='--')

    ax1.set_xlabel(xlabel, fontproperties=zhfont)
    ylabel = r'$\mathit{{\Psi}(\%)}$'

    ax1.set_ylabel(ylabel, fontproperties=zhfont)
    title += 'slot'
    # ax1.set_title(title, fontproperties=zhfont)
    plt.legend(handles=[l1, l2, l3, l4], labels=['GASM', 'MBCSM', 'MCCSM', 'RSM'], loc='best', prop=enfont)
    plt.savefig(title)

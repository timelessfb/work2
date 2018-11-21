import math
import sys

import numpy as np
import random
import simu.GA as ga


def check(s, rbsc, bs):
    if s[0] <= rbsc[bs][0] and s[1] <= rbsc[bs][1] and s[2] <= rbsc[bs][2]:
        return True
    else:
        return False


def random_select(sr, rbsc, delta):
    m = np.size(sr, 0)
    n = np.size(rbsc, 0)
    rbsc_realtime = np.array(rbsc)
    cost_all = 0
    solution = np.zeros((m, n), dtype=np.float)
    for i in range(m):
        s = sr[i]
        min_cost = sys.maxsize
        min_index = -1
        bss = np.arange(n)
        random.shuffle(bss)
        flag = 0
        for bs in bss:
            if check(s, rbsc_realtime, bs):
                flag = 1
                break
        if flag == 1:
            cost_all += s[0] / (rbsc_realtime[bs][0] + delta) + s[1] / (rbsc_realtime[bs][1] + delta) + s[2] / (
                    rbsc_realtime[bs][2] + delta)
            rbsc_realtime[bs][0] -= s[0]
            rbsc_realtime[bs][1] -= s[1]
            rbsc_realtime[bs][2] -= s[2]
            solution[i][bs] = 1
    return cost_all, rbsc_realtime, solution

# def random_select(sr, rbsc, delta):
#     m = np.size(sr, 0)
#     n = np.size(rbsc, 0)
#     rbsc_realtime = np.array(rbsc)
#     cost_all = 0
#     solution = np.zeros((m, n), dtype=np.float)
#     for i in range(m):
#         s = sr[i]
#         min_cost = sys.maxsize
#         min_index = -1
#         bss = np.arange(n)
#         # random.shuffle(bss) #############与第一个的不同点
#         flag = 0
#         for bs in bss:
#             if check(s, rbsc_realtime, bs):
#                 flag = 1
#                 break
#         if flag == 1:
#             cost_all += s[0] / (rbsc_realtime[bs][0] + delta) + s[1] / (rbsc_realtime[bs][1] + delta) + s[2] / (
#                     rbsc_realtime[bs][2] + delta)
#             rbsc_realtime[bs][0] -= s[0]
#             rbsc_realtime[bs][1] -= s[1]
#             rbsc_realtime[bs][2] -= s[2]
#             solution[i][bs] = 1
#     return cost_all, rbsc_realtime, solution


if __name__ == '__main__':
    print(sys.maxsize)

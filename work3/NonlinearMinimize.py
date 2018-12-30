#!usr/bin/env python
# -*- coding:utf-8 -*-
# import numpy as np
import numpy as np
from scipy import stats


def slice(lowbound=1, upbound=10):
    '''
    :param lowbound: 比例的下界
    :param upbound: 比例的上界
    :return:
    '''
    rho1, rho2, rho3 = np.random.uniform(lowbound, upbound, 3)
    sum = rho1 + rho2 + rho3
    # 归一化
    rho1 /= sum
    rho2 /= sum
    rho3 /= sum
    return rho1, rho2, rho3


def BSs(J, down=10, up=10, compute=10):
    bss = np.zeros((J, 3))
    for i in range(J):
        bss[i][0] = down
        bss[i][1] = up
        bss[i][2] = compute


def slices(S=18, lowbound=1, upbound=10):
    scs = np.zeros((S, 3))  # rho1, rho2, rho3
    for i in range(S):
        scs[i][0], scs[i][1], scs[i][2] = slice(lowbound, upbound)
    return scs


def ZtoX(i, X_map, S, J_num):
    if i == -1:
        return -1, -1
    t = -1
    for l_x in range(S):
        for l_y in range(J_num + 1):
            if X_map[l_x][l_y] == 0:
                t += 1
                if i == t:
                    return l_x, l_y


def XtoZ(l_x, l_y, X_map, S, J_num):
    if X_map[l_x][l_y] != 0:
        return -1
    t = -1
    if l_x == 0:
        for j in range(l_y + 1):
            if X_map[0][j] == 0:
                t += 1
        return t
    for i in range(l_x):
        for j in range(J_num + 1):
            if X_map[i][j] == 0:
                t += 1
    for j in range(l_y + 1):
        if X_map[l_x][j] == 0:
            t += 1
    return t


def ResourceConstraint(resorce_type, z, j, X_map, I, ROH, S, J_num, load):
    x_indexs = []
    y_indexs = []
    y_selected_indexs = []

    s_in_j = []
    for s in range(S):
        if X_map[s][j] == 0:
            x_indexs.append(XtoZ(s, j, X_map, S, J_num))
            y_indexs.append(XtoZ(s, J_num, X_map, S, J_num))
            s_in_j.append(s)
    for s in range(S):
        if X_map[s][j] == 1:
            y_selected_indexs.append(XtoZ(s, J_num, X_map, S, J_num))

    if resorce_type == 'down':
        o = load[j][0]
        for i in range(len(x_indexs)):
            o -= z[x_indexs[i]] * z[y_indexs[i]] * ROH[s][0]
        for i in range(len(y_selected_indexs)):
            o -= z[y_selected_indexs[i]] * ROH[s][0]

    if resorce_type == 'up':
        o = load[j][1]
        for i in range(len(x_indexs)):
            o -= z[x_indexs[i]] * z[y_indexs[i]] * ROH[s][1]
        for i in range(len(y_selected_indexs)):
            o -= z[y_selected_indexs[i]] * ROH[s][1]

    if resorce_type == 'compute':
        o = load[j][2]
        for i in range(len(x_indexs)):
            o -= z[x_indexs[i]] * z[y_indexs[i]] * ROH[s][2]
        for i in range(len(y_selected_indexs)):
            o -= z[y_selected_indexs[i]] * ROH[s][2]
    return o


def EqConstraint(z, s, X_map, I, ROH, S, J_num, load):
    o = 0
    for j in range(J_num):
        if X_map[s][j] == 0:
            o -= z[XtoZ(s, j, X_map, S, J_num)]
    return o


def foo(X_map, I, ROH, S, J_num, load):
    cons = []
    for j in range(J_num):
        cons.append({'type': 'ineq', 'fun': lambda z: ResourceConstraint('down', z, j, X_map, I, ROH, S, J_num, load)})
        cons.append({'type': 'ineq', 'fun': lambda z: ResourceConstraint('up', z, j, X_map, I, ROH, S, J_num, load)})
        cons.append(
            {'type': 'ineq', 'fun': lambda z: ResourceConstraint('compute', z, j, X_map, I, ROH, S, J_num, load)})
    for s in range(S):
        if np.max(X_map[s]) == 1:  # 已经选定了基站
            continue
        cons.append(
            {'type': 'eq', 'fun': lambda z: EqConstraint(z, s, X_map, I, ROH, S, J_num, load)})


if __name__ == '__main__':
    # 初始参数
    S = 18
    J_num = 6
    # 初始位置
    I = np.zeros(S, dtype=int)
    # 暂时初始化,后边需要改
    for i in range(S):
        I[i] = i % J_num

    X_map = np.random.binomial(1, 0.5, [S, J_num])
    X_map -= 1
    X_map = np.c_[X_map, np.zeros(S)]

    load = np.zeros((J_num, 3))  # 第一列是每个基站的down资源，第二列up资源，第三列compute资源
    load += 6

    # RHO=np.zeros((S,3))
    for i in range(S):
        RHO = slices(S)
    print(RHO)

    # for iter in range(10):

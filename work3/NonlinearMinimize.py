#!usr/bin/env python
# -*- coding:utf-8 -*-
# import numpy as np
import numpy as np
from scipy.optimize import minimize


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


def ValCount(X_map, S, J_num):
    o = 0
    for s in range(S):
        for j in range(J_num + 1):
            if X_map[s][j] == 0:
                o += 1


def UnSelectBS(X_map, S, J_num):
    o = []
    for s in range(S):
        if np.max(X_map[s][0:J_num]) == 1:  # todo(*风险点)
            o.append(s)
    return o


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


def cost(type, z, X_map, I, ROH, S, J_num, load, T):
    alpha = 1
    beta = 1

    o1 = 0
    for s in range(S):
        o1 += (1 - z[XtoZ(s, J_num, X_map, S, J_num)])
    o1 *= alpha * T

    o2 = 0
    for s in range(S):
        i = I[s]
        for j in range(J_num):
            if X_map[s][j] == 0:
                if j != i:
                    o2 += z[XtoZ(s, j, X_map, S, J_num)]
    o2 *= beta

    if type == 0:
        return o1 + o2
    if type == 1:
        return o1
    if type == 2:
        return o2


def opt(X_map, I, ROH, S, J_num, load, T):
    # 设置界
    bnd = (0, 1)
    bnds = []
    for s in range(S):
        for j in range(J_num + 1):
            if X_map[s][j] == 0:
                bnds.append(bnd)

    # 设置约束
    cons = []
    for j in range(J_num):
        cons.append({'type': 'ineq', 'fun': lambda z: ResourceConstraint('down', z, j, X_map, I, ROH, S, J_num, load)})
        cons.append({'type': 'ineq', 'fun': lambda z: ResourceConstraint('up', z, j, X_map, I, ROH, S, J_num, load)})
        cons.append(
            {'type': 'ineq', 'fun': lambda z: ResourceConstraint('compute', z, j, X_map, I, ROH, S, J_num, load)})
    for s in range(S):
        if np.max(X_map[s][0:J_num]) == 1:  # 已经选定了基站 todo(*风险点)
            continue
        cons.append(
            {'type': 'eq', 'fun': lambda z: EqConstraint(z, s, X_map, I, ROH, S, J_num, load)})

    # 设置目标
    objective = lambda z: cost(0, z, X_map, I, ROH, S, J_num, load, T)

    # 设置初始值z0
    z0 = np.zeros(ValCount(X_map, S, J_num))
    for s in range(S):
        if XtoZ(s, I[s], X_map, S, J_num) != -1:
            z0[XtoZ(s, I[s], X_map, S, J_num)]
    # todo(*做初始资源分配)

    solution = minimize(objective, z0, method='SLSQP', bounds=bnds, constraints=cons)
    # todo(*查一下这句有没有问题)
    z = solution.x
    return z, cost(0, z, X_map, I, ROH, S, J_num, load, T)


def solve(X_map, I, ROH, S, J_num, load, T):
    for s in range(S):
        z, cost = opt(X_map, I, ROH, S, J_num, load, T)
        # todo(*可以优化)
        max_z = -1
        max_z_index = -1
        for i in range(np.size(z)):
            if z[i] > max_z:
                l_x, l_y = ZtoX(i, X_map, S, J_num)
                if l_y != J_num:
                    max_z_index = i
        l_x, l_y = ZtoX(max_z_index, X_map, S, J_num)
        for j in range(J_num):
            X_map[l_x][j] = -1
        X_map[l_x][l_y] = 1
    # 确定为所有的基站，求解Js
    z, cost = opt(X_map, I, ROH, S, J_num, load, T)
    for s in range(S):
        X_map[s][J_num] = z[i]
    return X_map


if __name__ == '__main__':
    # 初始参数
    S = 18
    J_num = 6
    # 初始位置
    I = np.zeros(S, dtype=int)
    # 暂时初始化,后边需要改
    # todo(*还没仔细处理)
    for i in range(S):
        I[i] = i % J_num

    X_map = np.random.binomial(1, 0.5, [S, J_num])
    X_map -= 1
    X_map = np.c_[X_map, np.zeros(S)]  # X_map中0就是变量,1代表s映射到j或者ys=1,-1代表不可选基站

    load = np.zeros((J_num, 3))  # 第一列是每个基站的down资源，第二列up资源，第三列compute资源
    load += 6

    # RHO=np.zeros((S,3))
    for i in range(S):
        RHO = slices(S)
    print(RHO)

    # for iter in range(10):

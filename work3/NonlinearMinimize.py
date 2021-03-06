#!usr/bin/env python
# -*- coding:utf-8 -*-
# import numpy as np
import numpy as np
from scipy.optimize import minimize, linprog


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
    return o


def UnSelectBS(X_map, S, J_num):
    o = []
    for s in range(S):
        if np.max(X_map[s][0:J_num]) == 1:  # todo(*风险点)
            o.append(s)
    return o


def ResourceConstraint(resorce_type, z, j, X_map, I, ROH, S, J_num, load):
    # X_map中第j列为0的位置对应z中的index
    x_indexs = []
    # X_map中第j列为0的位置，该行为s切片，找到s行行模ys变量对应z中的index
    y_indexs = []
    # 保存找到确定映射到基站j的切片，即X_map中第j列为1的位置
    y_selected_indexs = []

    # 记录在j上部署切片s
    s_in_j_partly = []
    s_in_j_all = []
    for s in range(S):
        if X_map[s][j] == 0:
            x_indexs.append(XtoZ(s, j, X_map, S, J_num))
            y_indexs.append(XtoZ(s, J_num, X_map, S, J_num))
            s_in_j_partly.append(s)
    for s in range(S):
        if X_map[s][j] == 1:
            y_selected_indexs.append(XtoZ(s, J_num, X_map, S, J_num))
            s_in_j_all.append(s)
    if resorce_type == 'down':
        o = load[j][0]
        for i in range(len(x_indexs)):
            s = s_in_j_partly[i]
            o -= z[x_indexs[i]] * z[y_indexs[i]] * ROH[s][0]
        for i in range(len(y_selected_indexs)):
            s = s_in_j_all[i]
            o -= z[y_selected_indexs[i]] * ROH[s][0]

    if resorce_type == 'up':
        o = load[j][1]
        for i in range(len(x_indexs)):
            s = s_in_j_partly[i]
            o -= z[x_indexs[i]] * z[y_indexs[i]] * ROH[s][1]
        for i in range(len(y_selected_indexs)):
            s = s_in_j_all[i]
            o -= z[y_selected_indexs[i]] * ROH[s][1]

    if resorce_type == 'compute':
        o = load[j][2]
        for i in range(len(x_indexs)):
            s = s_in_j_partly[i]
            o -= z[x_indexs[i]] * z[y_indexs[i]] * ROH[s][2]
        for i in range(len(y_selected_indexs)):
            s = s_in_j_all[i]
            o -= z[y_selected_indexs[i]] * ROH[s][2]
    return o


# 对于s,xij=1
def EqConstraint(z, s, X_map, I, ROH, S, J_num, load):
    o = 1
    for j in range(J_num):
        if X_map[s][j] == 0:
            o -= z[XtoZ(s, j, X_map, S, J_num)]
    return o


def cost(type, z, X_map, I, ROH, S, J_num, load, alpha, beta):
    o1 = 0
    for s in range(S):
        o1 += (1 - z[XtoZ(s, J_num, X_map, S, J_num)])
    o1 *= alpha

    o2 = 0
    for s in range(S):
        i = I[s]
        if i == -1:  # 说明该切片是第一次映射，不存在迁移成本
            continue
        for j in range(J_num):
            if X_map[s][j] != -1:  # 找到s切片映射的基站j，注意X_map[s][j]对应的xij可能为小数，表示部分映射
                if j != i:  # 找到发生迁移的部分切片（因为xij可能为小数）
                    o2 += z[XtoZ(s, j, X_map, S, J_num)]
    o2 *= beta

    if type == 0:
        return o1 + o2
    if type == 1:
        return o1
    if type == 2:
        return o2


def num_of_migration(X_map, I):
    o = 0
    for s in range(S):
        i = I[s]
        if i == -1:  # 说明该切片是第一次映射，不存在迁移成本
            continue
        for j in range(J_num):
            if X_map[s][j] == 1:
                if j != i:
                    o += 1
    return o


def generate_k(S, multiple):
    k = np.random.uniform(1, 1000, S)  # todo(*可调参)
    sum = np.sum(k)
    k = multiple * k * S / sum
    return k


def generate_K(S, iter):
    K = np.zeros((iter, S))
    multiples = np.linspace(1, 3, iter)
    for i in range(iter):
        K[i] = generate_k(S, multiples[i])
    return K


def opt(X_map, I, RHO, S, J_num, load, alpha, beta, type):
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
        cons.append(
            {'type': 'ineq', 'fun': lambda z, j=j: ResourceConstraint('down', z, j, X_map, I, RHO, S, J_num, load)})
        cons.append(
            {'type': 'ineq', 'fun': lambda z, j=j: ResourceConstraint('up', z, j, X_map, I, RHO, S, J_num, load)})
        cons.append(
            {'type': 'ineq', 'fun': lambda z, j=j: ResourceConstraint('compute', z, j, X_map, I, RHO, S, J_num, load)})
    for s in range(S):
        if np.max(X_map[s][0:J_num]) == 1:  # 已经选定了基站 todo(*风险点)
            continue
        cons.append(
            {'type': 'eq', 'fun': lambda z, s=s: EqConstraint(z, s, X_map, I, RHO, S, J_num, load)})

    # 设置目标
    objective = lambda z: cost(type, z, X_map, I, RHO, S, J_num, load, alpha, beta)

    # 设置初始值z0
    z0 = np.zeros(ValCount(X_map, S, J_num))
    # 初始值为每个切片任意找一个基站作为初始基站，ys赋值为0，则一定是一个可行解
    X_map_for_z0 = np.copy(X_map)
    for s in range(S):
        if XtoZ(s, I[s], X_map, S, J_num) != -1:
            z0[XtoZ(s, I[s], X_map, S, J_num)] = 1
            for j in range(J_num):
                X_map_for_z0[s][j] = -1
            X_map_for_z0[s][I[s]] = 1
    # todo(*做初始资源分配,即ys暂时都定为0)
    ys = Simplex(X_map_for_z0, RHO, S, J_num, load)
    for s in range(S):
        z0[XtoZ(s, J_num, X_map, S, J_num)] = ys[s]

    solution = minimize(objective, z0, method='SLSQP', bounds=bnds, constraints=cons)
    # todo(*查一下这句有没有问题)
    z = solution.x
    return z, cost(0, z, X_map, I, RHO, S, J_num, load, alpha, beta), cost(1, z, X_map, I, RHO, S, J_num, load, alpha,
                                                                           beta), cost(2, z, X_map, I, RHO, S, J_num,
                                                                                       load, alpha, beta)


def Simplex(X_map, RHO, S, J_num, load):
    # 确定为所有的基站，求解ys，采用单纯形算法：min c'Y,  s.t. AY<=b, 0<=Y<=b
    A = np.zeros((J_num * 3, S))
    b = np.zeros(J_num * 3)
    c = np.zeros(S)
    c += -1
    for j in range(J_num):
        for s in range(S):
            if X_map[s][j] == 1:
                A[j * 3][s] = RHO[s][0]
                A[j * 3 + 1][s] = RHO[s][1]
                A[j * 3 + 2][s] = RHO[s][2]
    for j in range(J_num):
        b[j * 3] = load[j][0]
        b[j * 3 + 1] = load[j][1]
        b[j * 3 + 2] = load[j][2]
    bnd = (0, 1)
    bnds = []
    # 设置ys的bound
    for s in range(S):
        bnds.append(bnd)
    solution = linprog(c, A_ub=A, b_ub=b, bounds=bnds)
    ys = solution.x
    return ys


def solve(X_map, I, RHO, S, J_num, load, alpha, beta, type):
    J = np.zeros(S, dtype=int)  # 记录映射结果
    J -= 1
    X_map_init = np.copy(X_map)
    slice_has_select_bs = []
    for s in range(S):
        X_map_this_loop = np.copy(X_map)
        z, cost_all, cost_d, cost_m = opt(np.copy(X_map_this_loop), I, RHO, S, J_num, load, alpha, beta, type)
        # todo(*可以优化，比如设置大于一个阈值，就令xij=1)
        # 记录最大的xij，并令xij=1
        zz = z
        for s in range(S):
            zz[XtoZ(s, J_num, X_map, S, J_num)] = 0  # 将得到结果的ys置为0
        max_z = np.max(zz)
        max_z_index = np.where(zz == max_z)  # 可能存在多个同时最大的
        max_z_index = max_z_index[0]
        max_z_xii_index = []
        for z_index in max_z_index:
            l_x, l_y = ZtoX(z_index, np.copy(X_map_this_loop), S, J_num)
            if l_y == I[l_x]:
                max_z_xii_index.append(z_index)
        if len(max_z_xii_index) > 0:
            for z_index in max_z_xii_index:
                l_x, l_y = ZtoX(z_index, np.copy(X_map_this_loop), S, J_num)
                for j in range(J_num):
                    X_map[l_x][j] = -1
                X_map[l_x][l_y] = 1
                J[l_x] = l_y
        else:  # 最大的xij不是xii,则挑选出来最大的xij（多个，比如x01,x02都等于1，最大）对应的yj集合中的最大一个，设置为1
            max_ys = -1
            max_ys_index = -1
            for z_index in max_z_index:
                l_x, l_y = ZtoX(z_index, X_map, S, J_num)
                if z[XtoZ(s, J_num, X_map, S, J_num)] > max_ys:
                    max_ys = z[XtoZ(s, J_num, X_map, S, J_num)]
                    max_ys_index = z_index
            l_x, l_y = ZtoX(max_ys_index, X_map, S, J_num)
            for j in range(J_num):
                X_map[l_x][j] = -1
            X_map[l_x][l_y] = 1
            J[l_x] = l_y
        if ValCount(X_map, S, J_num) == S:
            break

    # 确定为所有的基站，求解ys，采用单纯形算法：min c'Y,  s.t. AY<=b, 0<=Y<=b
    ys = Simplex(np.copy(X_map), np.copy(RHO), S, J_num, load)
    for s in range(S):
        X_map[s][J_num] = ys[s]

    # 根据ys求解降级部分
    degradation = 0
    for s in range(S):
        degradation += (1 - ys[s])
    cost_d = degradation * alpha

    # 求解迁移部分
    num_migration = num_of_migration(X_map, I)
    cost_m = beta * num_migration

    # 求解两部分代价之和
    cost_all = cost_d + cost_m
    return X_map, J, ys, cost_all, cost_d, cost_m, degradation, num_migration


def solve1(X_map, I, RHO, S, J_num, load, alpha, beta, type):
    J = np.zeros(S, dtype=int)  # 记录映射结果
    J -= 1
    for s in range(S):
        X_map_this_loop = np.copy(X_map)
        z, cost_all, cost_d, cost_m = opt(np.copy(X_map_this_loop), I, RHO, S, J_num, load, alpha, beta, type)
        # todo(*可以优化，比如设置大于一个阈值，就令xij=1)
        # 记录最大的xij，并令xij=1
        # 记录最大的xij，并令xij=1
        zz = z
        for s in range(S):
            zz[XtoZ(s, J_num, X_map, S, J_num)] = 0  # 将得到结果的ys置为0
        max_z = np.max(zz)
        max_z_index = np.where(zz == max_z)  # 可能存在多个同时最大的
        max_z_index = max_z_index[0]
        for z_index in max_z_index:
            l_x, l_y = ZtoX(z_index, np.copy(X_map_this_loop), S, J_num)
            for j in range(J_num):
                X_map[l_x][j] = -1
            X_map[l_x][l_y] = 1
            J[l_x] = l_y
        if ValCount(X_map, S, J_num) == S:
            break

    # 确定为所有的基站，求解ys，采用单纯形算法：min c'Y,  s.t. AY<=b, 0<=Y<=b
    ys = Simplex(np.copy(X_map), np.copy(RHO), S, J_num, load)
    for s in range(S):
        X_map[s][J_num] = ys[s]

    # 根据ys求解降级部分
    degradation = 0
    for s in range(S):
        degradation += (1 - ys[s])
    cost_d = degradation * alpha

    # 求解迁移部分
    num_migration = num_of_migration(X_map, I)
    cost_m = beta * num_migration

    # 求解两部分代价之和
    cost_all = cost_d + cost_m
    return X_map, J, ys, cost_all, cost_d, cost_m, degradation, num_migration


# 方法1
def alg_optimize(S, J_num, X_map, load, RHO, I, ys, iter, K, mu):
    o = np.zeros((iter, 5))
    RHO_init = np.copy(RHO)
    for i in range(iter):
        print(i)
        RHO = RHO_init
        for s in range(S):
            RHO[s] *= K[i][s]
        # RHO = RHO_init * K[i]
        # todo(*计算降级函数上限，待验证1 / K[i])
        d = 0.0001  # 防止d=0的情况
        for s in range(S):
            if ys[s] * (1 / K[i][s]) < 1:
                d += 1 - ys[s] * (1 / K[i][s])
        alpha = mu / S
        # todo(*计算迁移上界，有些只能在一个基站上，多算了，算了S次)
        beta = (1 - mu) / S
        X_map_o, J, ys, cost_all, cost_d, cost_m, degradation, num_migration = solve(np.copy(X_map), I, RHO, S, J_num,
                                                                                     load, alpha, beta, 0)
        I = J  # 修改切片映射的基站
        o[i][0] = cost_d
        o[i][1] = cost_m
        o[i][2] = cost_all
        o[i][3] = degradation
        o[i][4] = num_migration
    return o


# 方法2
def alg_without_migration_cost(S, J_num, X_map, load, RHO, I, ys, iter, K, mu):
    o = np.zeros((iter, 5))
    RHO_init = np.copy(RHO)
    for i in range(iter):
        print(i)
        RHO = RHO_init
        for s in range(S):
            RHO[s] *= K[i][s]
        # RHO = RHO_init * K[i]
        # todo(*计算降级函数上限，待验证1 / K[i])
        d = 0.0001
        for s in range(S):
            if ys[s] * (1 / K[i][s]) < 1:
                d += 1 - ys[s] * (1 / K[i][s])
        alpha = mu / S
        # todo(*计算迁移上界，有些只能在一个基站上，多算了，算了S次)
        beta = (1 - mu) / S
        X_map_o, J, ys, cost_all, cost_d, cost_m, degradation, num_migration = solve1(np.copy(X_map), I, RHO, S, J_num,
                                                                                      load, alpha, beta, 1)
        I = J  # 修改切片映射的基站
        o[i][0] = cost_d
        o[i][1] = cost_m
        o[i][2] = cost_all
        o[i][3] = degradation
        o[i][4] = num_migration
    return o


# 方法3
def static_fix_prov(S, J_num, X_map, load, RHO, I, ys, iter, K, mu):
    o = np.zeros((iter, 5))
    for i in range(iter):
        print(i)
        # todo(*计算降级函数上限，待验证1 / K[i])
        d = 0.0001
        for s in range(S):
            if ys[s] * (1 / K[i][s]) < 1:
                d += 1 - ys[s] * (1 / K[i][s])
        alpha = mu / S
        # todo(*计算迁移上界，有些只能在一个基站上，多算了，算了S次)
        beta = (1 - mu) / S
        degradation = 0
        for s in range(S):
            if ys[s] * (1 / K[i][s]) < 1:
                degradation += 1 - ys[s] * (1 / K[i][s])
        cost_d = alpha * degradation
        num_migration = 0
        cost_m = beta * num_migration
        cost_all = cost_d + cost_m
        o[i][0] = cost_d
        o[i][1] = cost_m
        o[i][2] = cost_all
        o[i][3] = degradation
        o[i][4] = num_migration
    return o


# 方法4
def static_opt_prov(S, J_num, X_map, load, RHO, I, ys, iter, K, mu):
    o = np.zeros((iter, 5))
    RHO_init = np.copy(RHO)
    X_map_init = np.zeros_like(X_map)
    X_map_init -= 1
    for s in range(S):
        X_map_init[s][I[s]] = 1
    for i in range(iter):
        print(i)
        RHO = RHO_init
        for s in range(S):
            RHO[s] *= K[i][s]
        # todo(*计算降级函数上限，待验证1 / K[i])
        d = 0.0001
        for s in range(S):
            if ys[s] * (1 / K[i][s]) < 1:
                d += 1 - ys[s] * (1 / K[i][s])
        alpha = mu / S
        # todo(*计算迁移上界，有些只能在一个基站上，多算了，算了S次)
        beta = (1 - mu) / S
        ys = Simplex(np.copy(X_map_init), np.copy(RHO), S, J_num, load)
        # 根据ys求解降级部分
        degradation = 0
        for s in range(S):
            degradation += (1 - ys[s])
        cost_d = degradation * alpha

        # 求解迁移部分
        num_migration = 0
        cost_m = beta * num_migration

        # 求解两部分代价之和
        cost_all = cost_d + cost_m
        o[i][0] = cost_d
        o[i][1] = cost_m
        o[i][2] = cost_all
        o[i][3] = degradation
        o[i][4] = num_migration
    return o


if __name__ == '__main__':
    ############## 初始参数
    # 参数1:切片数量
    S = 18
    print("切片数量")
    print(S)
    # 参数2：基站数目
    J_num = 6
    print("MEC数量")
    print(J_num)
    # 参数3：可选基站集合
    X_map = np.random.binomial(1, 0.8, [S, J_num])
    candidate_bs_num = np.sum(X_map, 1)
    slices_of_candidate_bs_num_equalTo_0 = np.where(candidate_bs_num == 0)
    # 一个可选基站都没有的，随机为该切片选择一个
    for s in slices_of_candidate_bs_num_equalTo_0[0]:
        j = np.random.randint(0, J_num, 1)
        X_map[s][j] = 1
    X_map -= 1
    X_map = np.c_[X_map, np.zeros(S)]  # X_map中0就是变量,1代表s映射到j或者ys=1,-1代表不可选基站

    # 参数4：基站的资源
    load = np.zeros((J_num, 3))  # 第一列是每个基站的down资源，第二列up资源，第三列compute资源
    load += 3

    # 参数5：切片参数，C_req_s_down,C_req_s_up,C_req_s_compute,随机生成
    RHO = slices(S)

    # 参数6：权重因子 todo(*参数待调整)
    alpha = 1 / S  # 参数待调整
    beta = 1 / S  # 参数待调整

    # 参数7：初始位置,与初始ys
    I = np.zeros(S, dtype=int)
    I -= 1  # -1表示是第一次映射，无初始的映射基站
    X_map_o, J, ys, cost_all, cost_d, cost_m, degradation, num_migration = solve1(np.copy(X_map), I, RHO, S, J_num,
                                                                                  load, alpha,
                                                                                  beta,
                                                                                  0)  # 完成第一次映射过程 todo(*未传参alpha,beta)
    for s in range(S):
        I[s] = J[s]

    # 参数8：仿真图点数
    iter = 3  # todo(*参数可调)

    # 参数:9：生成切片调整因子K，RHO=K[i]*RHO
    K = generate_K(S, iter)
    print(K)
    mu = 0.5
    print("方法1:凸优化")
    o1 = alg_optimize(S, J_num, X_map, load, RHO, I, ys, iter, K, mu)
    print("方法2：静态")
    o2 = static_fix_prov(S, J_num, X_map, load, RHO, I, ys, iter, K, mu)
    print("方法3：半静态")
    o3 = static_opt_prov(S, J_num, X_map, load, RHO, I, ys, iter, K, mu)
    print("方法4：不带迁移代价")
    o4 = alg_without_migration_cost(S, J_num, X_map, load, RHO, I, ys, iter, K, mu)
    print("方法1:凸优化")
    print(o1)
    print("方法2：静态")
    print(o2)
    print("方法3：半静态")
    print(o3)
    print("方法4：不带迁移代价")
    print(o4)

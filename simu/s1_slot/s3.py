#!usr/bin/env python
# -*- coding:utf-8 -*-
# 稳定版，添加中间数据的持久化，网络高负载情况，增加记录拒绝服务的请求数
import datetime
import sys
import time
import numpy as np
import random

import pytz

import simu.greedy as greedy
import simu.greedy_computing as greedy_computing
import simu.greedy_down_bandwidth as greedy_down_bandwidth
import simu.greedy_up_bandwidth as greedy_up_bandwidth
import simu.greedy_resource as greedy_resource
import simu.RandomSelect as random_select
import simu.greedy_bandwidth as greedy_bandwidth
import simu.plot_bar as plot_bar
import json

import simu.plot as pt


def check(sr, rbsc, chromosome):
    m = np.size(sr, 0)
    n = np.size(rbsc, 0)
    for i in range(n):
        down_bandwidth = 0
        up_bandwidth = 0
        process = 0
        for j in range(m):
            try:
                down_bandwidth += chromosome[j][i] * sr[j][0]
            except:
                print(chromosome)
                print(sr)
            up_bandwidth += chromosome[j][i] * sr[j][1]
            process += chromosome[j][i] * sr[j][2]
        if down_bandwidth > rbsc[i][0] or up_bandwidth > rbsc[i][1] or process > rbsc[i][2]:
            return False
    return True


# 那么一个个体应该用M * N的数组表示(要求：每一行只有一个1，每一列请求的资源不能超过基站剩余资源)，所有数组应该有L*M*N大小的矩阵表示
def getInitialPopulation(sr, rbsc, populationSize, delta=0.000000001):
    m = np.size(sr, 0)
    n = np.size(rbsc, 0)
    chromosomes_list = []
    ####################################################################################
    cost, rbsc_realtime, solution = greedy_resource.greedy_min_cost(sr, rbsc, delta)
    if check(sr, rbsc, solution):
        chromosomes_list.append(solution)
        populationSize -= 1

    cost, rbsc_realtime, solution = greedy.greedy_min_cost(sr, rbsc, delta)
    if check(sr, rbsc, solution):
        chromosomes_list.append(solution)
        populationSize -= 1

    cost, rbsc_realtime, solution = greedy_down_bandwidth.greedy_min_down_bandwidth_cost(sr, rbsc, delta)
    if check(sr, rbsc, solution):
        chromosomes_list.append(solution)
        populationSize -= 1

    cost, rbsc_realtime, solution = greedy_up_bandwidth.greedy_min_up_bandwidth_cost(sr, rbsc, delta)
    if check(sr, rbsc, solution):
        chromosomes_list.append(solution)
        populationSize -= 1
    cost, rbsc_realtime, solution = greedy_computing.greedy_min_compute_cost(sr, rbsc, delta)
    if check(sr, rbsc, solution):
        chromosomes_list.append(solution)
        populationSize -= 1

    ####################################################################################
    for i in range(populationSize):
        # 随机产生一个染色体
        chromosome = np.zeros((m, n), dtype=int)
        rbsc_realtime = np.array(rbsc)
        # flag_of_matrix = 1
        # 产生一个染色体矩阵中的其中一行
        l = np.arange(m)
        if i != 0:
            np.random.shuffle(l)
        else:
            l = l[::-1]
        for j in l:
            min_cost_j = sys.maxsize
            min_bs_j = -1
            for bs_of_select in range(n):
                if sr[j][0] < rbsc_realtime[bs_of_select][0] and sr[j][1] < rbsc_realtime[bs_of_select][1] and sr[j][
                    2] < rbsc_realtime[bs_of_select][2]:
                    if (sr[j][0] / rbsc_realtime[bs_of_select][0] + sr[j][1] < rbsc_realtime[bs_of_select][1] + sr[j][
                        2] / rbsc_realtime[bs_of_select][2]) < min_cost_j:
                        min_cost_j = sr[j][0] / rbsc_realtime[bs_of_select][0] + sr[j][1] + rbsc_realtime[bs_of_select][
                            1] + sr[j][2] / rbsc_realtime[bs_of_select][2]
                        min_bs_j = bs_of_select
            if min_bs_j != -1:
                chromosome[j][min_bs_j] = 1
                rbsc_realtime[min_bs_j][0] -= sr[j][0]
                rbsc_realtime[min_bs_j][1] -= sr[j][1]
                rbsc_realtime[min_bs_j][2] -= sr[j][2]
        # 将产生的染色体加入到chromosomes_list中
        chromosomes_list.append(chromosome)
    chromosomes = np.array(chromosomes_list)
    return chromosomes


# 得到个体的适应度值(包括带宽和计算的代价)及每个个体被选择的累积概率
def getFitnessValue(sr, rbsc, chromosomes, delta):
    penalty = 10
    populations, m, n = np.shape(chromosomes)
    # 定义适应度函数，每一行代表一个染色体的适应度，每行包括四部分，分别为：带宽代价、计算代价、总代价、选择概率、累计概率
    fitness = np.zeros((populations, 6))
    for i in range(populations):
        # 取出来第i个染色体
        rbsc_realtime = np.array(rbsc)
        chromosome = chromosomes[i]
        cost_of_down_bandwidth = 0
        cost_of_up_bandwidth = 0
        cost_of_computing = 0
        for j in range(m):
            if np.sum(chromosome[j, :]) == 0:
                # cost_of_down_bandwidth += penalty
                # cost_of_up_bandwidth += penalty
                # cost_of_computing += penalty
                fitness[i][3] += 3 * penalty
                continue
            for k in range(n):
                if chromosome[j][k] == 1:
                    cost_of_down_bandwidth += sr[j][0] / (rbsc_realtime[k][0] + delta)
                    cost_of_up_bandwidth += sr[j][1] / (rbsc_realtime[k][1] + delta)
                    cost_of_computing += sr[j][2] / (rbsc_realtime[k][2] + delta)
                    rbsc_realtime[k][0] -= sr[j][0]
                    rbsc_realtime[k][1] -= sr[j][1]
                    rbsc_realtime[k][2] -= sr[j][2]
                    break
        fitness[i][0] = cost_of_down_bandwidth
        fitness[i][1] = cost_of_up_bandwidth
        fitness[i][2] = cost_of_computing
        fitness[i][3] += cost_of_down_bandwidth + cost_of_up_bandwidth + cost_of_computing
    # 计算被选择的概率
    sum_of_fitness = 0
    if populations > 1:
        for i in range(populations):
            sum_of_fitness += fitness[i][3]
        for i in range(populations):
            fitness[i][4] = (sum_of_fitness - fitness[i][3]) / ((populations - 1) * sum_of_fitness)
    else:
        fitness[0][4] = 1
    fitness[:, 5] = np.cumsum(fitness[:, 4])
    return fitness


# 选择算子
def selectNewPopulation(chromosomes, cum_probability):
    populations, m, n = np.shape(chromosomes)
    newpopulation = np.zeros((populations, m, n), dtype=int)
    # 随机产生populations个概率值
    randoms = np.random.rand(populations)
    for i, randoma in enumerate(randoms):
        logical = cum_probability >= randoma
        index = np.where(logical == 1)
        # index是tuple,tuple中元素是ndarray
        newpopulation[i, :, :] = chromosomes[index[0][0], :, :]
    return newpopulation
    pass


# 新种群交叉
def crossover(sr, rbsc, population, pc=0.8):
    """
    :param rbsc:
    :param sr:
    :param population: 新种群
    :param pc: 交叉概率默认是0.8
    :return: 交叉后得到的新种群
    """
    populations, m, n = np.shape(population)
    random_pop = np.arange(populations)
    np.random.shuffle(random_pop)
    random_pop = list(random_pop)
    # 保存交叉后得到的新种群
    updatepopulation = np.zeros((populations, m, n), dtype=int)

    while len(random_pop) > 0:
        if len(random_pop) == 1:
            updatepopulation[populations - 1] = population[random_pop.pop()]
            break
        a = random_pop.pop()
        b = random_pop.pop()
        l = len(random_pop)
        father = population[a]
        mather = population[b]
        younger_brother = np.zeros((m, n))
        elder_brother = np.zeros((m, n))
        # 此处可以增加多次探测
        for i in range(m):
            p = random.uniform(0, 1)
            if p < pc:
                younger_brother[i] = mather[i]
                elder_brother[i] = father[i]
            else:
                younger_brother[i] = father[i]
                elder_brother[i] = mather[i]
            if check(sr, rbsc, younger_brother) and check(sr, rbsc, elder_brother):
                continue
            else:
                temp = elder_brother[i]
                elder_brother[i] = younger_brother[i]
                younger_brother[i] = temp
            if check(sr, rbsc, younger_brother) and check(sr, rbsc, elder_brother):
                continue
            else:
                # 放弃杂交
                elder_brother = mather
                younger_brother = father
                break
        updatepopulation[populations - l - 2] = elder_brother
        updatepopulation[populations - l - 1] = younger_brother
    return updatepopulation
    pass


# 染色体变异
def mutation(sr, rbsc, population, pm=0.01):
    """
    :param rbsc:
    :param sr:
    :param population: 经交叉后得到的种群
    :param pm: 变异概率默认是0.01
    :return: 经变异操作后的新种群
    """
    updatepopulation = np.copy(population)
    populations, m, n = np.shape(population)
    # 计算需要变异的基因个数
    gene_num = np.uint8(populations * m * n * pm)
    # 将所有的基因按照序号进行10进制编码，则共有populations * m个基因
    # 随机抽取gene_num个基因进行基本位变异
    mutationGeneIndex = random.sample(range(0, populations * m * n), gene_num)
    # 确定每个将要变异的基因在整个染色体中的基因座(即基因的具体位置)
    for gene in mutationGeneIndex:
        # 确定变异基因位于第几个染色体
        chromosomeIndex = gene // (m * n)
        # 确定变异基因位于当前染色体的第几个基因位
        geneIndex = gene % (m * n)
        # 确定在染色体矩阵哪行
        sr_location = geneIndex // n
        # 确定在染色体矩阵哪行
        bs_location = geneIndex % n
        # mutation
        chromosome = np.array(population[chromosomeIndex])
        if chromosome[sr_location, bs_location] == 0:
            for i in range(n):
                chromosome[sr_location, i] = 0
            chromosome[sr_location, bs_location] = 1
        else:
            chromosome[sr_location, bs_location] = 0
            j = random.randint(0, n - 1)
            chromosome[sr_location, j] = 1
        if check(sr, rbsc, chromosome):
            updatepopulation[chromosomeIndex] = np.copy(chromosome)
    return updatepopulation
    pass


# 得到个体的适应度值(包括带宽和计算的代价)及每个个体被选择的累积概率
def update_rbsc(sr, rbsc, solution):
    m, n = np.shape(solution)
    rbsc_realtime = np.array(rbsc)
    chromosome = solution
    for j in range(m):
        for k in range(n):
            if chromosome[j][k] == 1:
                rbsc_realtime[k][0] -= sr[j][0]
                rbsc_realtime[k][1] -= sr[j][1]
                rbsc_realtime[k][2] -= sr[j][2]
                break
    return rbsc_realtime


def ga(SR, RBSC, max_iter=500, delta=0.0001, pc=0.8, pm=0.01, populationSize=10):
    # 每次迭代得到的最优解
    optimalSolutions = []
    optimalValues = []

    # 边界处理，当请求数只有1个时候
    m = np.size(SR, 0)
    n = np.size(RBSC, 0)

    if m == 1:
        chromosomes = np.zeros((n, 1, n))
        for i in range(n):
            chromosomes[i][0][i] = 1
        check_list = []
        for i in range(n):
            if check(SR, RBSC, chromosomes[i]):
                check_list.append(i)
        if len(check_list) == 0:
            return "failed", -1
        chromosomes = np.zeros((len(check_list), 1, n))
        for i in range(len(check_list)):
            chromosomes[i][0][check_list[i]] = 1

        fitness = getFitnessValue(SR, RBSC, chromosomes, delta)
        optimalValues.append(np.min(list(fitness[:, 3])))
        index = np.where(fitness[:, 3] == min(list(fitness[:, 3])))
        optimalSolutions.append(chromosomes[index[0][0], :, :])
        optimalValue = np.min(optimalValues)
        optimalIndex = np.where(optimalValues == optimalValue)
        optimalSolution = optimalSolutions[optimalIndex[0][0]]
        return optimalSolution, optimalValue

    # 得到初始种群编码
    chromosomes = getInitialPopulation(SR, RBSC, populationSize)
    population_num = np.size(chromosomes, 0)
    if population_num == 0:
        return "failed", -1

    fitness = getFitnessValue(SR, RBSC, chromosomes, delta)
    optimalValues.append(np.min(list(fitness[:, 3])))
    index = np.where(fitness[:, 3] == min(list(fitness[:, 3])))
    optimalSolutions.append(chromosomes[index[0][0], :, :])

    for iteration in range(max_iter):
        # 得到个体适应度值和个体的累积概率
        fitness = getFitnessValue(SR, RBSC, chromosomes, delta)
        # 选择新的种群
        cum_proba = fitness[:, 5]
        try:
            newpopulations = selectNewPopulation(chromosomes, cum_proba)
        except:
            print("except in ga:", population_num, chromosomes, np.size(chromosomes, 0), np.shape(chromosomes))
        # 进行交叉操作
        crossoverpopulation = crossover(SR, RBSC, newpopulations, pc)
        # mutation
        mutationpopulation = mutation(SR, RBSC, crossoverpopulation, pm)
        # 适应度评价
        fitness = getFitnessValue(SR, RBSC, mutationpopulation, delta)
        # 搜索每次迭代的最优解，以及最优解对应的目标函数的取值
        optimalValues.append(np.min(list(fitness[:, 3])))
        index = np.where(fitness[:, 3] == min(list(fitness[:, 3])))
        optimalSolutions.append(mutationpopulation[index[0][0], :, :])
        chromosomes = mutationpopulation
    # 搜索最优解
    optimalValue = np.min(optimalValues)
    optimalIndex = np.where(optimalValues == optimalValue)
    optimalSolution = optimalSolutions[optimalIndex[0][0]]
    return optimalSolution, optimalValue


def getRbsc(bs_num):
    rbsc = np.zeros((bs_num, 3), dtype=np.float)
    # rbsc = 1.5 - rbsc
    # r1 = 5
    # r2 = 3
    # r3 = 1
    r1 = 5
    r2 = 3
    r3 = 1
    rbsc[0][0] = r1
    rbsc[0][1] = r2
    rbsc[0][2] = r3
    rbsc[1][0] = r1
    rbsc[1][1] = r3
    rbsc[1][2] = r2
    rbsc[2][0] = r2
    rbsc[2][1] = r3
    rbsc[2][2] = r1
    rbsc[3][0] = r2
    rbsc[3][1] = r1
    rbsc[3][2] = r3
    rbsc[4][0] = r3
    rbsc[4][1] = r1
    rbsc[4][2] = r2
    rbsc[5][0] = r3
    rbsc[5][1] = r2
    rbsc[5][2] = r1
    # rbsc[6][0] = r2
    # rbsc[6][1] = r2
    # rbsc[6][2] = r2
    return rbsc


def simu(request_num=15, req_num_eachtime=4, sigma=50000, max_iter=1, bs_num=6):
    # bs_num = 6
    # BSC：base station capacity
    # RBSC: residuary base station capacity
    # SR: slice request
    # max_iter = 1  # ------------------------
    delta = 0.000000001
    pc = 0.8
    pm = 0.01
    # req_num_eachtime = 4
    # 构造request_num次请求
    # request_num = 15  # --------------------------
    values = np.zeros((request_num), dtype=np.float)
    solutions = []
    sr_all = []
    rbscs = []
    # 每轮处理请求失败的切片请求数，fails[0]是遗传、fails[1]是贪心总代价、fails[2]是贪心下行带宽、fails[3]是贪心上行带宽、fails[4]是贪心计算资源
    fails = np.zeros((7, request_num))
    # 记录7中算法每次迭代得到下行，上行，计算，总代价
    cost_result = np.zeros((7, request_num, 4), dtype=np.float)
    resource_used_radio = np.zeros((7, bs_num, 3), dtype=np.float)
    time_resouce_used = request_num - 1
    # sigma = 50000
    # 构造m个切片请求
    m = req_num_eachtime * request_num
    sr_total = np.zeros((m, 3), dtype=np.float)
    for i in range(m):
        s = np.abs(np.random.normal(100, sigma, 3)) + 1
        s = s / (sum(s))
        sr_total[i] = s
    for iter in range(request_num):
        # 随机构造每次请求的切片数
        m = (iter + 1) * req_num_eachtime
        # 构造基站资源
        rbsc = getRbsc(bs_num)
        total_rbsc = np.sum(rbsc, 0)  # 求每列之和，得到1*3向量，分别表示下行，上行，计算资源总量
        # 构造m个切片请求
        sr = np.zeros((m, 3), dtype=np.float)
        for i in range(m):
            s = sr_total[i]
            sr[i] = s

        rbscs.append(rbsc)
        print("rbsc:")
        print(rbsc)
        print("sr:")
        print(sr)
        sr_all.append(sr)  # 记录请求，为其他算法提供相同的请求环境
        populationSize = min(50, m * bs_num)
        solution, value = ga(sr, rbsc, max_iter, delta, pc, pm, populationSize)

        # 资源紧张的时候，采用greedy算法，得到可以满足的情况
        while solution == "failed" and np.size(sr, 0) >= 2:
            cost, rbsc_r, solution = greedy.greedy_min_cost(sr, rbsc, delta)
            x1 = np.sum(solution, 1)  # 求每行之和

            cost, rbsc_r, solution = greedy_resource.greedy_min_cost(sr, rbsc, delta)
            x2 = np.sum(solution, 1)  # 求每行之和

            cost, rbsc_r, solution = greedy_down_bandwidth.greedy_min_down_bandwidth_cost(sr, rbsc, delta)
            x3 = np.sum(solution, 1)  # 求每行之和

            cost, rbsc_r, solution = greedy_up_bandwidth.greedy_min_up_bandwidth_cost(sr, rbsc, delta)
            x4 = np.sum(solution, 1)  # 求每行之和

            cost, rbsc_r, solution = greedy_computing.greedy_min_compute_cost(sr, rbsc, delta)
            x5 = np.sum(solution, 1)  # 求每行之和
            XX = np.array((x1, x2, x3, x4, x5))
            X = np.array((np.sum(x1), np.sum(x2), np.sum(x3), np.sum(x4), np.sum(x5)))
            x = np.max(X)
            if x == 0:
                solution == "failed"
                value = 0
                sr = np.array([])
                break
            index = np.where(X == x)
            x = XX[index[0][0]]
            sr_list = []
            for s in range(np.size(x)):
                if x[s] == 1:
                    sr_list.append(sr[s])
            sr = np.array(sr_list)
            solution, value = ga(sr, rbsc, max_iter, delta, pc, pm, populationSize)
        # 记录失败数目
        fails[0][iter] = np.size(sr_all[iter], 0) - np.sum(np.sum(solution))
        print('最优目标函数值:', value)
        values[iter] = value
        print('solution:')
        print(solution)
        ##############################
        # 持久化结果
        fit = getFitnessValue(sr, rbsc, [solution], delta)
        o = [fit[0, 0], fit[0, 1], fit[0, 2], fit[0, 3]]
        cost_result[0][iter][0] = fit[0, 0]
        cost_result[0][iter][1] = fit[0, 1]
        cost_result[0][iter][2] = fit[0, 2]
        cost_result[0][iter][3] = fit[0, 0] + fit[0, 1] + fit[0, 2]
        ##############################
        solutions.append(np.copy(solution))
        if iter == time_resouce_used:
            rbsc_init = getRbsc(bs_num)
            rbsc = update_rbsc(sr, rbsc_init, solution)
            for bs in range(bs_num):
                for resource_type in range(3):
                    resource_used_radio[0][bs][resource_type] = rbsc[bs][resource_type] / rbsc_init[bs][resource_type]
    print("ga总结果")
    print(values)
    # print(rbsc)
    ###########################################################################################################
    for i in range(request_num):
        sr = sr_all[i]
        rbsc = rbscs[i]
        cost, rbsc, solution = greedy.greedy_min_cost(sr, rbsc, delta)
        values[i] = cost
        ##############################
        # 持久化结果
        fit = getFitnessValue(sr, rbscs[i], [solution], delta)
        o = [fit[0, 0], fit[0, 1], fit[0, 2], fit[0, 3]]
        cost_result[1][i][0] = fit[0, 0]
        cost_result[1][i][1] = fit[0, 1]
        cost_result[1][i][2] = fit[0, 2]
        cost_result[1][i][3] = fit[0, 0] + fit[0, 1] + fit[0, 2]
        ##############################
        # 记录失败数
        fails[1][i] = np.size(sr, 0) - np.sum(np.sum(solution, 0), 0)
        if i == time_resouce_used:
            rbsc_init = getRbsc(bs_num)
            rbsc = update_rbsc(sr, rbsc_init, solution)
            for bs in range(bs_num):
                for resource_type in range(3):
                    resource_used_radio[1][bs][resource_type] = rbsc[bs][resource_type] / rbsc_init[bs][resource_type]
    print("greedy_min_cost总结果")
    print(values)
    ##############################################################################################################
    for i in range(request_num):
        sr = sr_all[i]
        rbsc = rbscs[i]
        # cost, rbsc, solution = greedy_down_bandwidth.greedy_min_down_bandwidth_cost(sr, rbsc, delta)
        cost, rbsc, solution = greedy_bandwidth.greedy_min_bandwidth_cost(sr, rbsc, delta)
        values[i] = cost
        ##############################
        # 持久化结果
        fit = getFitnessValue(sr, rbscs[i], [solution], delta)
        o = [fit[0, 0], fit[0, 1], fit[0, 2], fit[0, 3]]
        cost_result[2][i][0] = fit[0, 0]
        cost_result[2][i][1] = fit[0, 1]
        cost_result[2][i][2] = fit[0, 2]
        cost_result[2][i][3] = fit[0, 0] + fit[0, 1] + fit[0, 2]
        ##############################
        # 记录失败数
        fails[2][i] = np.size(sr, 0) - np.sum(np.sum(solution, 0), 0)
        if i == time_resouce_used:
            rbsc_init = getRbsc(bs_num)
            rbsc = update_rbsc(sr, rbsc_init, solution)
            for bs in range(bs_num):
                for resource_type in range(3):
                    resource_used_radio[2][bs][resource_type] = rbsc[bs][resource_type] / rbsc_init[bs][resource_type]
    print("greedy_min_bandwidth_cost总结果")
    print(values)
    ##############################################################################################################
    for i in range(request_num):
        sr = sr_all[i]
        rbsc = rbscs[i]
        cost, rbsc, solution = greedy_up_bandwidth.greedy_min_up_bandwidth_cost(sr, rbsc, delta)
        values[i] = cost
        ##############################
        # 持久化结果
        fit = getFitnessValue(sr, rbscs[i], [solution], delta)
        o = [fit[0, 0], fit[0, 1], fit[0, 2], fit[0, 3]]
        cost_result[3][i][0] = fit[0, 0]
        cost_result[3][i][1] = fit[0, 1]
        cost_result[3][i][2] = fit[0, 2]
        cost_result[3][i][3] = fit[0, 0] + fit[0, 1] + fit[0, 2]
        ##############################
        # 记录失败数
        fails[3][i] = np.size(sr, 0) - np.sum(np.sum(solution, 0), 0)
        if i == time_resouce_used:
            rbsc_init = getRbsc(bs_num)
            rbsc = update_rbsc(sr, rbsc_init, solution)
            for bs in range(bs_num):
                for resource_type in range(3):
                    resource_used_radio[3][bs][resource_type] = rbsc[bs][resource_type] / rbsc_init[bs][resource_type]
    print("greedy_min_up_bandwidth_cost总结果")
    print(values)
    ##############################################################################################################
    for i in range(request_num):
        sr = sr_all[i]
        rbsc = rbscs[i]
        cost, rbsc, solution = greedy_computing.greedy_min_compute_cost(sr, rbsc, delta)
        values[i] = cost
        ##############################
        # 持久化结果
        fit = getFitnessValue(sr, rbscs[i], [solution], delta)
        o = [fit[0, 0], fit[0, 1], fit[0, 2], fit[0, 3]]
        cost_result[4][i][0] = fit[0, 0]
        cost_result[4][i][1] = fit[0, 1]
        cost_result[4][i][2] = fit[0, 2]
        cost_result[4][i][3] = fit[0, 0] + fit[0, 1] + fit[0, 2]
        ##############################
        # 记录失败数
        fails[4][i] = np.size(sr, 0) - np.sum(np.sum(solution, 0))
        if i == time_resouce_used:
            rbsc_init = getRbsc(bs_num)
            rbsc = update_rbsc(sr, rbsc_init, solution)
            for bs in range(bs_num):
                for resource_type in range(3):
                    resource_used_radio[4][bs][resource_type] = rbsc[bs][resource_type] / rbsc_init[bs][resource_type]
    print("greedy_min_compute_cost总结果")
    print(values)
    ##############################################################################################################
    for i in range(request_num):
        sr = sr_all[i]
        rbsc = rbscs[i]
        cost, rbsc, solution = greedy_resource.greedy_min_cost(sr, rbsc, delta)
        values[i] = cost
        ##############################
        # 持久化结果
        fit = getFitnessValue(sr, rbscs[i], [solution], delta)
        o = [fit[0, 0], fit[0, 1], fit[0, 2], fit[0, 3]]
        cost_result[5][i][0] = fit[0, 0]
        cost_result[5][i][1] = fit[0, 1]
        cost_result[5][i][2] = fit[0, 2]
        cost_result[5][i][3] = fit[0, 0] + fit[0, 1] + fit[0, 2]
        # 记录失败数
        fails[5][i] = np.size(sr, 0) - np.sum(np.sum(solution, 0), 0)
        if i == time_resouce_used:
            rbsc_init = getRbsc(bs_num)
            rbsc = update_rbsc(sr, rbsc_init, solution)
            for bs in range(bs_num):
                for resource_type in range(3):
                    resource_used_radio[5][bs][resource_type] = rbsc[bs][resource_type] / rbsc_init[bs][resource_type]
    print("greedy_min_max_cost总结果")
    print(values)
    ##############################################################################################################
    for i in range(request_num):
        sr = sr_all[i]
        rbsc = rbscs[i]
        cost, rbsc, solution = random_select.random_select(sr, rbsc, delta)
        values[i] = cost
        ##############################
        # 持久化结果
        fit = getFitnessValue(sr, rbscs[i], [solution], delta)
        o = [fit[0, 0], fit[0, 1], fit[0, 2], fit[0, 3]]
        cost_result[6][i][0] = fit[0, 0]
        cost_result[6][i][1] = fit[0, 1]
        cost_result[6][i][2] = fit[0, 2]
        cost_result[6][i][3] = fit[0, 0] + fit[0, 1] + fit[0, 2]
        # 记录失败数
        fails[6][i] = np.size(sr, 0) - np.sum(np.sum(solution, 0), 0)
        if i == time_resouce_used:
            rbsc_init = getRbsc(bs_num)
            rbsc = update_rbsc(sr, rbsc_init, solution)
            for bs in range(bs_num):
                for resource_type in range(3):
                    resource_used_radio[6][bs][resource_type] = rbsc[bs][resource_type] / rbsc_init[bs][resource_type]
    print("random总结果")
    print(values)
    ##############################################################################################################
    print(fails)
    return cost_result, fails, resource_used_radio


# 时间估算：1小时只能跑1200
if __name__ == '__main__':
    request_num = 12
    req_num_eachtime = 6
    sigma = 5000
    ###############
    # 遗传算法迭代次数
    max_iter = 1
    # 多次取平均
    n = 1
    tz = pytz.timezone('Asia/Shanghai')  # 东八区
    print(max_iter)
    print(n)
    ###############
    bs_num = 6
    cost_result = np.zeros((7, request_num, 4), dtype=np.float)
    fails = np.zeros((7, request_num))
    resource_used_radio = np.zeros((7, bs_num, 3), dtype=np.float)
    resource_radio = np.zeros((7, 3), dtype=np.float)
    for i in range(n):
        print('iter:')
        print(i)
        t = datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone('Asia/Shanghai')).strftime(
            '%Y-%m-%d %H:%M:%S')
        print(t)
        cost_result_, fails_, resource_used_radio_ = simu(request_num, req_num_eachtime, sigma, max_iter, bs_num)
        cost_result += cost_result_
        fails += fails_
        resource_used_radio += resource_used_radio_
    cost_result /= n
    fails /= n
    resource_used_radio /= n
    nowtime = (lambda: int(round(time.time() * 1000)))
    nowtime = nowtime()
    pt.plot_fun_slot(cost_result[:, :, 0], fails, req_num_eachtime, '切片请求数量（个）', '平均下行带宽映射代价',
                     str(nowtime) + '下行带宽映射代价')
    pt.plot_fun_slot(cost_result[:, :, 1], fails, req_num_eachtime, '切片请求数量（个）', '平均上行带宽映射代价',
                     str(nowtime) + '上行带宽映射代价')
    pt.plot_fun_slot(cost_result[:, :, 2], fails, req_num_eachtime, '切片请求数量（个）', 1,
                     str(nowtime) + '计算资源映射代价')
    pt.plot_fun_slot((cost_result[:, :, 0] + cost_result[:, :, 1]), fails, req_num_eachtime, '切片请求数量（个）',
                     2,
                     str(nowtime) + '带宽资源映射代价')
    pt.plot_fun_slot(cost_result[:, :, 3], fails, req_num_eachtime, '切片请求数量（个）', 3,
                     str(nowtime) + '总映射代价' + '_' + str(max_iter) + '_' + str(n))
    pt.plot_fun_fail_slot(fails, req_num_eachtime, '切片请求数量（个）', '失败率', str(nowtime) + '失败率')
    print(cost_result)
    rbsc = getRbsc(bs_num)
    r1 = rbsc[0][0]
    r2 = rbsc[0][1]
    r3 = rbsc[0][2]
    for algorithm in range(7):
        resource_radio[algorithm, :] = 1 - (np.sum(resource_used_radio[algorithm, :, :], 0) / ((r1 + r2 + r3) * 6))
    plot_bar.plat_bar(resource_radio, str(nowtime) + '资源使用率')
    print("结果")
    print("cost_result")
    print(cost_result)
    print("fails")
    print(fails)
    print("req_num_eachtime")
    print(req_num_eachtime)
    print("resource_radio")
    print(resource_radio)

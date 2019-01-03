#!usr/bin/env python
# -*- coding:utf-8 -*-
# import numpy as np
import numpy as np
from scipy.optimize import minimize, linprog
def generate_k(S, multiple):
    k = np.random.uniform(1, 100, S)  # todo(*可调参)
    sum = np.sum(k)
    k = multiple * k / sum
    return k


def generate_K(S, iter):
    K = np.zeros((iter, S))
    multiples = np.arange(iter) + 1
    for i in range(iter):
        K[i] = generate_k(S, multiples[i])
    return K


print(generate_K(18,10))
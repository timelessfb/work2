#!usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np


def slice(J, C, cover=3, delta=0.1, alpha=10):
    rho1, rho2 = np.random.uniform(0, alpha, 2)
    Rs = C * (np.random.rand() + delta)
    Js = np.random.choice(J, cover)
    return Rs, rho1, rho2, Js


def BSs(J, dowm=10, up=10, compute=10):
    bss = np.zeros((J, 3))
    for i in range(J):
        bss[i][0] = dowm
        bss[i][1] = up
        bss[i][2] = compute


def slices(J, S, C, cover=3, delta=0.1, alpha=10):
    scs = np.zeros((S, 6))  # Rs, rho1, rho2, Js, i, j
    for i in range(S):
        scs[0], scs[1], scs[2], scs[3] = slice(J, C, cover, delta, alpha)
    return scs


if __name__ == '__main__':
    print(slices(6, 10, ))
    Rs, rho1, rho2, Js = slice(6, 3)
    print(Rs, rho1, rho2, Js)

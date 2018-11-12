#!usr/bin/env python
# -*- coding:utf-8 -*-

import math
import numpy as np
import random
import matplotlib.pylab as plt
import json

# a = np.zeros((2, 3))
# print(a)
# a[0][1] = 2
# obj = {1: a.tolist()}
# fp = open('result.json', 'w')
# json.dump(obj, fp)
# fp.close()
# # s = json.dumps(obj)
# x = json.load(open('result.json', 'r'))
# print(x['1'])
a = np.zeros((2, 3), dtype=np.float)
a[0][0] = 1
a[0][1] = 2
a[0][2] = 3
a[1][0] = 4
a[1][1] = 5
a[1][2] = 6
a = a / sum(a)
a = math.log(1, 2)
print(a)
A = 1.00823439
B = 1.00924825
# A = math.exp(A)
# B = math.exp(B)
A=math.log(A,1.2)
B=math.log(B,1.2)
print(A, B)

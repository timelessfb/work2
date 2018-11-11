#!usr/bin/env python
# -*- coding:utf-8 -*-

import math
import numpy as np
import random
import matplotlib.pylab as plt
import json

a = np.zeros((2, 3))
print(a)
a[0][1] = 2
obj = {1: a.tolist()}
fp = open('result.json', 'w')
json.dump(obj, fp)
fp.close()
# s = json.dumps(obj)
x = json.load(open('result.json', 'r'))
print(x['1'])

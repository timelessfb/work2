#!usr/bin/env python
# -*- coding:utf-8 -*-
# import numpy as np
import numpy as np
from scipy.optimize import minimize
import copy


def fun(type, x):
    def f1(x):
        if type == 1:
            return x[0] - x[1]
        else:
            return x[0] + x[1]
    return f1


cons = []
s = [0, 1]
for i in s:
    cons.append(fun(i,s))

# cons.append(lambda x:fun(1,x))
# cons.append(lambda x:fun(0,x))
# f1=lambda x:fun(0,x)
# f2=lambda x:fun(1,x)
f1 = cons[0]
f2 = cons[1]
x = [1, 2]
print(f1(x))
print(f2(x))
f = lambda x, y: x + y
print(f(3, 2))

# def generate_constraints_wrong(n_params):
#     cons = []
#     for i in range(n_params):
#         cons.append({'type': 'ineq', 'fun': lambda x:  x[i]})
#     return tuple(cons)
#
# def generate_constraints_wrong2(n_params):
#     cons = tuple()
#     for i in range(n_params):
#         cons += ({'type': 'ineq', 'fun': lambda x:  x[i]},)
#     return cons

# def generate_constraints_right(n_params):
#     # let's create a function generator that uses closure to pass i to the generated function
#     def wrapper_fun(x, i):
#         def fun(x):
#             return x[i]
#         return fun
#     cons = []
#     for i in range(n_params):
#         f = wrapper_fun(x, i)
#         cons.append({'type': 'ineq', 'fun': f})
#     return tuple(cons)
#
# # verify the generated functions
# n_params = 3
# x = [1,10, 100]
# cons1 = generate_constraints_wrong(n_params)
# cons2 = generate_constraints_wrong2(n_params)
# cons3 = generate_constraints_right(n_params)
# print(cons1[0]['fun'](x)) # this should be 1 but instead we end up modifying all of our lambda objects to have the last i
# print(cons1[1]['fun'](x))
# print(cons2[0]['fun'](x))
# print(cons2[1]['fun'](x))
# print(cons3[0]['fun'](x))
# print(cons3[1]['fun'](x))

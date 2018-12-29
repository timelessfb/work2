import numpy as np
from scipy.optimize import minimize


def objective(x):
    return x[0] ** 3 - x[1] ** 3 - x[0] * x[1] + 2 * x[0] ** 2


def constraint1(x):
    return -x[0] ** 2 - x[1] ** 2 + 6.0


def constraint2(x):
    return 2.0 - x[0] * x[1]


def s1(i, x):
    if i == 1:
        return x[0] - x[1]
    else:
        return x[0] + [1]


# initial guesses
n = 2
x0 = np.zeros(n)
x0[0] = 1
x0[1] = 2

# show initial objective
print('Initial SSE Objective: ' + str(objective(x0)))

# optimize
b = (0.0, 3.0)
bnds = (b, b)
con1 = {'type': 'ineq', 'fun': lambda x: s1(1,x)}
con2 = {'type': 'eq', 'fun': lambda x: 2.0 - x[0] * x[1]}
cons = ([con1, con2])
print(cons)
print(type(cons))
solution = minimize(objective, x0, method='SLSQP', \
                    bounds=bnds, constraints=cons)
x = solution.x

# show final objective
print('Final SSE Objective: ' + str(objective(x)))

# print solution
print('Solution')
print('x1 = ' + str(x[0]))
print('x2 = ' + str(x[1]))

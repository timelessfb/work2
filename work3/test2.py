import numpy as np
import pandas as pd
import scipy.optimize as op


def LoadData(filename):
    data = pd.read_csv(filename, header=None)
    data = np.array(data)
    return data


def ReshapeData(data):
    m = np.size(data, 0)
    X = data[:, 0:2]
    Y = data[:, 2]
    Y = Y.reshape((m, 1))
    return X, Y


def InitData(X):
    m, n = X.shape
    initial_theta = np.zeros(n + 1)
    VecOnes = np.ones((m, 1))
    X = np.column_stack((VecOnes, X))
    return X, initial_theta


def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z


def costFunction(theta, X, Y):
    m = X.shape[0]
    J = (-np.dot(Y.T, np.log(sigmoid(X.dot(theta)))) - \
         np.dot((1 - Y).T, np.log(1 - sigmoid(X.dot(theta))))) / m
    return J


def gradient(theta, X, Y):
    m, n = X.shape
    theta = theta.reshape((n, 1))
    grad = np.dot(X.T, sigmoid(X.dot(theta)) - Y) / m
    return grad.flatten()


if __name__ == '__main__':
    data = LoadData('ex2data1csv.csv')
    X, Y = ReshapeData(data)
    X, initial_theta = InitData(X)
    result = op.minimize(fun=costFunction, x0=initial_theta, args=(X, Y), method='TNC', jac=gradient)
    print(result)

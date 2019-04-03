import numpy as np


def linear_cost(theta, X, y):
    m, n = X.shape()
    h = np.matmul(X, theta)
    sq = (y - h) ** 2
    return sq.sum() / (2*m)


def linear_cost_derivate(theta, X, y):
    m, n = X.shape()
    h = np.matmul(X, theta)
    sq = (y - h)
    s = sq * theta
    return s.sum() / m

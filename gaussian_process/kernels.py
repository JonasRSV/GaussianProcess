import numpy as np


def linear(x, y) -> np.dtype:
    if type(x) != type(y):
        raise Exception("Mismatching types: {} {}".format(type(x), type(y)))

    if type(x) == np.ndarray:
        return x.T @ y + 1e-2

    return x * y + 1e-2


def squared_exponential(variance, l):
    def kernel(x, y):
        return variance * np.exp(-l * np.square(np.abs(x - y) + 1e-6))

    return kernel


def brownian_motion(l):
    def kernel(x, y):
        return l * min(np.minimum(x, y).reshape(-1)).reshape(-1)

    return kernel

def Lp(p):
    def kernel(X, Y):
        return np.float_power(np.sum(np.float_power(np.abs(X - Y), p)), 1 / p)
    return kernel


def periodic(l, b):
    def kernel(x, y):
        return np.exp(-l * np.square(np.sin(np.pi * b * (x - y + 1e-6))))

    return kernel

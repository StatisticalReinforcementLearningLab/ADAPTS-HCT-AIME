import numpy as np


def _Bernoulli(p):
    return np.random.binomial(1, p)


def _standard_logistic(x):
    return 1 / (1 + np.exp(-x))


def Bernoulli_logistic(p):
    return _Bernoulli(_standard_logistic(p))


def integer_to_binary(n, length):
    return np.array([int(x) for x in bin(n)[2:].zfill(length)])


def binary_to_integer(b):
    for i in range(3):
        if np.isnan(b[i]):
            b[i] = 0
    return int("".join(str(x) for x in b), 2)
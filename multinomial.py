import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
from functools import reduce

import cma

def prod(xs):
    return reduce(lambda a,b: a*b, xs, 1.0)

def bin(n, k):
    fs = [n - o for o in range(k)]
    return prod(fs) / math.factorial(k)

def pmf(n, k):
    N = sum(n)
    K = sum(k)
    ps = [bin(ni, ki) for ni, ki in zip(n, k)]
    return prod(ps) / bin(N, K)

def outcomes(n, K):
    if len(n) == 1:
        return [[K]]
    else:
        rest = n[1:]
        nr = sum(rest)
        r_min = max(0, int(math.ceil(K - nr)))
        r_max = min(K , int(math.floor(n[0])))
        r = range(r_min, r_max + 1)
        return [[k] + o for k in r for o in outcomes(rest, K - k)]

N = 60
K = 4
def condition(k):
    return 2 <= k[0] and 1 <= k[1] and 1 <= k[2] and 3 <= k[3]

def vector_to_deck(x):
    e = np.exp(x)
    n = [N * ei / sum(e) for ei in e]
    return n

def loss(x):
    n = vector_to_deck(x)
    ks = [k for k in outcomes(n, 7) if condition(k)]
    ps = [pmf(n, k) for k in ks]
    return -sum(ps)

xopt, es = cma.fmin2(loss, K * [0], 0.5)
deck = [round(di, 2) for di in vector_to_deck(xopt)]
print(deck)
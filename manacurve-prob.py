
import numpy as np
from scipy.optimize import minimize
from scipy import stats
import matplotlib.pyplot as plt

def binomial(a, k):
    b = [(a - i) / (k - i) for i in range(k)]
    return max(0, np.prod(b))

def fractional_hypergeometric(M, n, N):
    x = range(N+1)
    p = [binomial(n, k) * binomial(M-n, N-k) / binomial(M, N) for k in x]
    return stats.rv_discrete(values=(x, p))

def combine(x, y, f):
    z = {}
    for xk in x.xk:
        for yk in y.xk:
            zk = f(xk, yk)
            pk = x.pmf(xk) * y.pmf(yk)
            if zk in z:
                z[zk] += pk
            else:
                z[zk] = pk
    keys = list(z.keys())
    values = list(z.values())
    return stats.rv_discrete(values=(keys, values))

def distribution(x):
    e = np.exp(x)
    es = sum(e)
    p = np.array([ei / es for ei in e])
    return p

def mix(x, y, a=0.5):
    keys = np.unique(np.concatenate((x.xk, y.xk)))
    values = [(1 - a) * x.pmf(k) + a * y.pmf(k) for k in keys]
    return stats.rv_discrete(values=(keys, values))

def toDeck(x, numCards=60):
    return numCards * distribution(np.concatenate([x, [0.0]]))

def closerTo(x):
    def f(a, b):
        if abs(a - x) < abs(b - x):
            return a
        else:
            return b
    return f

def landsDrawnSmooth(deck, numHands):

    landsExpected = 7 * deck[0] / sum(deck)
    landsDrawn = fractional_hypergeometric(sum(deck), deck[0], 7)

    l1 = landsDrawn
    for i in range(numHands - 1):
        landsDrawn = combine(landsDrawn, l1, closerTo(landsExpected))

    return landsDrawn

def fitness(x):

    deck = toDeck(x)
    lands = fractional_hypergeometric(sum(deck), deck[0], 7)
    return (lands.expect() - 2.5)**2

if __name__ == '__main__':    

    # l = range(0, 61)
    # decks = [[li, 60 - li] for li in l]

    # x = [di[0] for di in decks]
    # y1 = [landsDrawnSmooth(di, 1).expect() for di in decks]
    # y2 = [landsDrawnSmooth(di, 2).expect() for di in decks]
    # y3 = [landsDrawnSmooth(di, 3).expect() for di in decks]

    # plt.plot(x, y1)
    # plt.plot(x, y2)
    # plt.plot(x, y3)
    # plt.show()



    res = minimize(fitness, np.array([0.0]), method='nelder-mead')

    deck = toDeck(res.x)
    print(deck)
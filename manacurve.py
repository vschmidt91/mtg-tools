
import numpy as np
import cma

def play(deck, numTurns):

    draw = np.random.uniform() < 0.5
    hand = list(np.random.choice(len(deck), 7, p=deck))

    turns = []
    lands = 0
    for t in range(numTurns):
        turn = []
        if (t > 0) or draw:
            newCard = np.random.choice(len(deck), p=deck)
            hand.append(newCard)
        if 0 in hand:
            hand.remove(0)
            turn.append(0)
            lands += 1
        mana = lands
        d = mana
        while 0 < d:
            if (d in hand) & (d <= mana):
                mana -= d
                hand.remove(d)
                turn.append(d)
            else:
                d -= 1
        turns.append(turn)
    return turns

def valueByCmc(cmc):
    # return math.floor(0.5 + 1.5 * cmc)
    return 1 + cmc if cmc > 0 else 0

def evaluateByMana(turns):
    return sum([sum([valueByCmc(c) for c in t]) for t in turns])
    
def vectorToDeck(x):
    xe = np.exp(x)
    xs = xe / sum(xe)
    return xs

numSamples = 256
numTurns = 5

def loss(x):
    deck = vectorToDeck(x)
    games = [play(deck, numTurns) for _ in range(numSamples)]
    fd = np.average([evaluateByMana(g) for g in games])
    return -1.0 * fd

xopt, es = cma.fmin2(loss, (1 + numTurns) * [0], 0.5)

deck = tuple([round(xi * 60, 1) for xi in vectorToDeck(xopt)])
print(deck)
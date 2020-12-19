
import numpy as np
import cma

def play(probs, numTurns):

    draw = np.random.uniform() < 0.5
    hand = list(np.random.choice(len(probs), 7, p=probs))

    turns = []
    lands = 0
    for t in range(numTurns):
        turn = []
        if 0 < t or draw:
            newCard = np.random.choice(len(probs), p=probs)
            hand.append(newCard)
        if 0 in hand:
            hand.remove(0)
            turn.append(0)
            lands += 1
        mana = lands
        d = mana
        while 0 < d and 0 < mana:
            if d in hand and d <= mana:
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

def roundDeck(x, n):
    rounded = [int(round(xi * n)) for xi in x]
    rounded[0] += n - sum(rounded)
    return rounded

def vectorToDeck(x):
    xe = np.exp(x)
    xs = xe / sum(xe)
    return xs

numSamples = 1000
numTurns = 4
numCards = 15

table = {}
def lossDiscrete(x):
    deck = roundDeck(vectorToDeck(x), numCards)
    key = hash(frozenset(deck))
    oldValue, count = 0, 0
    if key in table:
        oldValue, count = table[key]
    probs = [di / sum(deck) for di in deck]
    games = [play(probs, numTurns) for _ in range(numSamples)]
    newValue = np.average([evaluateByMana(g) for g in games])
    value = (oldValue * count + newValue * numSamples) / (count + numSamples)
    count += numSamples
    table[key] = value, count
    return -value

def lossContinuous(x):
    deck = vectorToDeck(x)
    probs = [di / sum(deck) for di in deck]
    games = [play(probs, numTurns) for _ in range(numSamples)]
    value = np.average([evaluateByMana(g) for g in games])
    return -value

def loss(x):
    return lossDiscrete(x)

es = cma.CMAEvolutionStrategy((1 + numTurns) * [0], 1)

while not es.stop():

    solutions = es.ask()
    losses = [loss(x) for x in solutions]
    es.tell(solutions, losses)

    pop = list(zip(solutions, losses))
    pop.sort(key = lambda p : p[1])
    opt, val = pop[0]
    # opt = es.mean
    # val = loss(opt)
    print(roundDeck(vectorToDeck(opt), numCards), -val)

    # es.logger.add()
    # es.disp()

es.result_pretty()
cma.plot()  

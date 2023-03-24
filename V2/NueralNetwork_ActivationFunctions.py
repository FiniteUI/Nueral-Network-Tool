import math

def sigmoid(value):
    s = 0 - value
    s = math.pow(math.e, s)
    s += 1
    s = 1/s
    return s

def hyperbolicTangent(value):
    return math.tanh(value)

def rectifiedLinearUnits(value):
    if value < 0:
        value = 0
    return value

def positiveNegative(value):
    if value <= 0:
        value = 0
    else:
        value = 1
    return value

def direct(value):
    return value
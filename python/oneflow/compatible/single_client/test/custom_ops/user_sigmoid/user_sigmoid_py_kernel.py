import numpy as np

def forward(args):
    print('user sigmoid forward args', args)
    (x,) = args
    y = 1 / (1 + np.exp(-x))
    return y

def backward(args):
    print('user sigmoid backward args', args)
    (y, dy) = args
    return y * (1 - y) * dy
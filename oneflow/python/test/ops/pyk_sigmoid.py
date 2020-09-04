import numpy as np

i = 0

def sigmoid(x):
    global i
    print("i", i)
    i += 1
    print("run py kernel x", x)
    y = 1 / (1 + np.exp(-x))
    print("run py kernel y", y)
    return y

def forward(args):
    print("args", args)
    return sigmoid(*args)

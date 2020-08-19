import numpy as np

def forward(x):
    print("run py kernel x", x)
    y = 1 / (1 + np.exp(-x))
    print("run py kernel y", y)
    return y

import numpy as np


def forward(x):
    print("py kernel sigmoid numpy: ")
    print(x)
    return 1 / (1 + np.exp(-x))

import numpy as np

def forward(x):
    print("run py kernel")
    return 1 / (1 + np.exp(-x))

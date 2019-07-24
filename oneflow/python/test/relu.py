import numpy as np
import oneflow as flow

def relu(x, alpha=0., max_value=None, thrshold=0.)
    if max_value is not None:
        x = np.where(x>max_value, max_value, x)
    x[x<thrshold] = alpha * (x - thrshold)
    return x

        

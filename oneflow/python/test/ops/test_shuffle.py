import oneflow as flow
import numpy as np
import time

SHAPE = (5, 2)
@flow.function(flow.FunctionConfig())
def TestJob(x = flow.FixedTensorDef(SHAPE, dtype=flow.float32)): 
    return flow.random.shuffle(x) 

def test1():
    x = np.random.randn(*SHAPE).astype(np.float32)
    print(x)
    ret = TestJob(x).get()
    print(ret.ndarray())

if __name__ == '__main__': 
    for i in range(1):
        test1()

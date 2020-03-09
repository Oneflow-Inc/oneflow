import oneflow as flow
import numpy as np
import time

SHAPE = (100000,)
@flow.function(flow.FunctionConfig())
def TestJob(x = flow.FixedTensorDef(SHAPE, dtype=flow.float32)):
    return flow.random.shuffle(x) 

def test1():
    r = TestJob(np.random.randn(*SHAPE).astype(np.float32)).get().ndarray()
    # arange = np.arange(start=0, stop=SHAPE[0], step=1)
    print(r)
    # assert np.array_equal(
    #     arange, np.sort(r, axis=0)
    # )

if __name__ == '__main__': 
    for i in range(10):
        test1()

import numpy as np
import oneflow as flow

DIM_0_SIZE = 100
@flow.function(flow.FunctionConfig())
def job(x=flow.FixedTensorDef((DIM_0_SIZE, 1)), size=flow.FixedTensorDef((1,), dtype=flow.int32)):
    return flow.sync_dynamic_resize(x, size)

def test1():
    size = np.random.random_integers(0, DIM_0_SIZE)
    x = np.random.rand(DIM_0_SIZE, 1).astype(np.float32)
    y = job(x, np.array([size]).astype(np.int32)).get().ndarray_list()[0]
    assert(np.array_equal(y, x[:size]))

if __name__ == '__main__': 
    for i in range(10):
        test1()

import oneflow as flow
import numpy as np

def test_constant(test_case):
    # TODO(lixinqi)
    with flow.device("gpu:0"):
        x = flow.constant(0, shape=(10,), dtype=flow.float)
        test_case.assertTrue(np.array_equal(x.numpy(), np.zeros((10,), dtype=np.float32)))


import unittest
from collections import OrderedDict

import numpy as np

import oneflow as flow
import oneflow.unittest

def _test_reshape(test_case, device):
    x = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    ).astype(np.float32)
    input = flow.tensor(x, dtype=flow.float32, device=flow.device(device))
    class RG(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self, inp):
            of_shape = flow.reshape(inp, shape=[2, 2, 2, -1])
            return of_shape
    g = RG()
    of_shape = g(input)
    np_shape = (2, 2, 2, 2)
    test_case.assertTrue(np.array_equal(of_shape.shape, flow.Size(np_shape)))

@flow.unittest.skip_unless_1n1d()
class TestModule(flow.unittest.TestCase):
    def test_reshape(test_case):
        _test_reshape(test_case, "cuda")

if __name__ == "__main__":
    unittest.main()

import oneflow as flow
import unittest
import numpy as np
from oneflow.fx import symbolic_trace

def sort_op(x):
    return flow.sort(x)

def ones_like_op(x):
    return flow.ones_like(x)

@flow.unittest.skip_unless_1n1d()
class TestFX(flow.unittest.TestCase):
    def test_sort_op(test_case):
        gm : flow.fx.GraphModule = symbolic_trace(sort_op)
        print(gm.graph)
        input = flow.randn(3, 4)
        assert(np.allclose(gm(input)[0].numpy(), sort_op(input)[0].numpy(), equal_nan=True))

    def test_ones_like_op(test_case):
        gm : flow.fx.GraphModule = symbolic_trace(ones_like_op)
        input = flow.randn(3, 4)
        assert(np.allclose(gm(input).numpy(), ones_like_op(input).numpy(), equal_nan=True))

if __name__ == "__main__":
    unittest.main()
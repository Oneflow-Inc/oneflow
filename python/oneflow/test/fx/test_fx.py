import oneflow as flow
import unittest
import numpy as np
from oneflow.fx import symbolic_trace

@flow.unittest.skip_unless_1n1d()
class TestFX(flow.unittest.TestCase):
    def test_abs(test_case):
        class MyAbs(flow.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return flow.abs(x)
        m = MyAbs()
        gm : flow.fx.GraphModule = symbolic_trace(m)
        input = flow.randn(3, 4)
        assert(np.allclose(gm(input).numpy(), m(input).numpy()))

if __name__ == "__main__":
    unittest.main()

import oneflow as flow
import unittest
import numpy as np
from oneflow.fx import symbolic_trace

def abs_func(x):
    return flow.abs(x)

def exp_func(x):
    return flow.exp(x)

def acos_func(x):
    return flow.acos(x)

def atanh_func(x):
    return flow.atanh(x)

def sign_func(x):
    return flow.sign(x)
class ConvModule(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_mod = flow.nn.Conv2d(5, 5, 3)

    def forward(self, x):
        return self.conv_mod(x)

@flow.unittest.skip_unless_1n1d()
class TestFX(flow.unittest.TestCase):
    def test_abs(test_case):
        gm : flow.fx.GraphModule = symbolic_trace(abs_func)
        input = flow.randn(3, 4)
        assert(np.allclose(gm(input).numpy(), abs_func(input).numpy(), equal_nan=True))

    def test_exp(test_case):
        gm : flow.fx.GraphModule = symbolic_trace(exp_func)
        input = flow.randn(3, 4)
        assert(np.allclose(gm(input).numpy(), exp_func(input).numpy(), equal_nan=True))
    
    def test_acos(test_case):
        gm : flow.fx.GraphModule = symbolic_trace(acos_func)
        input = flow.randn(3, 4)
        assert(np.allclose(gm(input).numpy(), acos_func(input).numpy(), equal_nan=True))

    def test_atanh(test_case):
        gm : flow.fx.GraphModule = symbolic_trace(atanh_func)
        input = flow.randn(3, 4)
        assert(np.allclose(gm(input).numpy(), atanh_func(input).numpy(), equal_nan=True))
    
    def test_sign(test_case):
        gm : flow.fx.GraphModule = symbolic_trace(sign_func)
        input = flow.randn(3, 4)
        assert(np.allclose(gm(input).numpy(), sign_func(input).numpy(), equal_nan=True))

    def test_conv(test_case):
        m = ConvModule()
        gm : flow.fx.GraphModule = symbolic_trace(m)
        input = flow.randn(1, 5, 5, 5)
        assert(np.allclose(gm(input).numpy(), m(input).numpy(), equal_nan=True))

if __name__ == "__main__":
    unittest.main()

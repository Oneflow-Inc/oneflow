import unittest
from collections import OrderedDict
import random as rd
import numpy as np
import oneflow as flow
from test_util import GenArgList
from automated_test_util import *

def _numpy_fmod(x, y):
    sign = np.sign(x)
    res = np.fmod(np.abs(x), np.abs(y))
    return sign * res

def _numpy_fmod_grad(x):
    grad = np.ones_like(x)
    return grad

def _test_fmod_same_shape_tensor(test_case, shape, device):
    input = flow.Tensor(np.random.uniform(-100, 100, shape), dtype=flow.float32, device=flow.device(device), requires_grad=True)
    other = flow.Tensor(np.random.uniform(-10, 10, shape), dtype=flow.float32, device=flow.device(device))
    of_out = flow.fmod(input, other)
    np_out = _numpy_fmod(input.numpy(), other.numpy())
    of_out.sum().backward()
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    test_case.assertTrue(np.allclose(input.grad.numpy(), _numpy_fmod_grad(input.numpy()), 1e-05, 1e-05))

def _test_fmod_tensor_vs_scalar(test_case, shape, device):
    input = flow.Tensor(np.random.randint(-100, 100, shape), dtype=flow.float32, device=flow.device(device), requires_grad=True)
    other = rd.uniform(-1, 1) * 100
    of_out = flow.fmod(input, other)
    np_out = _numpy_fmod(input.numpy(), other)
    of_out.sum().backward()
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    test_case.assertTrue(np.allclose(input.grad.numpy(), _numpy_fmod_grad(input.numpy()), 1e-05, 1e-05))

class TestFmodModule(flow.unittest.TestCase):

    def test_fmod(test_case):
        arg_dict = OrderedDict()
        arg_dict['fun'] = [_test_fmod_same_shape_tensor, _test_fmod_tensor_vs_scalar]
        arg_dict['shape'] = [(2,), (2, 3), (2, 4, 5, 6)]
        arg_dict['device'] = ['cpu', 'cuda']
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest
    def test_flow_fmod_with_random_data(test_case):
        device = random_device()
        input = random_pytorch_tensor().to(device)
        other = random_pytorch_tensor().to(device)
        return torch.fmod(input, other)
if __name__ == '__main__':
    unittest.main()
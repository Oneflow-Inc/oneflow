import unittest
from collections import OrderedDict
import numpy as np
import oneflow as flow
from test_util import GenArgList
from automated_test_util import *

def _test_mul_impl(test_case, device):
    x = flow.Tensor(np.random.randn(2, 3), device=flow.device(device), requires_grad=True)
    y = flow.Tensor(np.random.randn(2, 3), device=flow.device(device), requires_grad=True)
    of_out = flow.mul(x, y)
    np_out = np.multiply(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    np_grad_x = y.numpy()
    np_grad_y = x.numpy()
    test_case.assertTrue(np.allclose(x.grad.numpy(), np_grad_x, 1e-05, 1e-05))
    test_case.assertTrue(np.allclose(y.grad.numpy(), np_grad_y, 1e-05, 1e-05))
    x = 5
    y = flow.Tensor(np.random.randn(2, 3), device=flow.device(device))
    of_out = flow.mul(x, y)
    np_out = np.multiply(x, y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    x = flow.Tensor(np.random.randn(2, 3), device=flow.device(device))
    y = 5
    of_out = flow.mul(x, y)
    np_out = np.multiply(x.numpy(), y)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    x = flow.Tensor(np.random.randn(1, 1), device=flow.device(device), requires_grad=True)
    y = flow.Tensor(np.random.randn(2, 3), device=flow.device(device), requires_grad=True)
    of_out = flow.mul(x, y)
    np_out = np.multiply(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(np.allclose(x.grad.numpy(), np.sum(y.numpy()), 1e-05, 1e-05))
    test_case.assertTrue(np.allclose(y.grad.numpy(), x.numpy(), 1e-05, 1e-05))
    x = flow.Tensor(np.random.randn(1, 1), device=flow.device(device), requires_grad=True)
    y = flow.Tensor(np.random.randn(2, 3, 4), device=flow.device(device), requires_grad=True)
    of_out = flow.mul(x, y)
    np_out = np.multiply(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(np.allclose(x.grad.numpy(), np.sum(y.numpy()), 1e-05, 1e-05))
    test_case.assertTrue(np.allclose(y.grad.numpy(), x.numpy(), 1e-05, 1e-05))
    x = flow.Tensor(np.random.randn(1, 1), device=flow.device(device), requires_grad=True)
    y = flow.Tensor(np.random.randn(2, 3, 4, 5), device=flow.device(device), requires_grad=True)
    of_out = flow.mul(x, y)
    np_out = np.multiply(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(np.allclose(x.grad.numpy(), np.sum(y.numpy()), 1e-05, 1e-05))
    test_case.assertTrue(np.allclose(y.grad.numpy(), x.numpy(), 1e-05, 1e-05))

@flow.unittest.skip_unless_1n1d()
class TestMulModule(flow.unittest.TestCase):

    def test_mul(test_case):
        arg_dict = OrderedDict()
        arg_dict['test_fun'] = [_test_mul_impl]
        arg_dict['device'] = ['cpu', 'cuda']
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    def test_mul_against_pytorch(test_case):
        arg_dict = OrderedDict()
        arg_dict['test_type'] = [test_flow_against_pytorch, test_tensor_against_pytorch]
        arg_dict['device'] = ['cpu', 'cuda']
        arg_dict['op'] = ['mul']
        for arg in GenArgList(arg_dict):
            arg[0](test_case, arg[2], extra_annotations={'other': flow.Tensor}, extra_generators={'input': random_tensor(ndim=2, dim0=2, dim1=3), 'other': random_tensor(ndim=2, dim0=2, dim1=3)}, device=arg[1])
            arg[0](test_case, arg[2], extra_annotations={'other': float}, extra_generators={'input': random_tensor(ndim=2, dim0=2, dim1=3), 'other': random(0, 5)}, device=arg[1])
if __name__ == '__main__':
    unittest.main()
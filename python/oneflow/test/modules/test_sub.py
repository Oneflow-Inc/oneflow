import unittest
from collections import OrderedDict
import numpy as np
import oneflow as flow
from test_util import GenArgList
from automated_test_util import *

def _test_sub_impl(test_case, shape, device):
    x = flow.Tensor(np.random.randn(*shape), device=flow.device(device), requires_grad=True)
    y = flow.Tensor(np.random.randn(*shape), device=flow.device(device), requires_grad=True)
    of_out = flow.sub(x, y)
    np_out = np.subtract(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    np_grad_x = np.ones(shape)
    np_grad_y = -np.ones(shape)
    test_case.assertTrue(np.allclose(x.grad.numpy(), np_grad_x, 1e-05, 1e-05))
    test_case.assertTrue(np.allclose(y.grad.numpy(), np_grad_y, 1e-05, 1e-05))
    x = 5
    y = flow.Tensor(np.random.randn(*shape), device=flow.device(device))
    of_out = flow.sub(x, y)
    np_out = np.subtract(x, y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    x = flow.Tensor(np.random.randn(*shape), device=flow.device(device))
    y = 5
    of_out = flow.sub(x, y)
    np_out = np.subtract(x.numpy(), y)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    x = flow.Tensor(np.random.randn(*shape), device=flow.device(device))
    y = flow.Tensor(np.random.randn(1, 1), device=flow.device(device))
    of_out = flow.sub(x, y)
    np_out = np.subtract(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    x = flow.Tensor(np.array([5.0]))
    y = flow.Tensor(np.random.randn(1, 1))
    of_out = flow.sub(x, y)
    np_out = np.subtract(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    x = flow.Tensor(np.random.randn(1, 1), requires_grad=True)
    y = flow.Tensor(np.array([5.0]), requires_grad=True)
    of_out = flow.sub(x, y)
    np_out = np.subtract(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    np_grad_x = np.ones((1, 1))
    np_grad_y = -np.ones(1)
    test_case.assertTrue(np.allclose(x.grad.numpy(), np_grad_x, 1e-05, 1e-05))
    test_case.assertTrue(np.allclose(y.grad.numpy(), np_grad_y, 1e-05, 1e-05))

@flow.unittest.skip_unless_1n1d()
class TestSubModule(flow.unittest.TestCase):

    def test_sub(test_case):
        arg_dict = OrderedDict()
        arg_dict['shape'] = [(2, 3), (2, 3, 4), (2, 4, 5, 6)]
        arg_dict['device'] = ['cpu', 'cuda']
        for arg in GenArgList(arg_dict):
            _test_sub_impl(test_case, *arg)

    def test_sub_against_pytorch(test_case):
        arg_dict = OrderedDict()
        arg_dict['test_type'] = [test_flow_against_pytorch, test_tensor_against_pytorch]
        arg_dict['device'] = ['cpu', 'cuda']
        arg_dict['op'] = ['sub']
        for arg in GenArgList(arg_dict):
            arg[0](test_case, arg[2], extra_annotations={'other': flow.Tensor}, extra_generators={'input': random_tensor(ndim=2, dim0=2, dim1=3), 'other': random_tensor(ndim=2, dim0=2, dim1=3)}, device=arg[1])
            arg[0](test_case, arg[2], extra_annotations={'other': float}, extra_generators={'input': random_tensor(ndim=2, dim0=2, dim1=3), 'other': random(0, 5)}, device=arg[1])
if __name__ == '__main__':
    unittest.main()
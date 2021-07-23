import unittest
from collections import OrderedDict
import numpy as np
import oneflow as flow
from test_util import GenArgList
from automated_test_util import *

def _test_squeeze(test_case, device):
    np_arr = np.random.rand(1, 1, 1, 3)
    input = flow.Tensor(np_arr, device=flow.device(device))
    of_shape = flow.squeeze(input, dim=[1, 2]).numpy().shape
    np_shape = (1, 3)
    test_case.assertTrue(np.array_equal(of_shape, np_shape))
    test_case.assertTrue(np.allclose(flow.squeeze(input, dim=[1, 2]).numpy(), np.squeeze(input.numpy(), axis=(1, 2)), 0.0001, 0.0001))

def _test_squeeze_1d_input(test_case, device):
    np_arr = np.random.rand(10)
    input = flow.Tensor(np_arr, device=flow.device(device))
    output = flow.squeeze(input)
    test_case.assertTrue(np.allclose(output.numpy(), np_arr, 1e-05, 1e-05))

def _test_tensor_squeeze(test_case, device):
    np_arr = np.random.rand(1, 1, 1, 3)
    input = flow.Tensor(np_arr, device=flow.device(device))
    of_shape = input.squeeze(dim=[1, 2]).numpy().shape
    np_shape = (1, 3)
    test_case.assertTrue(np.array_equal(of_shape, np_shape))
    test_case.assertTrue(np.allclose(input.squeeze(dim=[1, 2]).numpy(), np.squeeze(input.numpy(), axis=(1, 2)), 0.0001, 0.0001))

def _test_squeeze_int(test_case, device):
    np_arr = np.random.rand(1, 1, 1, 3)
    input = flow.Tensor(np_arr, device=flow.device(device))
    of_shape = flow.squeeze(input, 1).numpy().shape
    np_shape = (1, 1, 3)
    test_case.assertTrue(np.array_equal(of_shape, np_shape))
    test_case.assertTrue(np.allclose(input.squeeze(1).numpy(), np.squeeze(input.numpy(), axis=1), 0.0001, 0.0001))

def _test_squeeze_backward(test_case, device):
    np_arr = np.random.rand(1, 1, 1, 3)
    input = flow.Tensor(np_arr, device=flow.device(device), requires_grad=True)
    y = flow.squeeze(input, dim=1).sum()
    y.backward()
    np_grad = np.ones((1, 1, 1, 3))
    test_case.assertTrue(np.array_equal(input.grad.numpy(), np_grad))

@flow.unittest.skip_unless_1n1d()
class TestSqueeze(flow.unittest.TestCase):

    def test_squeeze(test_case):
        arg_dict = OrderedDict()
        arg_dict['test_fun'] = [_test_squeeze, _test_squeeze_1d_input, _test_squeeze_int, _test_tensor_squeeze, _test_squeeze_backward]
        arg_dict['device'] = ['cpu', 'cuda']
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    def test_flow_squeeze_with_random_data(test_case):
        for device in ['cpu', 'cuda']:
            test_flow_against_pytorch(test_case, 'squeeze', extra_annotations={'dim': int}, extra_generators={'dim': random(0, 6)}, device=device)

    def test_flow_tensor_squeeze_with_random_data(test_case):
        for device in ['cpu', 'cuda']:
            test_tensor_against_pytorch(test_case, 'squeeze', extra_annotations={'dim': int}, extra_generators={'dim': random(0, 6)}, device=device)
if __name__ == '__main__':
    unittest.main()
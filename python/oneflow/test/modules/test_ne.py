import unittest
from collections import OrderedDict
import numpy as np
import oneflow as flow
from test_util import GenArgList

def _test_ne(test_case, shape, device):
    arr1 = np.random.randn(*shape)
    arr2 = np.random.randn(*shape)
    input = flow.Tensor(arr1, dtype=flow.float32, device=flow.device(device))
    other = flow.Tensor(arr2, dtype=flow.float32, device=flow.device(device))
    of_out = flow.ne(input, other)
    of_out2 = flow.not_equal(input, other)
    np_out = np.not_equal(arr1, arr2)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))
    test_case.assertTrue(np.array_equal(of_out2.numpy(), np_out))

def _test_tensor_ne_operator(test_case, shape, device):
    arr1 = np.random.randn(*shape)
    arr2 = np.random.randn(*shape)
    input = flow.Tensor(arr1, dtype=flow.float32, device=flow.device(device))
    other = flow.Tensor(arr2, dtype=flow.float32, device=flow.device(device))
    of_out = input.ne(other)
    np_out = np.not_equal(arr1, arr2)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))

def _test_ne_int(test_case, shape, device):
    arr = np.random.randn(*shape)
    input = flow.Tensor(arr, dtype=flow.float32, device=flow.device(device))
    num = 1
    of_out = flow.ne(input, num)
    np_out = np.not_equal(arr, num)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))

def _test_tensor_ne_operator_int(test_case, shape, device):
    arr = np.random.randn(*shape)
    input = flow.Tensor(arr, dtype=flow.float32, device=flow.device(device))
    num = 1
    of_out = input.ne(num)
    np_out = np.not_equal(arr, num)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))

def _test_ne_float(test_case, shape, device):
    arr = np.random.randn(*shape)
    input = flow.Tensor(arr, dtype=flow.float32, device=flow.device(device))
    num = 1.0
    of_out = flow.ne(input, num)
    np_out = np.not_equal(arr, num)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))

def _test_tensor_ne_operator_float(test_case, shape, device):
    arr = np.random.randn(*shape)
    input = flow.Tensor(arr, dtype=flow.float32, device=flow.device(device))
    num = 1.0
    of_out = input.ne(num)
    np_out = np.not_equal(arr, num)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))

@flow.unittest.skip_unless_1n1d()
class TestNe(flow.unittest.TestCase):

    def test_ne(test_case):
        arg_dict = OrderedDict()
        arg_dict['test_func'] = [_test_ne, _test_tensor_ne_operator, _test_ne_int, _test_tensor_ne_operator_int, _test_ne_float, _test_tensor_ne_operator_float]
        arg_dict['shape'] = [(2, 3), (2, 3, 4), (2, 4, 5, 6)]
        arg_dict['device'] = ['cpu', 'cuda']
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])
if __name__ == '__main__':
    unittest.main()
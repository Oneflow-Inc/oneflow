import unittest
from collections import OrderedDict
import numpy as np
import oneflow as flow
from test_util import GenArgList

def np_repeat(x, sizes):
    return np.tile(x, sizes)

def _test_repeat_new_dim(test_case, device):
    input = flow.Tensor(np.random.randn(2, 4, 1, 3), dtype=flow.float32, device=flow.device(device))
    sizes = (4, 3, 2, 3, 3)
    np_out = np_repeat(input.numpy(), sizes)
    of_out = input.repeat(sizes=sizes)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))

def _test_repeat_same_dim(test_case, device):
    input = flow.Tensor(np.random.randn(1, 2, 5, 3), dtype=flow.float32, device=flow.device(device))
    sizes = (4, 2, 3, 19)
    of_out = input.repeat(sizes=sizes)
    np_out = np_repeat(input.numpy(), sizes)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))

def _test_repeat_same_dim_int(test_case, device):
    input = flow.Tensor(np.random.randn(1, 2, 5, 3), dtype=flow.int32, device=flow.device(device))
    size_tensor = flow.Tensor(np.random.randn(4, 2, 3, 19))
    sizes = size_tensor.size()
    of_out = input.repeat(sizes=sizes)
    np_out = np_repeat(input.numpy(), sizes)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out.astype(np.int32)))

def _test_repeat_same_dim_int8(test_case, device):
    input = flow.Tensor(np.random.randn(1, 2, 5, 3), dtype=flow.int8, device=flow.device(device))
    size_tensor = flow.Tensor(np.random.randn(4, 2, 3, 19))
    sizes = size_tensor.size()
    of_out = input.repeat(sizes=sizes)
    np_out = np_repeat(input.numpy(), sizes)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out.astype(np.int32)))

def _test_repeat_new_dim_backward(test_case, device):
    input = flow.Tensor(np.random.randn(2, 4, 1, 3), dtype=flow.float32, device=flow.device(device), requires_grad=True)
    sizes = (4, 3, 2, 3, 3)
    of_out = input.repeat(sizes=sizes)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[[216.0, 216.0, 216.0]], [[216.0, 216.0, 216.0]], [[216.0, 216.0, 216.0]], [[216.0, 216.0, 216.0]]], [[[216.0, 216.0, 216.0]], [[216.0, 216.0, 216.0]], [[216.0, 216.0, 216.0]], [[216.0, 216.0, 216.0]]]]
    test_case.assertTrue(np.array_equal(input.grad.numpy(), np_grad))

def _test_repeat_same_dim_backward(test_case, device):
    input = flow.Tensor(np.random.randn(1, 2, 5, 3), dtype=flow.float32, device=flow.device(device), requires_grad=True)
    sizes = (1, 2, 3, 1)
    of_out = input.repeat(sizes=sizes)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[[6.0, 6.0, 6.0], [6.0, 6.0, 6.0], [6.0, 6.0, 6.0], [6.0, 6.0, 6.0], [6.0, 6.0, 6.0]], [[6.0, 6.0, 6.0], [6.0, 6.0, 6.0], [6.0, 6.0, 6.0], [6.0, 6.0, 6.0], [6.0, 6.0, 6.0]]]]
    test_case.assertTrue(np.array_equal(input.grad.numpy(), np_grad))

@flow.unittest.skip_unless_1n1d()
class TestRepeat(flow.unittest.TestCase):

    def test_repeat(test_case):
        arg_dict = OrderedDict()
        arg_dict['test_fun'] = [_test_repeat_new_dim, _test_repeat_same_dim, _test_repeat_same_dim_int, _test_repeat_same_dim_int8, _test_repeat_new_dim_backward, _test_repeat_same_dim_backward]
        arg_dict['device'] = ['cpu', 'cuda']
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])
if __name__ == '__main__':
    unittest.main()
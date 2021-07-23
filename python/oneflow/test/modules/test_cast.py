import unittest
from collections import OrderedDict
import numpy as np
import oneflow as flow
from test_util import GenArgList

def _test_cast_float2int(test_case, device, shape):
    np_arr = np.random.randn(*shape).astype(np.float32)
    input = flow.Tensor(np_arr, dtype=flow.float32, device=flow.device(device))
    output = flow.cast(input, flow.int8)
    np_out = np_arr.astype(np.int8)
    test_case.assertTrue(np.array_equal(output.numpy(), np_out))

def _test_cast_int2float(test_case, device, shape):
    np_arr = np.random.randn(*shape).astype(np.int8)
    input = flow.Tensor(np_arr, dtype=flow.int8, device=flow.device(device))
    output = flow.cast(input, flow.float32)
    np_out = np_arr.astype(np.float32)
    test_case.assertTrue(np.array_equal(output.numpy(), np_out))

def _test_cast_backward(test_case, device, shape):
    np_arr = np.random.randn(*shape).astype(np.float32)
    x = flow.Tensor(np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True)
    y = flow.cast(x, flow.int8)
    z = y.sum()
    z.backward()
    np_out = np_arr.astype(np.int8)
    test_case.assertTrue(np.array_equal(x.grad.numpy(), np.ones(shape=shape)))

@flow.unittest.skip_unless_1n1d()
class TestCast(flow.unittest.TestCase):

    def test_cast(test_case):
        arg_dict = OrderedDict()
        arg_dict['test_fun'] = [_test_cast_float2int, _test_cast_int2float, _test_cast_backward]
        arg_dict['device'] = ['cpu', 'cuda']
        arg_dict['shape'] = [(2, 3), (2, 3, 4), (2, 3, 4, 5)]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])
if __name__ == '__main__':
    unittest.main()
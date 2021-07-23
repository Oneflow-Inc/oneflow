import unittest
from collections import OrderedDict
import numpy as np
import oneflow as flow
from test_util import GenArgList

def _test_acos_impl(test_case, shape, device):
    input = flow.Tensor(np.random.rand(*shape) - 0.5, device=flow.device(device), requires_grad=True)
    of_out = flow.acos(input)
    np_out = np.arccos(input.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05, equal_nan=True))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = -1.0 / np.sqrt(1 - np.square(input.numpy()))
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 0.0001, 0.0001, equal_nan=True))

@flow.unittest.skip_unless_1n1d()
class TestAcos(flow.unittest.TestCase):

    def test_acos(test_case):
        arg_dict = OrderedDict()
        arg_dict['shape'] = [(2,), (2, 3), (2, 3, 4), (2, 4, 5, 6)]
        arg_dict['device'] = ['cpu', 'cuda']
        for arg in GenArgList(arg_dict):
            _test_acos_impl(test_case, *arg)
if __name__ == '__main__':
    unittest.main()
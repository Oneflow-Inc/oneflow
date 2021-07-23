import unittest
from collections import OrderedDict
import numpy as np
import oneflow as flow
from test_util import GenArgList

def _test_exp_impl(test_case, shape, device):
    np_input = np.random.randn(*shape)
    of_input = flow.Tensor(np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True)
    of_out = flow.exp(of_input)
    np_out = np.exp(np_input)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(np.allclose(of_input.grad.numpy(), np_out, 0.0001, 0.0001))

@flow.unittest.skip_unless_1n1d()
class TestExp(flow.unittest.TestCase):

    def test_exp(test_case):
        arg_dict = OrderedDict()
        arg_dict['shape'] = [(2, 3), (2, 3, 4), (2, 4, 5, 6)]
        arg_dict['device'] = ['cpu', 'cuda']
        for arg in GenArgList(arg_dict):
            _test_exp_impl(test_case, *arg)
if __name__ == '__main__':
    unittest.main()
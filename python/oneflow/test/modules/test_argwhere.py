import unittest
from collections import OrderedDict
import numpy as np
import oneflow as flow
from test_util import GenArgList

def _test_argwhere(test_case, shape, device):
    np_input = np.random.randn(*shape)
    input = flow.Tensor(np_input, device=flow.device(device))
    of_out = flow.argwhere(input)
    np_out = np.argwhere(np_input)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
    test_case.assertTrue(np.array_equal(of_out.numpy().shape, np_out.shape))

@flow.unittest.skip_unless_1n1d()
class TestArgwhere(flow.unittest.TestCase):

    def test_argwhere(test_case):
        arg_dict = OrderedDict()
        arg_dict['test_fun'] = [_test_argwhere]
        arg_dict['shape'] = [(2, 3), (2, 3, 4), (2, 4, 5, 6)]
        arg_dict['device'] = ['cpu', 'cuda']
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])
if __name__ == '__main__':
    unittest.main()
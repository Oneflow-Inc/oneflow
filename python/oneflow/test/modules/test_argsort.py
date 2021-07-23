import unittest
from collections import OrderedDict
import numpy as np
import oneflow as flow
from test_util import GenArgList, type_name_to_flow_type

def _test_argsort(test_case, data_shape, axis, descending, data_type, device):
    input = flow.Tensor(np.random.randn(*data_shape), dtype=type_name_to_flow_type[data_type], device=flow.device(device))
    of_out = flow.argsort(input, dim=axis, descending=descending)
    np_input = -input.numpy() if descending else input.numpy()
    np_out = np.argsort(np_input, axis=axis)
    test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))

def _test_tensor_argsort(test_case, data_shape, axis, descending, data_type, device):
    input = flow.Tensor(np.random.randn(*data_shape), dtype=type_name_to_flow_type[data_type], device=flow.device(device))
    of_out = input.argsort(dim=axis, descending=descending)
    np_input = -input.numpy() if descending else input.numpy()
    np_out = np.argsort(np_input, axis=axis)
    test_case.assertTrue(np.array_equal(of_out.numpy().shape, np_out.shape))
    test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))

@flow.unittest.skip_unless_1n1d()
class TestArgsort(flow.unittest.TestCase):

    def test_argsort(test_case):
        arg_dict = OrderedDict()
        arg_dict['test_fun'] = [_test_argsort, _test_tensor_argsort]
        arg_dict['data_shape'] = [(2, 6, 5, 4), (3, 4, 8)]
        arg_dict['axis'] = [-1, 0, 2]
        arg_dict['descending'] = [True, False]
        arg_dict['data_type'] = ['double', 'float32', 'int32']
        arg_dict['device'] = ['cpu', 'cuda']
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])
if __name__ == '__main__':
    unittest.main()
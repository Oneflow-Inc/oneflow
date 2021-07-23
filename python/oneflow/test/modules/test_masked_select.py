import unittest
from collections import OrderedDict
import numpy as np
import oneflow as flow
from test_util import GenArgList

def _test_masked_select(test_case, device):
    x = flow.Tensor(np.array([[-0.462, 0.3139], [0.3898, -0.7197], [0.0478, -0.1657]]), dtype=flow.float32, device=flow.device(device), requires_grad=True)
    mask = x.gt(0.05)
    of_out = flow.masked_select(x, mask)
    np_out = np.array([0.3139, 0.3898])
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = np.array([[0, 1], [1, 0], [0, 0]])
    test_case.assertTrue(np.allclose(x.grad.numpy(), np_grad, 1e-05, 1e-05))

def _test_masked_select_broadcast(test_case, device):
    x = flow.Tensor(np.array([[[-0.462, 0.3139], [0.3898, -0.7197], [0.0478, -0.1657]]]), dtype=flow.float32, device=flow.device(device), requires_grad=True)
    mask = flow.Tensor(np.array([[[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], [[1.0, 0], [1.0, 1.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, 1.0], [1.0, 1.0]]]), dtype=flow.int8, device=flow.device(device))
    of_out = flow.masked_select(x, mask)
    np_out = [-0.462, 0.3898, -0.7197, -0.1657, -0.462, 0.3898, -0.7197, -0.1657, -0.462, 0.3139, -0.7197, 0.0478, -0.1657]
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [[[3.0, 1.0], [2.0, 3.0], [1.0, 3.0]]]
    test_case.assertTrue(np.allclose(x.grad.numpy(), np_grad, 1e-05, 1e-05))

@flow.unittest.skip_unless_1n1d()
class TestAbs(flow.unittest.TestCase):

    def test_cosh(test_case):
        arg_dict = OrderedDict()
        arg_dict['test_fun'] = [_test_masked_select, _test_masked_select_broadcast]
        arg_dict['device'] = ['cpu', 'cuda']
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])
if __name__ == '__main__':
    unittest.main()
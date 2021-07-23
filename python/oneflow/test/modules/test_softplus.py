import unittest
from collections import OrderedDict
import numpy as np
from test_util import GenArgList
import oneflow as flow


def _test_softplus_impl(test_case, shape, device):
    np_input = np.random.randn(*shape)
    of_input = flow.Tensor(
        np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    np_x_grad = np.exp(np_input) / (1 + np.exp(np_input))
    of_out = flow.softplus(of_input)
    np_out = np.log(1 + np.exp(np_input))
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(np.allclose(of_input.grad.numpy(), np_x_grad, 0.0001, 0.0001))


@flow.unittest.skip_unless_1n1d()
class Testsoftplus(flow.unittest.TestCase):
    def test_softplus(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 4, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_softplus_impl(test_case, *arg)


if __name__ == "__main__":
    unittest.main()

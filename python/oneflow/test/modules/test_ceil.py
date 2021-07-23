import unittest
from collections import OrderedDict

import numpy as np
from test_util import GenArgList

import oneflow as flow


def _test_ceil_impl(test_case, device, shape):
    x = flow.Tensor(
        np.random.randn(*shape), device=flow.device(device), requires_grad=True
    )
    of_out = flow.ceil(x)
    np_out = np.ceil(x.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(np.allclose(x.grad.numpy(), np.zeros(shape), 0.0001, 0.0001))


@flow.unittest.skip_unless_1n1d()
class TestCeilModule(flow.unittest.TestCase):
    def test_ceil(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_ceil_impl]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["shape"] = [(1,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()

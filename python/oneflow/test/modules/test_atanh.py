import unittest
from collections import OrderedDict
import numpy as np
import oneflow as flow
from test_util import GenArgList


def _test_atanh_impl(test_case, shape, device):
    np_input = np.random.random(shape) - 0.5
    of_input = flow.Tensor(
        np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = flow.atanh(of_input)
    np_out = np.arctanh(np_input)
    test_case.assertTrue(
        np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001, equal_nan=True)
    )
    of_out = of_out.sum()
    of_out.backward()
    np_out_grad = 1.0 / (1.0 - np.square(np_input))
    test_case.assertTrue(
        np.allclose(of_input.grad.numpy(), np_out_grad, 0.0001, 0.0001, equal_nan=True)
    )


def _test_arctanh_impl(test_case, shape, device):
    np_input = np.random.random(shape) - 0.5
    of_input = flow.Tensor(
        np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = flow.arctanh(of_input)
    np_out = np.arctanh(np_input)
    test_case.assertTrue(
        np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001, equal_nan=True)
    )
    of_out = of_out.sum()
    of_out.backward()
    np_out_grad = 1.0 / (1.0 - np.square(np_input))
    test_case.assertTrue(
        np.allclose(of_input.grad.numpy(), np_out_grad, 0.0001, 0.0001, equal_nan=True)
    )


@flow.unittest.skip_unless_1n1d()
class TestAtanh(flow.unittest.TestCase):
    def test_atanh(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2,), (2, 3), (2, 3, 4), (2, 4, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_atanh_impl(test_case, *arg)
            _test_arctanh_impl(test_case, *arg)


if __name__ == "__main__":
    unittest.main()

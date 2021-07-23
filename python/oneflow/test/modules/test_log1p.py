import unittest
from collections import OrderedDict
import numpy as np
import oneflow as flow
from test_util import GenArgList


def _test_log1p(test_case, shape, device):
    input_arr = np.exp(np.random.randn(*shape)) - 1
    np_out = np.log1p(input_arr)
    x = flow.Tensor(
        input_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = flow.log1p(x)
    test_case.assertTrue(
        np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05, equal_nan=True)
    )
    of_out = of_out.sum()
    of_out.backward()
    np_out_grad = 1.0 / (1 + input_arr)
    test_case.assertTrue(
        np.allclose(x.grad.numpy(), np_out_grad, 0.0001, 0.0001, equal_nan=True)
    )


def _test_log1p_tensor_function(test_case, shape, device):
    input_arr = np.exp(np.random.randn(*shape)) - 1
    np_out = np.log1p(input_arr)
    x = flow.Tensor(
        input_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = x.log1p()
    test_case.assertTrue(
        np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05, equal_nan=True)
    )
    of_out = of_out.sum()
    of_out.backward()
    np_out_grad = 1.0 / (1 + input_arr)
    test_case.assertTrue(
        np.allclose(x.grad.numpy(), np_out_grad, 0.0001, 0.0001, equal_nan=True)
    )


@flow.unittest.skip_unless_1n1d()
class TestLog1p(flow.unittest.TestCase):
    def test_log1p(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_log1p, _test_log1p_tensor_function]
        arg_dict["shape"] = [(2,), (2, 3), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()

import unittest
from collections import OrderedDict
import numpy as np
import oneflow as flow
from test_util import GenArgList


def _test_floor(test_case, shape, device):
    np_input = np.random.randn(*shape)
    of_input = flow.Tensor(
        np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = flow.floor(of_input)
    np_out = np.floor(np_input)
    test_case.assertTrue(
        np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05, equal_nan=True)
    )
    of_out = of_out.sum()
    of_out.backward()
    np_out_grad = np.zeros_like(of_out, dtype=np.float32)
    test_case.assertTrue(
        np.allclose(of_input.grad.numpy(), np_out_grad, 0.0001, 0.0001, equal_nan=True)
    )


@flow.unittest.skip_unless_1n1d()
class TestFloor(flow.unittest.TestCase):
    def test_floor(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2,), (2, 3), (2, 4, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_floor(test_case, *arg)


if __name__ == "__main__":
    unittest.main()

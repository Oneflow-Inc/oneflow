import unittest
from collections import OrderedDict

import numpy as np
from automated_test_util import *
from test_util import GenArgList

import oneflow as flow


def _test_sum_impl(test_case, device):
    input = flow.Tensor(
        np.random.randn(2, 3), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.sum(input, dim=0)
    np_out = np.sum(input.numpy(), axis=0)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    input = flow.Tensor(
        np.random.randn(2, 3), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.sum(input, dim=0)
    np_out = np.sum(input.numpy(), axis=0)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    input = flow.Tensor(
        np.random.randn(2, 3), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.sum(input, dim=1)
    of_out2 = input.sum(dim=1)
    np_out = np.sum(input.numpy(), axis=1)
    test_case.assertTrue(np.allclose(of_out2.numpy(), of_out.numpy(), 1e-05, 1e-05))
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    input = flow.Tensor(
        np.random.randn(4, 5, 6),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    of_out = flow.sum(input, dim=(2, 1))
    np_out = np.sum(input.numpy(), axis=(2, 1))
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = np.ones((4, 5, 6))
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05))


@flow.unittest.skip_unless_1n1d()
class TestSumModule(flow.unittest.TestCase):
    def test_sum(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_sum_impl(test_case, *arg)

    def test_sum_against_pytorch(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_type"] = [test_flow_against_pytorch, test_tensor_against_pytorch]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, "sum", device=arg[1])


if __name__ == "__main__":
    unittest.main()

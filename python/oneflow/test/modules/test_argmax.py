import unittest
from collections import OrderedDict
import numpy as np
import oneflow as flow
from test_util import GenArgList


def _test_argmax_aixs_negative(test_case, device):
    input = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    axis = -1
    of_out = flow.argmax(input, dim=axis)
    np_out = np.argmax(input.numpy(), axis=axis)
    test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))


def _test_tensor_argmax(test_case, device):
    input = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    axis = 0
    of_out = input.argmax(dim=axis)
    np_out = np.argmax(input.numpy(), axis=axis)
    test_case.assertTrue(np.array_equal(of_out.numpy().shape, np_out.shape))
    test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))


def _test_argmax_axis_postive(test_case, device):
    input = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    axis = 1
    of_out = flow.argmax(input, dim=axis)
    np_out = np.argmax(input.numpy(), axis=axis)
    test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))


def _test_argmax_keepdims(test_case, device):
    input = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    axis = 0
    of_out = input.argmax(axis, True)
    np_out = np.argmax(input.numpy(), axis=axis)
    np_out = np.expand_dims(np_out, axis=axis)
    test_case.assertTrue(np.array_equal(of_out.numpy().shape, np_out.shape))
    test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))


def _test_argmax_dim_equal_none(test_case, device):
    input = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    of_out = input.argmax()
    np_out = np.argmax(input.numpy().flatten(), axis=0)
    test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))


@flow.unittest.skip_unless_1n1d()
class TestArgmax(flow.unittest.TestCase):
    def test_argmax(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_argmax_aixs_negative,
            _test_tensor_argmax,
            _test_argmax_axis_postive,
            _test_argmax_keepdims,
            _test_argmax_dim_equal_none,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()

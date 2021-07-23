import unittest
from collections import OrderedDict
import numpy as np
import oneflow as flow
from test_util import GenArgList


def _test_view(test_case, device):
    x = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    ).astype(np.float32)
    input = flow.Tensor(x, device=flow.device(device))
    of_shape = flow.view(input, shape=[2, 2, 2, -1]).numpy().shape
    of_shape = flow.view(input, shape=[2, 2, 2, -1]).numpy().shape
    np_shape = (2, 2, 2, 2)
    test_case.assertTrue(np.array_equal(of_shape, np_shape))


def _test_view_tuple(test_case, device):
    x = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    ).astype(np.float32)
    input = flow.Tensor(x, device=flow.device(device))
    of_shape = flow.view(input, shape=(2, 2, 2, -1)).numpy().shape
    np_shape = (2, 2, 2, 2)
    test_case.assertTrue(np.array_equal(of_shape, np_shape))


def _test_tensor_view(test_case, device):
    x = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    ).astype(np.float32)
    input = flow.Tensor(x, device=flow.device(device))
    of_shape = input.view(shape=[2, 2, 2, -1]).numpy().shape
    np_shape = (2, 2, 2, 2)
    test_case.assertTrue(np.array_equal(of_shape, np_shape))


def _test_view_backward(test_case, device):
    x = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    ).astype(np.float32)
    input = flow.Tensor(x, device=flow.device(device), requires_grad=True)
    of_out = flow.view(input, shape=[2, 2, 2, -1]).sum()
    of_out.backward()
    np_grad = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    test_case.assertTrue(np.allclose(np_grad, input.grad.numpy(), 0.0001, 0.0001))


@flow.unittest.skip_unless_1n1d()
class TestModule(flow.unittest.TestCase):
    def test_view(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_view,
            _test_view_tuple,
            _test_tensor_view,
            _test_view_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()

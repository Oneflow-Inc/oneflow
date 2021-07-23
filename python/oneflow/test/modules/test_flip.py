import unittest
from collections import OrderedDict

import numpy as np
from automated_test_util import *
from test_util import GenArgList

import oneflow as flow


def _test_flip(test_case, device):
    np_arr = np.arange(0, 16).reshape((2, 2, 2, 2)).astype(np.float32)
    input = flow.Tensor(np_arr, device=flow.device(device), requires_grad=True)
    out = flow.flip(input, [0, 1, 2])
    np_out = [
        [[[14.0, 15.0], [12.0, 13.0]], [[10.0, 11.0], [8.0, 9.0]]],
        [[[6.0, 7.0], [4.0, 5.0]], [[2.0, 3.0], [0.0, 1.0]]],
    ]
    test_case.assertTrue(np.allclose(out.numpy(), np_out, 1e-05, 1e-05))
    out = out.sum()
    out = out.backward()
    np_grad = np.ones_like(np_arr)
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05))


class TestFlip(flow.unittest.TestCase):
    def test_flip(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_flip]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()

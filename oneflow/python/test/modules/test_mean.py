"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import unittest
from collections import OrderedDict

import numpy as np

import oneflow.experimental as flow
from test_util import GenArgList


def _test_mean(test_case, shape, device):
    input = flow.Tensor(
        np.random.randn(*shape), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.mean(input, dim=1)
    np_out = np.mean(input.numpy(), axis=1)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

    input = flow.Tensor(
        np.random.randn(*shape), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.mean(input, dim=0)
    np_out = np.mean(input.numpy(), axis=0)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))


def _test_mean_negative_dim(test_case, shape, device):
    if len(shape) < 4:
        shape = (2, 3, 4, 5)
    input = flow.Tensor(
        np.random.randn(*shape), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.mean(input, dim=(-2, -1, -3))
    np_out = np.mean(input.numpy(), axis=(-2, -1, -3))
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))


def _test_mean_backward(test_case, shape, device):
    np_arr = np.random.randn(*shape)
    x = flow.Tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    y = flow.mean(x, dim=1)
    z = y.sum()
    z.backward()
    np_grad = np.zeros(shape=np_arr.shape)
    np_grad[:] = 1 / x.size(1)
    test_case.assertTrue(np.allclose(x.grad.numpy(), np_grad, 1e-5, 1e-5))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestMean(flow.unittest.TestCase):
    def test_mean(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_mean,
            _test_mean_negative_dim,
            _test_mean_backward,
        ]
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 4, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()

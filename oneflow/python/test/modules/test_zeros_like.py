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
import torch

import oneflow.experimental as flow
from test_util import GenArgList


def _test_zeros_like_float(test_case, shape, device):
    x = flow.Tensor(np.random.randn(*shape), device=flow.device(device))
    y = flow.zeros_like(x)
    test_case.assertTrue(y.dtype is flow.float32)
    test_case.assertTrue(y.shape == x.shape)
    test_case.assertTrue(y.device == x.device)

    y_numpy = np.zeros_like(x.numpy())
    test_case.assertTrue(np.array_equal(y.numpy(), y_numpy))


def _test_zeros_like_int(test_case, shape, device):
    x = flow.Tensor(np.random.randn(*shape), dtype=flow.int, device=flow.device(device))
    y = flow.zeros_like(x)
    test_case.assertTrue(y.dtype is flow.int)
    test_case.assertTrue(y.shape == x.shape)
    test_case.assertTrue(y.device == x.device)

    y_numpy = np.zeros_like(x.numpy())
    test_case.assertTrue(np.array_equal(y.numpy(), y_numpy))


def _test_zeros_like_torch(test_case, shape, device):
    np_arr = np.random.randn(*shape)
    x = flow.Tensor(np_arr, device=device)
    torch_x = torch.tensor(np_arr, device=device)

    y = flow.zeros_like(x)
    y_torch = torch.zeros_like(torch_x)

    test_case.assertTrue(np.allclose(y.numpy(), y_torch.cpu().numpy()))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestModule(flow.unittest.TestCase):
    def test_zeros_like(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_zeros_like_float,
            _test_zeros_like_int,
            _test_zeros_like_torch,
        ]
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 4, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()

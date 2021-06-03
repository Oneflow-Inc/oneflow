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


def _test_flatten(test_case, device):
    m = flow.nn.Flatten()
    x = flow.Tensor(32, 2, 5, 5, device=flow.device(device))
    flow.nn.init.uniform_(x)
    y = m(x)
    test_case.assertTrue(y.shape == flow.Size((32, 50)))
    test_case.assertTrue(np.array_equal(y.numpy().flatten(), x.numpy().flatten()))

    y2 = flow.flatten(x, start_dim=2)
    test_case.assertTrue(y2.shape == flow.Size((32, 2, 25)))
    test_case.assertTrue(np.array_equal(y2.numpy().flatten(), x.numpy().flatten()))

    y3 = x.flatten(start_dim=1)
    test_case.assertTrue(y3.shape == flow.Size((32, 50)))
    test_case.assertTrue(np.array_equal(y3.numpy().flatten(), x.numpy().flatten()))

    y4 = x.flatten(start_dim=1, end_dim=2)
    test_case.assertTrue(y4.shape == flow.Size((32, 10, 5)))
    test_case.assertTrue(np.array_equal(y4.numpy().flatten(), x.numpy().flatten()))

    y5 = flow.flatten(x)
    test_case.assertTrue(y5.shape == flow.Size((1600,)))
    test_case.assertTrue(np.array_equal(y5.numpy().flatten(), x.numpy().flatten()))


def _test_flatten_backward(test_case, device):
    m = flow.nn.Flatten().to(flow.device(device))
    x = flow.Tensor(2, 3, 4, 5, device=flow.device(device), requires_grad=True)
    flow.nn.init.uniform_(x)
    y = m(x)
    z = y.sum()
    z.backward()
    test_case.assertTrue(np.array_equal(np.ones(shape=(2, 3, 4, 5)), x.grad.numpy()))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestFlattenModule(flow.unittest.TestCase):
    def test_cast(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_flatten, _test_flatten_backward]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()

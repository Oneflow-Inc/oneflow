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


def _test_reshape_impl(test_case, device):
    x = np.random.randn(4, 4)
    input = flow.Tensor(
        x, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_shape = flow.reshape(input, shape=[2, 2, 2, -1]).numpy().shape
    np_shape = (2, 2, 2, 2)
    test_case.assertTrue(np.array_equal(of_shape, np_shape))
    of_out = flow.reshape(input, shape=[2, 2, 2, -1]).sum()
    of_out.backward()
    np_grad = [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-4, 1e-4))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestModule(flow.unittest.TestCase):
    def test_reshape(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_reshape_impl(test_case, *arg)


if __name__ == "__main__":
    unittest.main()

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

import numpy as np

import oneflow.experimental as flow
import oneflow.typing as tp


def _prelu(input, alpha):
    alpha = np.expand_dims(alpha, 0)
    alpha = np.expand_dims(alpha, 2)
    alpha = np.expand_dims(alpha, 3)
    return np.where(input > 0, input, input * alpha)


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestPReLU(flow.unittest.TestCase):
    def test_prelu(test_case):
        np_input = np.random.randn(2, 6, 5, 3)
        input = flow.Tensor(np_input, dtype=flow.float32)
        np_alpha = np.random.randn(1)
        prelu = flow.nn.PReLU(init=np_alpha)
        np_out = _prelu(np_input, np_alpha)
        of_out = prelu(input)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    def test_prelu_ndims(test_case):
        np_input = np.random.randn(2, 6, 5, 3)
        input = flow.Tensor(np_input, dtype=flow.float32)
        np_alpha = np.random.randn(6)
        prelu = flow.nn.PReLU(init=1.0, num_parameters=6)
        prelu_alpha = np.expand_dims(np_alpha, (1, 2))
        prelu.weight = flow.nn.Parameter(flow.Tensor(prelu_alpha, dtype=flow.float32))
        np_out = _prelu(np_input, np_alpha)
        of_out = prelu(input)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    def test_prelu_grad(test_case):
        np_input = np.random.randn(2, 6, 5, 3)
        input = flow.Tensor(np_input, dtype=flow.float32)
        np_alpha = np.random.randn(1)
        prelu = flow.nn.PReLU(init=np_alpha)
        of_out = prelu(input).sum()
        flow.add()
        of_out.backward()


if __name__ == "__main__":
    unittest.main()

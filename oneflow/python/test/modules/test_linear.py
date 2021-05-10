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


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestLinear(flow.unittest.TestCase):
    def test_identity(test_case):
        m = flow.nn.Identity()
        x = flow.Tensor(np.random.rand(2, 3, 4, 5))
        y = m(x)
        test_case.assertTrue(np.allclose(x.numpy(), y.numpy()))

    def test_linear_v1(test_case):
        linear = flow.nn.Linear(3, 8, False)
        input_arr = np.array(
            [
                [-0.94630778, -0.83378579, -0.87060891],
                [2.0289922, -0.28708987, -2.18369248],
                [0.35217619, -0.67095644, -1.58943879],
                [0.08086036, -1.81075924, 1.20752494],
                [0.8901075, -0.49976737, -1.07153746],
                [-0.44872912, -1.07275683, 0.06256855],
                [-0.22556897, 0.74798368, 0.90416439],
                [0.48339456, -2.32742195, -0.59321527],
            ],
            dtype=np.float32,
        )
        np_weight = np.ones((3, 8)).astype(np.float32)
        np_weight.fill(2.3)
        x = flow.Tensor(input_arr)
        flow.nn.init.constant_(linear.weight, 2.3)
        of_out = linear(x)
        np_out = np.matmul(input_arr, np_weight)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out))

    def test_linear_v2(test_case):
        linear = flow.nn.Linear(3, 8)
        input_arr = np.array(
            [
                [-0.94630778, -0.83378579, -0.87060891],
                [2.0289922, -0.28708987, -2.18369248],
                [0.35217619, -0.67095644, -1.58943879],
                [0.08086036, -1.81075924, 1.20752494],
                [0.8901075, -0.49976737, -1.07153746],
                [-0.44872912, -1.07275683, 0.06256855],
                [-0.22556897, 0.74798368, 0.90416439],
                [0.48339456, -2.32742195, -0.59321527],
            ],
            dtype=np.float32,
        )
        np_weight = np.ones((3, 8)).astype(np.float32)
        np_weight.fill(2.068758)
        np_bias = np.ones((8))
        np_bias.fill(0.23)
        x = flow.Tensor(input_arr)
        flow.nn.init.constant_(linear.weight, 2.068758)
        flow.nn.init.constant_(linear.bias, 0.23)
        of_out = linear(x)
        np_out = np.matmul(input_arr, np_weight)
        np_out += np_bias
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestIdentity(flow.unittest.TestCase):
    def test_identity(test_case):
        m = flow.nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        x = flow.Tensor(np.random.rand(2, 3, 4, 5))
        y = m(x)
        test_case.assertTrue(np.array_equal(x.numpy(), y.numpy()))


if __name__ == "__main__":
    unittest.main()

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


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestReduceSum(flow.unittest.TestCase):
    def test_reduce_sum(test_case):
        input_arr = np.random.randn(2, 3, 4, 5)
        of_out = flow.reduce_sum(flow.Tensor(input_arr), 1, True)
        np_out = np.sum(input_arr, 1, keepdims=True)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    def test_reduce_sum_v2(test_case):
        input_arr = np.random.randn(4, 1, 3, 2)
        of_out = flow.reduce_sum(flow.Tensor(input_arr), 3, False)
        np_out = np.sum(input_arr, 3, keepdims=False)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestReduceMean(flow.unittest.TestCase):
    def test_reduce_mean(test_case):
        input_arr = np.random.randn(2, 3, 4, 5)
        of_out = flow.reduce_mean(flow.Tensor(input_arr), 1, True)
        np_out = np.mean(input_arr, 1, keepdims=True)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    def test_reduce_mean_v2(test_case):
        input_arr = np.random.randn(4, 1, 3, 2)
        of_out = flow.reduce_mean(flow.Tensor(input_arr), 2, False)
        np_out = np.mean(input_arr, 2, keepdims=False)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


class TestReduceVariance(flow.unittest.TestCase):
    def test_reduce_variance(test_case):
        input_arr = np.random.randn(2, 3, 4, 5)
        of_out = flow.reduce_variance(flow.Tensor(input_arr), 1, True)
        np_out = np.var(input_arr, 1, keepdims=True)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    def test_reduce_variance_v2(test_case):
        input_arr = np.random.randn(4, 1, 3, 2)
        of_out = flow.reduce_variance(flow.Tensor(input_arr), 2, False)
        np_out = np.var(input_arr, 2, keepdims=False)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


if __name__ == "__main__":
    unittest.main()

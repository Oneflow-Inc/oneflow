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
import numpy as np
import oneflow as flow
import oneflow.typing as oft
import unittest
import os

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def _check(test_case, a, b, out):
    test_case.assertTrue(np.array_equal(np.minimum(a, b), out))


def _run_test(test_case, a, b, dtype, device):
    @flow.global_function(function_config=func_config)
    def BroadcastMinimum(
        a: oft.Numpy.Placeholder(a.shape, dtype=dtype),
        b: oft.Numpy.Placeholder(b.shape, dtype=dtype),
    ):
        with flow.scope.placement(device, "0:0"):
            return flow.math.minimum(a, b)

    out = BroadcastMinimum(a, b).get()
    _check(test_case, a, b, out.numpy())


@flow.unittest.skip_unless_1n1d()
class TestBroadcastMinimum(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_broadcast_minimum_random_gpu(test_case):
        a = np.random.rand(1024, 1024).astype(np.float32)
        b = np.random.rand(1024, 1024).astype(np.float32)
        _run_test(test_case, a, b, flow.float32, "gpu")

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_broadcast_minimum_broadcast_gpu(test_case):
        a = np.random.rand(1024, 1).astype(np.float32)
        b = np.random.rand(1, 1024).astype(np.float32)
        _run_test(test_case, a, b, flow.float32, "gpu")


if __name__ == "__main__":
    unittest.main()

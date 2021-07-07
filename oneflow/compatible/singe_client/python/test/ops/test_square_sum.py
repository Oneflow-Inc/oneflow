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


def _check(test_case, x, y):
    ref_y = np.array(np.sum(x ** 2))
    test_case.assertTrue(np.allclose(y, ref_y))


def _run_test(test_case, x, dtype, device):
    @flow.global_function(function_config=func_config)
    def SquareSum(x: oft.Numpy.Placeholder(x.shape, dtype=dtype)):
        with flow.scope.placement(device, "0:0"):
            return flow.experimental.square_sum(x)

    y = SquareSum(x).get()
    _check(test_case, x, y.numpy())


@flow.unittest.skip_unless_1n1d()
class TestSquareSum(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_square_sum_random_gpu(test_case):
        x = np.random.uniform(-0.01, 0.01, (64, 64)).astype(np.float32)
        _run_test(test_case, x, flow.float32, "gpu")

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_square_sum_small_blob_gpu(test_case):
        x = np.random.uniform(-0.01, 0.01, (64,)).astype(np.float32)
        _run_test(test_case, x, flow.float32, "gpu")


if __name__ == "__main__":
    unittest.main()

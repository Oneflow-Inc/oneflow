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


def _check(test_case, x, y, value, dtype=None):
    np_constant_like = np.full(x.shape, value)
    test_case.assertTrue(np.array_equal(np_constant_like, y))


def _run_test(test_case, x, value, dtype=None, device="gpu"):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.consistent_view())

    @flow.global_function(function_config=func_config)
    def ConstantLikeJob(x: oft.Numpy.Placeholder(x.shape)):
        return flow.constant_like(x, value=value, dtype=dtype)

    y = ConstantLikeJob(x).get()
    _check(test_case, x, y.numpy(), value, dtype=dtype)


@flow.unittest.skip_unless_1n1d()
class TestConstantLike(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_constant_like_gpu_float(test_case):
        x = np.random.rand(10, 3, 32, 1024).astype(np.float32)
        _run_test(test_case, x, 1.0, flow.float, "gpu")

    def test_constant_like_cpu_float(test_case):
        x = np.random.rand(10, 3, 32, 1024).astype(np.float32)
        _run_test(test_case, x, 2.0, flow.float, "cpu")

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_constant_like_gpu_double(test_case):
        x = np.random.rand(10, 3, 32, 1024).astype(np.float32)
        _run_test(test_case, x, 3.0, flow.double, "gpu")

    def test_constant_like_cpu_double(test_case):
        x = np.random.rand(10, 3, 32, 1024).astype(np.float32)
        _run_test(test_case, x, 4.0, flow.double, "cpu")

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_constant_like_gpu_int8(test_case):
        x = np.random.rand(10, 3, 32, 1024).astype(np.float32)
        _run_test(test_case, x, 5.0, flow.int8, "gpu")

    def test_constant_like_cpu_int8(test_case):
        x = np.random.rand(10, 3, 32, 1024).astype(np.float32)
        _run_test(test_case, x, 6.0, flow.int8, "cpu")

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_constant_like_gpu_int32(test_case):
        x = np.random.rand(10, 3, 32, 1024).astype(np.float32)
        _run_test(test_case, x, 7.0, flow.int32, "gpu")

    def test_constant_like_cpu_int32(test_case):
        x = np.random.rand(10, 3, 32, 1024).astype(np.float32)
        _run_test(test_case, x, 8.0, flow.int32, "cpu")

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_constant_like_gpu_int64(test_case):
        x = np.random.rand(10, 3, 32, 1024).astype(np.float32)
        _run_test(test_case, x, 9.0, flow.int64, "gpu")

    def test_constant_like_cpu_int64(test_case):
        x = np.random.rand(10, 3, 32, 1024).astype(np.float32)
        _run_test(test_case, x, 10.0, flow.int64, "cpu")

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_constant_like_gpu(test_case):
        x = np.random.rand(10, 3, 32, 1024).astype(np.float32)
        _run_test(test_case, x, 11.0, device="gpu")

    def test_constant_like_cpu(test_case):
        x = np.random.rand(10, 3, 32, 1024).astype(np.float32)
        _run_test(test_case, x, 12.0, device="cpu")


if __name__ == "__main__":
    unittest.main()

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
import oneflow as flow
import oneflow.typing as oft


def _check(test_case, x, y, out, case):
    if case == "add":
        np_out = np.add(x, y)
    elif case == "sub":
        np_out = np.subtract(x, y)
    elif case == "mul":
        np_out = np.multiply(x, y)
    elif case == "div":
        if type(y[0]) == np.float32 or type(y[0]) == np.double:
            np_out = np.divide(x, y)
        else:
            np_out = np.floor_divide(x, y)

    test_case.assertTrue(np.allclose(np_out, out, rtol=1e-5, atol=1e-5))


def _run_test(test_case, x, y, case, dtype=None, device="gpu"):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.consistent_view())

    @flow.global_function(function_config=func_config)
    def ScalarByTensorJob(
        x: oft.Numpy.Placeholder(x.shape, dtype=dtype),
        y: oft.Numpy.Placeholder(y.shape, dtype=dtype),
    ):
        if case == "add":
            return flow.math.add(x, y)
        elif case == "sub":
            return flow.math.subtract(x, y)
        elif case == "mul":
            return flow.math.multiply(x, y)
        elif case == "div":
            return flow.math.divide(x, y)

    out = ScalarByTensorJob(x, y).get()
    _check(test_case, x, y, out.numpy(), case)


@flow.unittest.skip_unless_1n1d()
class TestScalarByTensorInt(flow.unittest.TestCase):
    def test_scalar_add_by_tensor_gpu_float(test_case):
        x = np.random.rand(10, 3, 32, 1024).astype(np.float32)
        y = np.random.rand(1).astype(np.float32)
        _run_test(test_case, x, y, "add", flow.float, "gpu")

    def test_scalar_add_by_tensor_cpu_float(test_case):
        x = np.random.rand(10, 3, 32, 1024).astype(np.float32)
        y = np.random.rand(1).astype(np.float32)
        _run_test(test_case, x, y, "add", flow.float, "cpu")

    def test_scalar_add_by_tensor_gpu_double(test_case):
        x = np.random.rand(10, 3, 32, 1024).astype(np.double)
        y = np.random.rand(1).astype(np.double)
        _run_test(test_case, x, y, "add", flow.double, "gpu")

    def test_scalar_add_by_tensor_cpu_double(test_case):
        x = np.random.rand(10, 3, 32, 1024).astype(np.double)
        y = np.random.rand(1).astype(np.double)
        _run_test(test_case, x, y, "add", flow.double, "cpu")

    def test_scalar_add_by_tensor_gpu_int8(test_case):
        x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int8)
        y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int8)
        _run_test(test_case, x, y, "add", flow.int8, "gpu")

    def test_scalar_add_by_tensor_cpu_int8(test_case):
        x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int8)
        y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int8)
        _run_test(test_case, x, y, "add", flow.int8, "cpu")

    def test_scalar_add_by_tensor_gpu_int32(test_case):
        x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int32)
        y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int32)
        _run_test(test_case, x, y, "add", flow.int32, "gpu")

    def test_scalar_add_by_tensor_cpu_int32(test_case):
        x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int32)
        y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int32)
        _run_test(test_case, x, y, "add", flow.int32, "cpu")

    def test_scalar_add_by_tensor_gpu_int64(test_case):
        x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int64)
        y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int64)
        _run_test(test_case, x, y, "add", flow.int64, "gpu")

    def test_scalar_add_by_tensor_cpu_int64(test_case):
        x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int64)
        y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int64)
        _run_test(test_case, x, y, "add", flow.int64, "cpu")

    def test_scalar_sub_by_tensor_gpu_float(test_case):
        x = np.random.rand(10, 3, 32, 1024).astype(np.float32)
        y = np.random.rand(1).astype(np.float32)
        _run_test(test_case, x, y, "sub", flow.float, "gpu")

    def test_scalar_sub_by_tensor_cpu_float(test_case):
        x = np.random.rand(10, 3, 32, 1024).astype(np.float32)
        y = np.random.rand(1).astype(np.float32)
        _run_test(test_case, x, y, "sub", flow.float, "cpu")

    def test_scalar_sub_by_tensor_gpu_double(test_case):
        x = np.random.rand(10, 3, 32, 1024).astype(np.double)
        y = np.random.rand(1).astype(np.double)
        _run_test(test_case, x, y, "sub", flow.double, "gpu")

    def test_scalar_sub_by_tensor_cpu_double(test_case):
        x = np.random.rand(10, 3, 32, 1024).astype(np.double)
        y = np.random.rand(1).astype(np.double)
        _run_test(test_case, x, y, "sub", flow.double, "cpu")

    def test_scalar_sub_by_tensor_gpu_int8(test_case):
        x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int8)
        y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int8)
        _run_test(test_case, x, y, "sub", flow.int8, "gpu")

    def test_scalar_sub_by_tensor_cpu_int8(test_case):
        x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int8)
        y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int8)
        _run_test(test_case, x, y, "sub", flow.int8, "cpu")

    def test_scalar_sub_by_tensor_gpu_int32(test_case):
        x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int32)
        y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int32)
        _run_test(test_case, x, y, "sub", flow.int32, "gpu")

    def test_scalar_sub_by_tensor_cpu_int32(test_case):
        x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int32)
        y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int32)
        _run_test(test_case, x, y, "sub", flow.int32, "cpu")

    def test_scalar_sub_by_tensor_gpu_int64(test_case):
        x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int64)
        y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int64)
        _run_test(test_case, x, y, "sub", flow.int64, "gpu")

    def test_scalar_sub_by_tensor_cpu_int64(test_case):
        x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int64)
        y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int64)
        _run_test(test_case, x, y, "sub", flow.int64, "cpu")

    def test_scalar_mul_by_tensor_gpu_float(test_case):
        x = np.random.rand(10, 3, 32, 1024).astype(np.float32)
        y = np.random.rand(1).astype(np.float32)
        _run_test(test_case, x, y, "mul", flow.float, "gpu")

    def test_scalar_mul_by_tensor_cpu_float(test_case):
        x = np.random.rand(10, 3, 32, 1024).astype(np.float32)
        y = np.random.rand(1).astype(np.float32)
        _run_test(test_case, x, y, "mul", flow.float, "cpu")

    def test_scalar_mul_by_tensor_gpu_double(test_case):
        x = np.random.rand(10, 3, 32, 1024).astype(np.double)
        y = np.random.rand(1).astype(np.double)
        _run_test(test_case, x, y, "mul", flow.double, "gpu")

    def test_scalar_mul_by_tensor_cpu_double(test_case):
        x = np.random.rand(10, 3, 32, 1024).astype(np.double)
        y = np.random.rand(1).astype(np.double)
        _run_test(test_case, x, y, "mul", flow.double, "cpu")

    def test_scalar_mul_by_tensor_gpu_int8(test_case):
        x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int8)
        y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int8)
        _run_test(test_case, x, y, "mul", flow.int8, "gpu")

    def test_scalar_mul_by_tensor_cpu_int8(test_case):
        x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int8)
        y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int8)
        _run_test(test_case, x, y, "mul", flow.int8, "cpu")

    def test_scalar_mul_by_tensor_gpu_int32(test_case):
        x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int32)
        y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int32)
        _run_test(test_case, x, y, "mul", flow.int32, "gpu")

    def test_scalar_mul_by_tensor_cpu_int32(test_case):
        x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int32)
        y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int32)
        _run_test(test_case, x, y, "mul", flow.int32, "cpu")

    def test_scalar_mul_by_tensor_gpu_int64(test_case):
        x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int64)
        y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int64)
        _run_test(test_case, x, y, "mul", flow.int64, "gpu")

    def test_scalar_mul_by_tensor_cpu_int64(test_case):
        x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int64)
        y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int64)
        _run_test(test_case, x, y, "mul", flow.int64, "cpu")

    def test_scalar_div_by_tensor_gpu_float(test_case):
        x = np.random.rand(10, 3, 32, 1024).astype(np.float32)
        y = np.random.rand(1).astype(np.float32)
        _run_test(test_case, x, y, "div", flow.float, "gpu")

    def test_scalar_div_by_tensor_cpu_float(test_case):
        x = np.random.rand(10, 3, 32, 1024).astype(np.float32)
        y = np.random.rand(1).astype(np.float32)
        _run_test(test_case, x, y, "div", flow.float, "cpu")

    def test_scalar_div_by_tensor_gpu_double(test_case):
        x = np.random.rand(10, 3, 32, 1024).astype(np.double)
        y = np.random.rand(1).astype(np.double)
        _run_test(test_case, x, y, "div", flow.double, "gpu")

    def test_scalar_div_by_tensor_cpu_double(test_case):
        x = np.random.rand(10, 3, 32, 1024).astype(np.double)
        y = np.random.rand(1).astype(np.double)
        _run_test(test_case, x, y, "div", flow.double, "cpu")

    def test_scalar_div_by_tensor_gpu_int8(test_case):
        x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int8)
        y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int8)
        _run_test(test_case, x, y, "div", flow.int8, "gpu")

    def test_scalar_div_by_tensor_cpu_int8(test_case):
        x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int8)
        y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int8)
        _run_test(test_case, x, y, "div", flow.int8, "cpu")

    def test_scalar_div_by_tensor_gpu_int32(test_case):
        x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int32)
        y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int32)
        _run_test(test_case, x, y, "div", flow.int32, "gpu")

    def test_scalar_div_by_tensor_cpu_int32(test_case):
        x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int32)
        y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int32)
        _run_test(test_case, x, y, "div", flow.int32, "cpu")

    def test_scalar_div_by_tensor_gpu_int64(test_case):
        x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int64)
        y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int64)
        _run_test(test_case, x, y, "div", flow.int64, "gpu")

    def test_scalar_div_by_tensor_cpu_int64(test_case):
        x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int64)
        y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int64)
        _run_test(test_case, x, y, "div", flow.int64, "cpu")


if __name__ == "__main__":
    unittest.main()

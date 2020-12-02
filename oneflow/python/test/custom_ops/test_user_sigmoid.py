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
import os
import numpy as np
import math

import oneflow as flow
import oneflow.typing as oft


func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)

module_path = os.path.dirname(os.path.abspath(__file__))
print("module_path:", module_path)
print("pwd_path:", os.getcwd())

user_sigmoid_op = flow.experimental.custom_op_module("user_sigmoid", module_path)
user_sigmoid_op.py_api().cpp_def().py_kernel().build_load()


def numpy_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def numpy_sigmoid_grad(y, dy):
    return y * (1 - y) * dy


def make_job(input_shape, dtype=flow.float32):
    @flow.global_function(function_config=func_config)
    def sigmoid_job(x: oft.Numpy.Placeholder(input_shape, dtype=dtype)):
        return flow.math.sigmoid(x)

    return sigmoid_job


def make_grad_job(y_shape, dy_shape, dtype=flow.float32):
    @flow.global_function(function_config=func_config)
    def sigmoid_grad_job(
        y: oft.Numpy.Placeholder(y_shape, dtype=dtype),
        dy: oft.Numpy.Placeholder(dy_shape, dtype=dtype),
    ):
        return flow.math.sigmoid_grad(y, dy)

    return sigmoid_grad_job


@flow.unittest.skip_unless_1n1d()
class TestUserSigmoid(flow.unittest.TestCase):
    def test_user_sigmoid(test_case):
        flow.clear_default_session()

        def make_py_job(input_shape, dtype=flow.float32):
            @flow.global_function(function_config=func_config)
            def sigmoid_py_job(x: oft.Numpy.Placeholder(input_shape, dtype=dtype)):
                with flow.scope.placement("cpu", "0:0"):
                    return user_sigmoid_op.api.user_sigmoid_forward(x)

            return sigmoid_py_job

        x = np.ones((1, 10), dtype=np.float32)
        sig_job = make_job(x.shape)
        py_sig_job = make_py_job(x.shape)
        sig = sig_job(x).get().numpy()
        py_sig = py_sig_job(x).get().numpy()
        numpy_sig = numpy_sigmoid(x)
        print("sig : ", sig)
        print("py_sig : ", py_sig)
        print("numpy_sig : ", numpy_sig)
        test_case.assertTrue(np.allclose(sig, py_sig, rtol=1e-03, atol=1e-05))
        test_case.assertTrue(np.allclose(py_sig, numpy_sig, rtol=1e-03, atol=1e-05))

    def test_user_sigmoid_grad(test_case):
        flow.clear_default_session()

        def make_py_grad_job(y_shape, dy_shape, dtype=flow.float32):
            @flow.global_function(function_config=func_config)
            def sigmoid_py_grad_job(
                y: oft.Numpy.Placeholder(y_shape, dtype=dtype),
                dy: oft.Numpy.Placeholder(dy_shape, dtype=dtype),
            ):
                with flow.scope.placement("cpu", "0:0"):
                    return user_sigmoid_op.api.user_sigmoid_backward(y, dy)

            return sigmoid_py_grad_job

        x = np.ones((1, 10), dtype=np.float32)
        y = 0.5 * np.ones((1, 10), dtype=np.float32)
        dy = 0.2 * np.ones((1, 10), dtype=np.float32)
        sig_grad_job = make_grad_job(y.shape, dy.shape)
        py_sig_grad_job = make_py_grad_job(y.shape, dy.shape)
        sig_grad = sig_grad_job(y, dy).get().numpy()
        py_sig_grad = py_sig_grad_job(y, dy).get().numpy()
        numpy_sig_grad = numpy_sigmoid_grad(y, dy)
        print("sig_grad", sig_grad)
        print("py_sig_grad", py_sig_grad)
        print("numpy_sig_grad", numpy_sig_grad)
        test_case.assertTrue(np.allclose(sig_grad, py_sig_grad, rtol=1e-03, atol=1e-05))
        test_case.assertTrue(
            np.allclose(py_sig_grad, numpy_sig_grad, rtol=1e-03, atol=1e-05)
        )


if __name__ == "__main__":
    unittest.main()

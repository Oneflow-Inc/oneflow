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
from collections import OrderedDict
import unittest
import numpy as np
import oneflow as flow
import oneflow.typing as tp
import test_global_storage
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type


def diag_grad_np(input, diagonal, output, grad):
    input_shape = input.shape
    output_shape = output.shape
    grad_output = np.zeros(input_shape)

    if len(input_shape) == 1:
        stride0 = output_shape[1]
        beg = diagonal if diagonal >= 0 else stride0 * abs(diagonal)
        for i in range(input_shape[0]):
            if i > 0:
                beg += stride0 + 1

            if diagonal >= 0:
                grad_output[i] = grad[i][beg % stride0]
            if diagonal < 0:
                grad_output[i] = grad[(beg - i) // stride0][i]

        return grad_output
    else:
        stride01 = input_shape[1]
        beg = diagonal if diagonal >= 0 else stride01 * abs(diagonal)
        for i in range(output.shape[0]):
            if i > 0:
                beg += stride01 + 1

            if diagonal >= 0:
                grad_output[i][beg % stride01] = grad[i]
            if diagonal < 0:
                stride02 = input_shape[0]
                grad_output[(beg - i) // stride02][i] = grad[i]

        return grad_output


def compare_with_np(device_type, dtype, input_shape, diagonal):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_placement_scope(flow.scope.placement(device_type, "0:0"))

    input = (np.random.random(input_shape) * 100).astype(dtype)

    @flow.global_function(type="train", function_config=func_config)
    def diag_job(
        input: tp.Numpy.Placeholder(shape=input_shape, dtype=flow.float),
    ) -> tp.Numpy:
        input_var = flow.get_variable(
            "input",
            shape=input_shape,
            dtype=flow.float,
            initializer=flow.zeros_initializer(),
            trainable=True,
        )

        input = input + input_var
        input = flow.cast_to_current_logical_view(input)
        input = flow.cast(input, type_name_to_flow_type[dtype])
        output = flow.diag(input, diagonal)
        if (
            output.dtype == flow.int64
            or output.dtype == flow.int8
            or output.dtype == flow.int32
        ):
            output = flow.cast(output, flow.float)
        flow.optimizer.Adam(
            flow.optimizer.PiecewiseConstantScheduler([], [1e-4])
        ).minimize(output)

        flow.watch(input, test_global_storage.Setter("x"))
        flow.watch_diff(input, test_global_storage.Setter("x_diff"))
        flow.watch(output, test_global_storage.Setter("output"))
        flow.watch_diff(output, test_global_storage.Setter("output_diff"))

        return output

    # OneFlow
    output_of = diag_job(input)
    output_diff = test_global_storage.Get("output_diff").astype(dtype)
    x_diff_of = test_global_storage.Get("x_diff").astype(dtype)
    # np
    output_np = np.diag(input, diagonal)
    x_diff_np = diag_grad_np(input, diagonal, output_np, output_diff)

    assert np.allclose(output_of, output_np)
    assert np.allclose(x_diff_of, x_diff_np)


@flow.unittest.skip_unless_1n1d()
class TestCast(flow.unittest.TestCase):
    def test_diag_1d(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["dtype"] = ["float32", "int32"]
        arg_dict["input_shape"] = [(3,), (3, 3)]
        arg_dict["diagonal"] = [0, 2, -1]
        for arg in GenArgList(arg_dict):
            compare_with_np(*arg)


if __name__ == "__main__":
    unittest.main()

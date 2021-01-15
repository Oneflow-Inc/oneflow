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
import os
import math
import unittest
import oneflow as flow
import numpy as np
import oneflow.typing as tp
from test_util import GenArgList
from collections import OrderedDict
from typing import Dict

# use numpy calc softsign forward
def np_softsign_forward(input):
    input_shape = input.shape
    input = input.flatten()
    len = input.size()
    out = np.zeros_like(input)
    for i in range(len):
        out[i] = input[i] / (1 + abs(input[i]))
    return np.reshape(out, newshape=input_shape)

# use numpy calc softsign backward
def np_softsign_backward(input):
    input_shape = input.shape
    input = input.flatten()
    len = input.size()
    out = np.zeros_like(input)
    for i in range(len):
        out[i] = 1.0 / (1 + abs(input[i])) / (1 + abs(input[i]))
    return np.reshape(out, newshape=input_shape)


def compare_softsign_with_np(
    input_shape, device_type, value_type, machine_ids, device_counts
):
    if value_type[1] == flow.float16:
        input_1 = np.random.uniform(-1, 7, size=input_shape).astype(np.float16)
        input_1 = np.array(input_1, dtype=value_type[0])
    else:
        input_1 = np.random.uniform(-1, 7, size=input_shape).astype(value_type[0])
    assert device_type in ["cpu", "gpu"]

    flow.clear_default_session()

    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)
    
    func_config = flow.FunctionConfig()
    func_config.default_placement_scope(flow.scope.placement(device_type, machine_ids))

    if value_type == flow.float16:
        func_config.default_data_type(flow.float32)
    else:
        func_config.default_data_type(value_type[1])
    
    np_softsign_out = np_softsign_forward(input_1)
    np_softsign_diff = np_softsign_backward(input_1)

    np_softsign_out = np.array(np_softsign_out).astype(value_type[0])

    def assert_prediction_grad(blob: tp.Numpy):
        if value_type[1] == flow.float16:
            assert np.allclose(blob, _np_grad, atol=1e-3)
        else:
            assert np.allclose(blob, _np_grad, atol=1e-5)
    
    if value_type[1] == flow.float16:

        @flow.global_function(
            type="train", function_config=func_config,
        )
        def oneflow_softsign(
            of_input_1: tp.Numpy.Placeholder(shape=input_1.shape, dtype=flow.float32),
        ) -> tp.Numpy:
            with flow.scope.placement(device_type, "0:0"):
                v = flow.get_variable(
                    shape=input_1.shape,
                    dtype=flow.float32,
                    initializer=flow.zeros_initializer(),
                    name="x_var",
                )
                x_var = of_input_1 + v
                x_f16 = flow.cast(x_var, flow.float16)

            of_softsign_out_f16 = flow.nn.softsign(x_f16)
            of_softsign_out_f32 = flow.cast(of_softsign_out_f16, flow.float32)

            with flow.scope.placement(device_type, "0:0"):
                flow.optimizer.SGD(
                    flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
                ).minimize(of_softsign_out_f32)

            flow.watch_diff(x_var, assert_prediction_grad)

            return of_softsign_out_f32

    # Test float32/64
    else:

        @flow.global_function(
            type="train", function_config=func_config,
        )
        def oneflow_softsign(
            of_input_1: tp.Numpy.Placeholder(shape=input_1.shape, dtype=value_type[1]),
        ) -> tp.Numpy:
            with flow.scope.placement(device_type, "0:0"):
                v = flow.get_variable(
                    shape=input_1.shape,
                    dtype=value_type[1],
                    initializer=flow.zeros_initializer(),
                    name="x_var",
                )
                x_var = of_input_1 + v

            flow.watch_diff(x_var, assert_prediction_grad)

            of_softsign_out = flow.nn.softsign(x_var)

            with flow.scope.placement(device_type, "0:0"):
                flow.optimizer.SGD(
                    flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
                ).minimize(of_softsign_out)

            return of_softsign_out

    of_out_softsign = oneflow_softsign(input_1)

    if value_type[1] == flow.float16:
        assert np.allclose(of_out_softsign, np_out_softsign, atol=1e-2)
    else:
        assert np.allclose(of_out_softsign, np_out_softsign, atol=1e-5)

    
def gen_arg_dict(shape, device_type, value_type, machine_ids, device_counts):
    # Generate a dict to pass parameter to test case
    arg_dict = OrderedDict()
    arg_dict["input_shape"] = [shape]
    arg_dict["device_type"] = [device_type]
    if value_type == "float" and device_type == "cpu":
        arg_dict["value_type"] = [
            (np.float32, flow.float32),
            (np.float64, flow.float64),
        ]
    else:
        arg_dict["value_type"] = [
            (np.float32, flow.float16),
            (np.float32, flow.float32),
            (np.float64, flow.float64),
        ]
    arg_dict["machine_ids"] = [machine_ids]
    arg_dict["device_counts"] = [device_counts]
    return arg_dict


@flow.unittest.skip_unless_1n1d()
class Testsoftsign1n1d(flow.unittest.TestCase):
    def test_softsign_cpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=(3, 16),
            device_type="cpu",
            value_type="float",
            machine_ids="0:0",
            device_counts=1,
        )
        for arg in GenArgList(arg_dict):
            compare_softsign_with_np(*arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_softsign_gpu(test_case):
        arg_dict = gen_arg_dict(
            shape=(16, 16),
            device_type="gpu",
            value_type="float",
            machine_ids="0:0",
            device_counts=1,
        )
        for arg in GenArgList(arg_dict):
            compare_softsign_with_np(*arg)

@flow.unittest.skip_unless_1n2d()
class Testsoftsign1n2d(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_softsign_gpu_1n2d(test_case):
        arg_dict = gen_arg_dict(
            shape=(4, 8, 16),
            device_type="gpu",
            value_type="float",
            machine_ids="0:0-1",
            device_counts=2,
        )
        for arg in GenArgList(arg_dict):
            compare_softsign_with_np(*arg)

if __name__ == "__main__":
    unittest.main()

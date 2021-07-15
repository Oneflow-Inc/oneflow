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
import oneflow as flow
import numpy as np
import oneflow.typing as tp
from test_util import GenArgList
import unittest
from collections import OrderedDict
from typing import Dict
import os


def _compare_relu6_with_np(
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

    def np_relu6(input):
        out = np.clip(input, 0.0, 6.0)
        return np.array(out).astype(value_type[0])

    np_out_relu6 = np_relu6(input_1)

    def np_diff(input):
        input_shape = input.shape
        input = input.flatten()
        elem_cnt = input.size
        diff = np.zeros(shape=(elem_cnt,))
        for i in range(elem_cnt):
            if input[i] > 0 and input[i] < 6:
                diff[i] = 1
        diff = np.reshape(diff, newshape=input_shape)
        diff = np.array(diff, dtype=value_type[0])
        return diff

    _np_grad = np_diff(input_1)

    def assert_prediction_grad(blob: tp.Numpy):
        if value_type[1] == flow.float16:
            assert np.allclose(blob, _np_grad, atol=1e-3)
        else:
            assert np.allclose(blob, _np_grad, atol=1e-5)

    if value_type[1] == flow.float16:

        @flow.global_function(
            type="train", function_config=func_config,
        )
        def oneflow_relu6(
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

            of_relu6_out_f16 = flow.nn.relu6(x_f16)
            of_relu6_out_f32 = flow.cast(of_relu6_out_f16, flow.float32)

            with flow.scope.placement(device_type, "0:0"):
                flow.optimizer.SGD(
                    flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
                ).minimize(of_relu6_out_f32)

            flow.watch_diff(x_var, assert_prediction_grad)

            return of_relu6_out_f32

    elif value_type[1] == flow.float32 or value_type[1] == flow.float64:

        @flow.global_function(
            type="train", function_config=func_config,
        )
        def oneflow_relu6(
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

            of_relu6_out = flow.nn.relu6(x_var)

            with flow.scope.placement(device_type, "0:0"):
                flow.optimizer.SGD(
                    flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
                ).minimize(of_relu6_out)

            return of_relu6_out

    of_out_relu6 = oneflow_relu6(input_1)

    if value_type[1] == flow.float16:
        assert np.allclose(of_out_relu6, np_out_relu6, atol=1e-3)
    else:
        assert np.allclose(of_out_relu6, np_out_relu6, atol=1e-5)


def _gen_arg_dict(shape, device_type, value_type, machine_ids, device_counts):
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
class Testrelu61n1d(flow.unittest.TestCase):
    def test_relu6_cpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=(16, 16),
            device_type="cpu",
            value_type="float",
            machine_ids="0:0",
            device_counts=1,
        )
        for arg in GenArgList(arg_dict):
            _compare_relu6_with_np(*arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_relu6_gpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=(4, 8, 16),
            device_type="gpu",
            value_type="float",
            machine_ids="0:0",
            device_counts=1,
        )
        for arg in GenArgList(arg_dict):
            _compare_relu6_with_np(*arg)


@flow.unittest.skip_unless_1n2d()
class Testrelu61n2d(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_relu6_gpu_1n2d(test_case):
        arg_dict = _gen_arg_dict(
            shape=(4, 8, 16),
            device_type="gpu",
            value_type="float",
            machine_ids="0:0-1",
            device_counts=2,
        )
        for arg in GenArgList(arg_dict):
            _compare_relu6_with_np(*arg)


if __name__ == "__main__":
    unittest.main()

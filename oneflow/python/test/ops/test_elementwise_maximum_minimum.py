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


def _compare_Xmum_with_np(
    input_shape,
    compare_type,
    device_type,
    machine_ids,
    device_counts,
    value_type,
    dx_only,
):
    input_1 = np.random.random(size=input_shape).astype(value_type["np_type"])
    if dx_only:
        input_2 = (np.zeros(input_shape) + 1.5).astype(value_type["np_type"])
    else:
        input_2 = np.random.random(size=input_shape).astype(value_type["np_type"])

    assert compare_type in ["maximum", "minimum"]
    assert device_type in ["cpu", "gpu"]

    flow.clear_default_session()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)

    func_config = flow.FunctionConfig()
    func_config.default_placement_scope(flow.scope.placement(device_type, machine_ids))

    def np_Xmum(input1, input2, compare_type):
        if compare_type == "minimum":
            return np.minimum(input1, input2)
        elif compare_type == "maximum":
            return np.maximum(input1, input2)

    np_out_Xmum = np_Xmum(input_1, input_2, compare_type)

    def np_diff(input1, input2, compare_type):
        elem_cnt = input1.size
        init_shape = input1.shape
        input1 = input1.flatten()
        input2 = input2.flatten()
        np_diff = np.zeros_like(input1)
        for i in range(elem_cnt):
            if compare_type == "maximum":
                if input1[i] > input2[i]:
                    np_diff[i] = 1
            elif compare_type == "minimum":
                if input1[i] < input2[i]:
                    np_diff[i] = 1
        return np.reshape(np_diff, init_shape)

    _np_grad = np_diff(input_1, input_2, compare_type)

    def assert_prediction_grad(blob: tp.Numpy):
        assert np.allclose(blob, _np_grad)

    @flow.global_function(
        type="train", function_config=func_config,
    )
    def oneflow_Xmum(
        of_input_1: tp.Numpy.Placeholder(
            shape=input_1.shape, dtype=value_type["of_type"]
        ),
        of_input_2: tp.Numpy.Placeholder(
            shape=input_2.shape, dtype=value_type["of_type"]
        ),
    ) -> tp.Numpy:
        with flow.scope.placement(device_type, "0:0"):
            v1 = flow.get_variable(
                shape=input_1.shape,
                dtype=value_type["of_type"],
                initializer=flow.zeros_initializer(),
                name="x1_var",
            )
            x1_var = of_input_1 + v1
        if not dx_only:
            v2 = flow.get_variable(
                shape=input_2.shape,
                dtype=value_type["of_type"],
                initializer=flow.zeros_initializer(),
                name="x2_var",
            )
            x2_var = of_input_2 + v2
        else:
            x2_var = flow.constant(
                value=1.5, shape=of_input_2.shape, dtype=value_type["of_type"]
            )

        flow.watch_diff(x1_var, assert_prediction_grad)  # Only Compare input1 Grad

        if compare_type == "maximum":
            of_Xmum_out = flow.math.maximum(x1_var, x2_var)
        elif compare_type == "minimum":
            of_Xmum_out = flow.math.minimum(x1_var, x2_var)

        with flow.scope.placement(device_type, "0:0"):
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
            ).minimize(of_Xmum_out)

        return of_Xmum_out

    of_out_Xmum = oneflow_Xmum(input_1, input_2)
    assert np.allclose(of_out_Xmum, np_out_Xmum)


def _gen_arg_dict(
    shape, compare_type, device_type, machine_ids, device_counts, value_type, dx_only,
):
    arg_dict = OrderedDict()
    arg_dict["input_shape"] = [*shape]
    arg_dict["compare_type"] = [*compare_type]
    arg_dict["device_type"] = [device_type]
    arg_dict["machine_ids"] = [machine_ids]
    arg_dict["device_counts"] = [device_counts]
    arg_dict["value_type"] = [*value_type]
    arg_dict["dx_only"] = [*dx_only]
    return arg_dict


@flow.unittest.skip_unless_1n1d()
class TestXmum1n1d(flow.unittest.TestCase):
    def test_Xmum_cpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=[(3, 3)],
            compare_type=["maximum", "minimum"],
            device_type="cpu",
            machine_ids="0:0",
            device_counts=1,
            value_type=[{"np_type": np.float32, "of_type": flow.float32}],
            dx_only=[True, False],
        )
        for arg in GenArgList(arg_dict):
            _compare_Xmum_with_np(*arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_Xmum_gpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=[(3, 3)],
            compare_type=["maximum", "minimum"],
            device_type="gpu",
            machine_ids="0:0",
            device_counts=1,
            value_type=[
                {"np_type": np.float32, "of_type": flow.float32},
                {"np_type": np.float64, "of_type": flow.float64},
            ],
            dx_only=[True, False],
        )
        for arg in GenArgList(arg_dict):
            _compare_Xmum_with_np(*arg)


@flow.unittest.skip_unless_1n2d()
class TestXmum1n2d(flow.unittest.TestCase):
    def test_Xmum_cpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=[(3, 3)],
            compare_type=["maximum", "minimum"],
            device_type="cpu",
            machine_ids="0:0-1",
            device_counts=2,
            value_type=[{"np_type": np.float32, "of_type": flow.float32}],
            dx_only=[True, False],
        )
        for arg in GenArgList(arg_dict):
            _compare_Xmum_with_np(*arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_Xmum_gpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=[(3, 3)],
            compare_type=["maximum", "minimum"],
            device_type="gpu",
            machine_ids="0:0-1",
            device_counts=2,
            value_type=[{"np_type": np.float32, "of_type": flow.float32}],
            dx_only=[True, False],
        )
        for arg in GenArgList(arg_dict):
            _compare_Xmum_with_np(*arg)


if __name__ == "__main__":
    unittest.main()

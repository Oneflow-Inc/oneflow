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


def _compare_stack_with_np(input_shape, axis, device_type, machine_ids, device_counts):
    input_1 = np.random.random(size=input_shape).astype(np.float32)
    input_2 = np.random.random(size=input_shape).astype(np.float32)

    assert device_type in ["cpu", "gpu"]

    flow.clear_default_session()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)

    func_config = flow.FunctionConfig()

    def np_stack(np_input_1, np_input_2, np_axis):
        return np.stack([np_input_1, np_input_2], axis=np_axis)

    np_out_stack = np_stack(input_1, input_2, axis)

    np_random_mul = np.random.random(size=np_out_stack.shape).astype(np.float32)

    def np_diff(np_input, np_axis):
        # We only test input_1 diff
        np_stack_grad = np.split(np_input, indices_or_sections=2, axis=np_axis)[0]
        # squeeze the useless dimension
        return np.squeeze(np_stack_grad, np_axis)

    _np_grad = np_diff(np_random_mul, axis)

    def assert_prediction_grad(blob: tp.Numpy):
        assert np.allclose(blob, _np_grad)

    @flow.global_function(
        type="train", function_config=func_config,
    )
    def oneflow_stack(
        of_input_1: tp.Numpy.Placeholder(shape=input_1.shape),
        of_input_2: tp.Numpy.Placeholder(shape=input_2.shape),
        of_mul: tp.Numpy.Placeholder(shape=np_random_mul.shape),
    ) -> tp.Numpy:
        with flow.scope.placement(device_type, "0:0"):
            v = flow.get_variable(
                shape=input_1.shape,
                dtype=flow.float32,
                initializer=flow.zeros_initializer(),
                name="x_var",
            )
            x_var = of_input_1 + v

        flow.watch_diff(x_var, assert_prediction_grad)

        of_stack_out = flow.stack([x_var, of_input_2], axis=axis)
        out = of_stack_out * of_mul
        with flow.scope.placement(device_type, "0:0"):
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
            ).minimize(out)

        return of_stack_out

    of_out_stack = oneflow_stack(input_1, input_2, np_random_mul)

    assert np.allclose(of_out_stack, np_out_stack)


def _gen_arg_dict(shape, axis, device_type, machine_ids, device_counts):
    # Generate a dict to pass parameter to test case
    arg_dict = OrderedDict()
    arg_dict["input_shape"] = [shape]
    arg_dict["axis"] = [axis]
    arg_dict["device_type"] = [device_type]
    arg_dict["machine_ids"] = [machine_ids]
    arg_dict["device_counts"] = [device_counts]
    return arg_dict


@flow.unittest.skip_unless_1n1d()
class Teststack1n1d(flow.unittest.TestCase):
    def test_stack_cpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=(4, 6), axis=2, device_type="cpu", machine_ids="0:0", device_counts=1
        )
        for arg in GenArgList(arg_dict):
            _compare_stack_with_np(*arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_stack_gpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=(3, 16, 32),
            axis=-4,
            device_type="gpu",
            machine_ids="0:0",
            device_counts=1,
        )
        for arg in GenArgList(arg_dict):
            _compare_stack_with_np(*arg)


@flow.unittest.skip_unless_1n2d()
class Teststack1n2d(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_stack_gpu_1n2d(test_case):
        arg_dict = _gen_arg_dict(
            shape=(3, 8, 8, 4),
            axis=3,
            device_type="gpu",
            machine_ids="0:0-1",
            device_counts=2,
        )
        for arg in GenArgList(arg_dict):
            _compare_stack_with_np(*arg)


if __name__ == "__main__":
    unittest.main()

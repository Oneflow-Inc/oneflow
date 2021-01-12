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
import random


def _compare_scalar_pow_with_np(
    input_shape, exponent, device_type, value_type, machine_ids, device_counts
):
    input_1 = np.random.uniform(0, 1, size=input_shape).astype(value_type[0])

    assert device_type in ["cpu", "gpu"]

    flow.clear_default_session()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)

    func_config = flow.FunctionConfig()
    func_config.default_placement_scope(flow.scope.placement(device_type, machine_ids))

    func_config.default_data_type(value_type[1])

    def np_pow(input, exponent):
        out = np.power(input, exponent)
        return np.array(out).astype(value_type[0])

    np_out_pow = np_pow(input_1, exponent)

    def np_diff(input, exponent):
        diff = exponent * np.power(input, exponent - 1)
        return diff

    _np_grad = np_diff(input_1, exponent)

    def assert_prediction_grad(blob: tp.Numpy):
        assert np.allclose(blob, _np_grad, atol=1e-5)

    @flow.global_function(
        type="train", function_config=func_config,
    )
    def oneflow_pow(
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

        of_pow_out = flow.math.pow(x_var, exponent)

        with flow.scope.placement(device_type, "0:0"):
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
            ).minimize(of_pow_out)

        return of_pow_out

    of_out_pow = oneflow_pow(input_1)

    assert np.allclose(of_out_pow, np_out_pow, atol=1e-5)


def _gen_arg_dict(shape, exponent, device_type, value_type, machine_ids, device_counts):
    # Generate a dict to pass parameter to test case
    arg_dict = OrderedDict()
    arg_dict["input_shape"] = [shape]
    arg_dict["exponent"] = [exponent]
    arg_dict["device_type"] = [device_type]
    arg_dict["value_type"] = [
        (np.float32, flow.float32),
        (np.float64, flow.float64),
    ]
    arg_dict["machine_ids"] = [machine_ids]
    arg_dict["device_counts"] = [device_counts]
    return arg_dict


@flow.unittest.skip_unless_1n1d()
class TestScalarPow1n1d(flow.unittest.TestCase):
    def test_scalar_pow_cpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=(3, 3),
            exponent=1.4,
            device_type="cpu",
            value_type="float",
            machine_ids="0:0",
            device_counts=1,
        )
        for arg in GenArgList(arg_dict):
            _compare_scalar_pow_with_np(*arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_scalar_pow_gpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=(4, 4),
            exponent=2.3,
            device_type="gpu",
            value_type="float",
            machine_ids="0:0",
            device_counts=1,
        )
        for arg in GenArgList(arg_dict):
            _compare_scalar_pow_with_np(*arg)


@flow.unittest.skip_unless_1n2d()
class TestScalarPow1n2d(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_pow_gpu_1n2d(test_case):
        arg_dict = _gen_arg_dict(
            shape=(4, 8, 4),
            exponent=2.0,
            device_type="gpu",
            value_type="float",
            machine_ids="0:0-1",
            device_counts=2,
        )
        for arg in GenArgList(arg_dict):
            _compare_scalar_pow_with_np(*arg)


if __name__ == "__main__":
    unittest.main()

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


def _compare_logsoftmax_with_np(
    input_shape, axis, device_type, machine_ids, device_counts
):
    input_1 = np.random.random(size=input_shape).astype(np.float32)

    assert device_type in ["cpu", "gpu"]

    flow.clear_default_session()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)

    func_config = flow.FunctionConfig()
    func_config.default_placement_scope(flow.scope.placement(device_type, machine_ids))

    def np_logsoftmax(input, axis):
        exps = np.exp(input)
        softmax = exps / np.sum(exps, axis=axis, keepdims=True)
        return np.log(softmax)

    np_out_logsoftmax = np_logsoftmax(input_1, axis)

    def np_diff(x, axis):
        # assert output grad is 1
        _grad = np.ones_like(x)
        _sum = np.sum(_grad, axis=axis, keepdims=True)
        _diff = _grad - np.exp(x) * _sum
        return _diff

    _np_grad = np_diff(np_out_logsoftmax, axis)

    def assert_prediction_grad(blob: tp.Numpy):
        assert np.allclose(blob, _np_grad, atol=1e-5)

    @flow.global_function(
        type="train", function_config=func_config,
    )
    def oneflow_logsoftmax(
        of_input_1: tp.Numpy.Placeholder(shape=input_1.shape),
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

        of_logsoftmax_out = flow.nn.logsoftmax(x_var, axis)
        with flow.scope.placement(device_type, "0:0"):
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
            ).minimize(of_logsoftmax_out)

        return of_logsoftmax_out

    of_logsoftmax_out = oneflow_logsoftmax(input_1)

    assert np.allclose(of_logsoftmax_out, np_out_logsoftmax, rtol=1e-5)


def _gen_arg_dict(shape, axis, device_type, machine_ids, device_counts):
    # Generate a dict to pass parameter to test case
    arg_dict = OrderedDict()
    arg_dict["input_shape"] = [shape]
    arg_dict["axis"] = [*axis]  # Pass (-1, 1) -> unzip as -1, 1 two test cases
    arg_dict["device_type"] = [device_type]
    arg_dict["machine_ids"] = [machine_ids]
    arg_dict["device_counts"] = [device_counts]
    return arg_dict


@flow.unittest.skip_unless_1n1d()
class Testlogsoftmax1n1d(flow.unittest.TestCase):
    def test_logsoftmax_cpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=(2, 64, 32),
            axis=(1, -1),
            device_type="cpu",
            machine_ids="0:0",
            device_counts=1,
        )
        for arg in GenArgList(arg_dict):
            _compare_logsoftmax_with_np(*arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_logsoftmax_gpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=(2, 4, 6, 2),
            axis=(2, -2),
            device_type="gpu",
            machine_ids="0:0",
            device_counts=1,
        )
        for arg in GenArgList(arg_dict):
            _compare_logsoftmax_with_np(*arg)


@flow.unittest.skip_unless_1n2d()
class Testlogsoftmax1n2d(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_logsoftmax_gpu_1n2d(test_case):
        arg_dict = _gen_arg_dict(
            shape=(2, 8),
            axis=(-1, 1),
            device_type="gpu",
            machine_ids="0:0-1",
            device_counts=2,
        )
        for arg in GenArgList(arg_dict):
            _compare_logsoftmax_with_np(*arg)


if __name__ == "__main__":
    unittest.main()

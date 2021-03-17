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
import unittest
import numpy as np
import tensorflow as tf
import oneflow as flow
from collections import OrderedDict

from test_util import GenArgList, type_name_to_flow_type
import test_global_storage
import oneflow.typing as tp
import os


def exclu_func(np_rev, shape, axis):
    np_exclu = np.zeros_like(np_rev)
    if len(shape) == 4:
        if 0 == axis:
            np_exclu[1:, :, :, :] = np_rev[:-1, :, :, :]
        elif 1 == axis:
            np_exclu[:, 1:, :, :] = np_rev[:, :-1, :, :]
        elif 2 == axis:
            np_exclu[:, :, 1:, :] = np_rev[:, :, :-1, :]
        elif 3 == axis:
            np_exclu[:, :, :, 1:] = np_rev[:, :, :, :-1]
    elif len(shape) == 3:
        if 0 == axis:
            np_exclu[1:, :, :] = np_rev[:-1, :, :]
        elif 1 == axis:
            np_exclu[:, 1:, :] = np_rev[:, :-1, :]
        elif 2 == axis:
            np_exclu[:, :, 1:] = np_rev[:, :, :-1]
    elif len(shape) == 2:
        if 0 == axis:
            np_exclu[1:, :] = np_rev[:-1, :]
        elif 1 == axis:
            np_exclu[:, 1:] = np_rev[:, :-1]
    elif len(shape) == 1:
        if 0 == axis:
            np_exclu[1:] = np_rev[:-1]
    return np_exclu


def _compare_cumsum_with_np(
    device_type, shape, axis, reverse, exclusive, machine_ids, device_counts,
):
    assert device_type in ["gpu", "cpu"]
    assert axis >= 0 and axis < len(shape)

    flow.clear_default_session()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)

    func_config = flow.FunctionConfig()
    func_config.default_placement_scope(flow.scope.placement(device_type, machine_ids))
    func_config.default_data_type(flow.float32)

    @flow.global_function(type="train", function_config=func_config)
    def CumsumJob() -> tp.Numpy:
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                "x",
                shape=shape,
                dtype=flow.float32,
                initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                trainable=True,
            )
            out = flow.math.cumsum(x, axis=axis, exclusive=exclusive, reverse=reverse)
            with flow.scope.placement(device_type, "0:0"):
                flow.optimizer.SGD(
                    flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
                ).minimize(out)

            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(out, test_global_storage.Setter("out"))
            flow.watch_diff(out, test_global_storage.Setter("out_diff"))

            return out

    # OneFlow
    of_out = CumsumJob()

    # Numpy
    of_x = test_global_storage.Get("x")
    if reverse == True:
        np_rev = np.flip(of_x, axis)
        np_rev = np.cumsum(np_rev, axis, float)
        if exclusive == True:
            np_rev = exclu_func(np_rev, shape, axis)
        np_out = np.flip(np_rev, axis)
    else:
        np_rev = np.cumsum(of_x, axis, float)
        if exclusive == True:
            np_out = exclu_func(np_rev, shape, axis)
        else:
            np_out = np_rev

    assert np.allclose(of_out, np_out, atol=1e-03)


def _gen_arg_dict(device_type, shape, machine_ids, device_counts):
    # Generate a dict to pass parameter to test case
    arg_dict = OrderedDict()
    arg_dict["device_type"] = [device_type]
    arg_dict["shape"] = [shape]
    arg_dict["axis"] = [0, 1]
    arg_dict["reverse"] = [True, False]
    arg_dict["exclusive"] = [True, False]
    arg_dict["machine_ids"] = [machine_ids]
    arg_dict["device_counts"] = [device_counts]
    return arg_dict


@flow.unittest.skip_unless_1n1d()
class Testzeros1n1d(flow.unittest.TestCase):
    def test_cumsum_cpu(test_case):
        arg_dict = _gen_arg_dict(
            device_type="cpu", shape=[5, 4, 3], machine_ids="0:0", device_counts=1,
        )
        for arg in GenArgList(arg_dict):
            _compare_cumsum_with_np(*arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_cumsum_gpu(test_case):
        arg_dict = _gen_arg_dict(
            device_type="gpu", shape=[5, 4, 3], machine_ids="0:0", device_counts=1,
        )
        for arg in GenArgList(arg_dict):
            _compare_cumsum_with_np(*arg)


@flow.unittest.skip_unless_1n2d()
class Testzeros1n2d(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_cumsum_gpu_1n2d(test_case):
        arg_dict = _gen_arg_dict(
            device_type="gpu", shape=[5, 4, 3], machine_ids="0:0-1", device_counts=2,
        )
        for arg in GenArgList(arg_dict):
            _compare_cumsum_with_np(*arg)


if __name__ == "__main__":
    unittest.main()

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
from collections import OrderedDict

import numpy as np
import oneflow as flow
from test_util import GenArgDict
import oneflow.typing as oft
import os

flow_to_np_dtype_dict = {
    flow.int32: np.int32,
    flow.float: np.single,
    flow.double: np.float,
}


def _random_input(shape, dtype):
    if np.issubdtype(dtype, np.integer):
        return np.random.random_integers(low=-10, high=10, size=shape)
    elif np.issubdtype(dtype, np.floating):
        rng = np.random.default_rng()
        return rng.standard_normal(size=shape, dtype=dtype)
    else:
        raise NotImplementedError


def _of_assign_and_relu(value, dtype, device_type, assign=flow.assign):
    flow.clear_default_session()
    if os.getenv("ONEFLOW_TEST_CPU_ONLY") is None:
        flow.config.gpu_device_num(1)
    flow.config.cpu_device_num(1)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(dtype)
    func_config.default_placement_scope(flow.scope.placement(device_type, "0:0"))

    @flow.global_function(function_config=func_config)
    def assign_fn(value_def: oft.Numpy.Placeholder(value.shape, dtype=dtype)):
        var = flow.get_variable(
            name="var",
            shape=value.shape,
            dtype=dtype,
            initializer=flow.constant_initializer(0),
        )
        assign(var, value_def)

    @flow.global_function(function_config=func_config)
    def relu_fn():
        var = flow.get_variable(
            name="var",
            shape=value.shape,
            dtype=dtype,
            initializer=flow.constant_initializer(0),
        )
        return flow.nn.relu(var)

    assign_fn(value)
    return relu_fn().get().numpy()


def _np_relu(x):
    return np.maximum(x, 0)


def _compare_with_np(test_case, shape, dtype, device_type, assign):
    x = _random_input(shape, flow_to_np_dtype_dict[dtype])
    of_y = _of_assign_and_relu(x, dtype, device_type, assign=assign)
    test_case.assertTrue(np.allclose(_np_relu(x), of_y))


@flow.unittest.skip_unless_2n1d()
class TestTwoNodeAssign(flow.unittest.TestCase):
    def test_2node_assign(test_case):
        if flow.eager_execution_enabled():
            assign = flow.experimental.eager_assign_121
        else:
            assign = flow.assign
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(10), (30, 4), (8, 256, 20)]
        arg_dict["dtype"] = [flow.float, flow.double]
        arg_dict["device_type"] = ["cpu"]
        arg_dict["assign"] = [assign]
        for arg in GenArgDict(arg_dict):
            _2node_compare_with_np(test_case, **arg)


def _2node_compare_with_np(test_case, shape, dtype, device_type, assign):
    x = _random_input(shape, flow_to_np_dtype_dict[dtype])
    of_y = _2node_of_assign_and_relu(x, dtype, device_type, assign=assign)
    np_y = _np_relu(x)
    test_case.assertTrue(np.allclose(np_y, of_y))


def _2node_of_assign_and_relu(value, dtype, device_type, assign=flow.assign):
    flow.clear_default_session()
    flow.config.machine_num(2)
    if os.getenv("ONEFLOW_TEST_CPU_ONLY") is None:
        flow.config.gpu_device_num(1)
    flow.config.cpu_device_num(1)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(dtype)
    func_config.default_placement_scope(flow.scope.placement(device_type, "0:0"))

    @flow.global_function(function_config=func_config)
    def assign_fn(value_def: oft.Numpy.Placeholder(value.shape, dtype=dtype)):
        with flow.scope.placement(device_type, "1:0"):
            var = flow.get_variable(
                name="var",
                shape=value.shape,
                dtype=dtype,
                initializer=flow.constant_initializer(0),
            )
            assign(var, value_def)

    @flow.global_function(function_config=func_config)
    def relu_fn():
        with flow.scope.placement(device_type, "1:0"):
            var = flow.get_variable(
                name="var",
                shape=value.shape,
                dtype=dtype,
                initializer=flow.constant_initializer(0),
            )
        ret = flow.nn.relu(var)
        return ret

    assign_fn(value)
    relu_ret = relu_fn().get()
    return relu_ret.numpy()


@flow.unittest.skip_unless_1n1d()
class TestAssign(flow.unittest.TestCase):
    def test_assign(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(10), (30, 4), (8, 256, 20)]
        arg_dict["dtype"] = [flow.float, flow.double]
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["assign"] = [flow.assign]
        for arg in GenArgDict(arg_dict):
            _compare_with_np(test_case, **arg)

    def test_eager_assign_121(test_case):
        if not flow.eager_execution_enabled():
            return
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(10), (30, 4), (8, 256, 20)]
        arg_dict["dtype"] = [flow.float, flow.double]
        arg_dict["device_type"] = ["cpu"]
        arg_dict["assign"] = [flow.experimental.eager_assign_121]
        for arg in GenArgDict(arg_dict):
            _compare_with_np(test_case, **arg)


if __name__ == "__main__":
    unittest.main()

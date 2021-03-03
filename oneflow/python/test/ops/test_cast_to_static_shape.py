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
from collections import OrderedDict
import oneflow as flow
from test_util import GenArgDict, type_name_to_flow_type, type_name_to_np_type


def _make_cast_to_static_shape_fn(
    test_case, shape, data_type, device_type, device_num, compare_diff_fn
):
    dtype = type_name_to_flow_type[data_type]
    require_grad = dtype is flow.float32

    flow.clear_default_session()
    if device_type == "gpu":
        flow.config.gpu_device_num(device_num)
    elif device_type == "cpu":
        flow.config.cpu_device_num(device_num)
    else:
        raise ValueError

    assert device_num > 0
    func_config = flow.FunctionConfig()
    func_config.default_data_type(dtype)
    func_config.default_placement_scope(
        flow.scope.placement(device_type, "0:0-{}".format(device_num - 1))
    )
    func_config.default_logical_view(flow.scope.mirrored_view())

    @flow.global_function(
        type="train" if require_grad else "predict", function_config=func_config
    )
    def cast_to_static_shape_fn(
        x: flow.typing.ListNumpy.Placeholder(shape=shape, dtype=dtype)
    ) -> flow.typing.ListNumpy:
        x_var = flow.get_variable(
            name="x_var",
            shape=(1,),
            dtype=flow.float32,
            initializer=flow.zeros_initializer(),
        )
        x = x + flow.cast(x_var, dtype=dtype)
        y = flow.cast_to_static_shape(x)
        test_case.assertFalse(y.is_dynamic)
        if require_grad:
            flow.watch_diff(x, compare_diff_fn)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
            ).minimize(y)
        return y

    return cast_to_static_shape_fn


def _random_input(shape, data_type):
    dtype = type_name_to_np_type[data_type]
    if data_type == "float32" or data_type == "double":
        return np.random.random_sample(shape).astype(dtype)
    elif data_type == "int32":
        return np.random.randint(low=0, high=100, size=shape).astype(dtype)
    else:
        raise NotImplementedError


def _check_cast_to_static_shape(test_case, shape, data_type, device_type, device_num):
    x = _random_input(shape, data_type)

    def comp(x, y):
        test_case.assertTrue(np.array_equal(x, y))

    def comp_diff(diff):
        dx = np.ones(shape)
        for d in diff.numpy_list():
            test_case.assertTrue(np.array_equal(d, dx))

    cast_to_static_shape_fn = _make_cast_to_static_shape_fn(
        test_case, shape, data_type, device_type, device_num, comp_diff
    )
    y = cast_to_static_shape_fn([x] * device_num)

    if isinstance(y, list):
        for y_ in y:
            comp(x, y_)
    elif isinstance(y, np.ndarray):
        comp(x, y)
    else:
        raise ValueError


# @flow.unittest.skip_unless_1n1d()
# TODO(zhangwenxiao, jiangxuefei): refine in multi-client
@unittest.skipIf(True, "skip for now because of single-client tensor_list removed")
class TestCastToStaticShape(flow.unittest.TestCase):
    def test_case_1(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(5, 4, 3), (10, 7)]
        arg_dict["data_type"] = ["float32", "double", "int32"]
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["device_num"] = [1]
        for arg in GenArgDict(arg_dict):
            _check_cast_to_static_shape(test_case, **arg)


# @flow.unittest.skip_unless_1n1d()
# TODO(zhangwenxiao, jiangxuefei): refine in multi-client
@unittest.skipIf(True, "skip for now because of single-client tensor_list removed")
class TestCastToStaticShapeParallel(flow.unittest.TestCase):
    def test_case_1(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(10,)]
        arg_dict["data_type"] = ["float32", "double", "int32"]
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["device_num"] = [4]
        for arg in GenArgDict(arg_dict):
            _check_cast_to_static_shape(test_case, **arg)


if __name__ == "__main__":
    unittest.main()

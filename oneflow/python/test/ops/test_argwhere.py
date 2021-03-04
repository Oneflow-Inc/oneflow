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
import numpy as np
import unittest
from collections import OrderedDict
import os

import oneflow as flow

from test_util import GenArgDict


def _np_dtype_to_of_dtype(np_dtype):
    if np_dtype == np.float32:
        return flow.float32
    elif np_dtype == np.int32:
        return flow.int32
    elif np_dtype == np.int64:
        return flow.int64
    elif np_dtype == np.int8:
        return flow.int8
    else:
        raise NotImplementedError


def _random_input(shape, dtype):
    if dtype == np.float32:
        rand_ = np.random.random_sample(shape).astype(np.float32)
        rand_[np.nonzero(rand_ < 0.5)] = 0.0
        return rand_
    elif dtype == np.int32:
        return np.random.randint(low=0, high=2, size=shape).astype(np.int32)
    elif dtype == np.int8:
        return np.random.randint(low=0, high=2, size=shape).astype(np.int8)
    else:
        raise NotImplementedError


def _of_argwhere(x, index_dtype, device_type="gpu", device_num=1, dynamic=False):
    data_type = _np_dtype_to_of_dtype(x.dtype)
    out_data_type = _np_dtype_to_of_dtype(index_dtype)

    flow.clear_default_session()
    if device_type == "gpu":
        flow.config.gpu_device_num(device_num)
    elif device_type == "cpu":
        flow.config.cpu_device_num(device_num)
    else:
        raise ValueError

    assert device_num > 0
    func_config = flow.FunctionConfig()
    func_config.default_data_type(data_type)
    func_config.default_placement_scope(
        flow.scope.placement(device_type, "0:0-{}".format(device_num - 1))
    )

    if dynamic is True:
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function("predict", function_config=func_config)
        def argwhere_fn(
            x: flow.typing.ListNumpy.Placeholder(x.shape, dtype=data_type)
        ) -> flow.typing.ListNumpy:
            return flow.argwhere(x, dtype=out_data_type)

        return argwhere_fn([x] * device_num)[0]

    else:
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function("predict", function_config=func_config)
        def argwhere_fn(
            x: flow.typing.Numpy.Placeholder(x.shape, dtype=data_type)
        ) -> flow.typing.ListNumpy:
            return flow.argwhere(x, dtype=out_data_type)

        return argwhere_fn(x)[0]


def _compare_with_np(
    test_case,
    shape,
    value_dtype,
    index_dtype,
    device_type="gpu",
    device_num=1,
    dynamic=False,
    verbose=False,
):
    if verbose:
        print("shape:", shape)
        print("value_dtype:", value_dtype)
        print("index_dtype:", index_dtype)
        print("device_type:", device_type)
        print("device_num:", device_num)
        print("dynamic:", dynamic)

    x = _random_input(shape, value_dtype)
    y = np.argwhere(x)
    of_y = _of_argwhere(
        x, index_dtype, device_type=device_type, device_num=device_num, dynamic=dynamic
    )
    if verbose is True:
        print("input:", x)
        print("np result:", y)
        print("of result:", of_y)
    test_case.assertTrue(np.array_equal(y, of_y))


def _dynamic_multi_iter_compare(
    test_case,
    iter_num,
    shape,
    value_dtype,
    index_dtype,
    device_type="gpu",
    verbose=False,
):
    x = [_random_input(shape, value_dtype) for _ in range(iter_num)]
    y = [np.argwhere(x_) for x_ in x]

    data_type = _np_dtype_to_of_dtype(value_dtype)
    out_data_type = _np_dtype_to_of_dtype(index_dtype)

    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(data_type)
    func_config.default_placement_scope(flow.scope.placement(device_type, "0:0"))
    func_config.default_logical_view(flow.scope.mirrored_view())

    @flow.global_function("predict", function_config=func_config)
    def argwhere_fn(
        x: flow.typing.Numpy.Placeholder(tuple(shape), dtype=data_type)
    ) -> flow.typing.ListNumpy:
        return flow.argwhere(x, dtype=out_data_type)

    results = []
    for x_ in x:
        y_ = argwhere_fn(x_)[0]
        results.append(y_)

    for i, result in enumerate(results):
        test_case.assertTrue(np.array_equal(result, y[i]))


@flow.unittest.skip_unless_1n1d()
class TestArgwhere(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_argwhere(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(10), (30, 4), (8, 256, 20)]
        arg_dict["value_dtype"] = [np.float32, np.int32, np.int8]
        arg_dict["index_dtype"] = [np.int32, np.int64]
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["dynamic"] = [True, False]
        arg_dict["verbose"] = [False]
        for arg in GenArgDict(arg_dict):
            _compare_with_np(test_case, **arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_argwhere_multi_iter(test_case):
        arg_dict = OrderedDict()
        arg_dict["iter_num"] = [2]
        arg_dict["shape"] = [(20, 4)]
        arg_dict["value_dtype"] = [np.float32, np.int32, np.int8]
        arg_dict["index_dtype"] = [np.int32, np.int64]
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["verbose"] = [False]
        for arg in GenArgDict(arg_dict):
            _dynamic_multi_iter_compare(test_case, **arg)


@flow.unittest.skip_unless_1n4d()
class TestArgwhere4D(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_argwhere(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(10, 5)]
        arg_dict["value_dtype"] = [np.float32, np.int32, np.int8]
        arg_dict["index_dtype"] = [np.int32, np.int64]
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["device_num"] = [4]
        arg_dict["dynamic"] = [True]
        arg_dict["verbose"] = [False]
        for arg in GenArgDict(arg_dict):
            _compare_with_np(test_case, **arg)


if __name__ == "__main__":
    unittest.main()

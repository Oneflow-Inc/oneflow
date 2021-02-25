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
from test_util import GenArgList, type_name_to_flow_type
import oneflow.typing as oft
import os


def func_equal(a, b):
    return a == b


def func_not_equal(a, b):
    return a != b


def func_greater_than(a, b):
    return a > b


def func_greater_equal(a, b):
    return a >= b


def func_less_than(a, b):
    return a < b


def func_less_equal(a, b):
    return a <= b


def np_array(dtype, shape):
    if dtype == flow.int8:
        return np.random.randint(0, 127, shape).astype(np.int8)
    elif dtype == flow.int32:
        return np.random.randint(0, 10000, shape).astype(np.int32)
    elif dtype == flow.int64:
        return np.random.randint(0, 10000, shape).astype(np.int64)
    elif dtype == flow.float:
        return np.random.rand(*shape).astype(np.float32)
    elif dtype == flow.double:
        return np.random.rand(*shape).astype(np.double)
    else:
        assert False


def GenerateTest(test_case, a_shape, b_shape, dtype=flow.int32, device_type="gpu"):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(dtype)

    @flow.global_function(function_config=func_config)
    def MyTestJob(
        a: oft.Numpy.Placeholder(a_shape, dtype=dtype),
        b: oft.Numpy.Placeholder(b_shape, dtype=dtype),
    ):
        with flow.scope.placement(device_type, "0:0"):
            equal_out = func_equal(a, b)
            not_equal_out = func_not_equal(a, b)
            greater_than_out = func_greater_than(a, b)
            greater_equal_out = func_greater_equal(a, b)
            less_than_out = func_less_than(a, b)
            less_equal_out = func_less_equal(a, b)
            return (
                equal_out,
                not_equal_out,
                greater_than_out,
                greater_equal_out,
                less_than_out,
                less_equal_out,
            )

    a = np_array(dtype, a_shape)
    b = np_array(dtype, b_shape)

    (
        equal_out,
        not_equal_out,
        greater_than_out,
        greater_equal_out,
        less_than_out,
        less_equal_out,
    ) = MyTestJob(a, b).get()
    test_case.assertTrue(np.array_equal(equal_out.numpy(), func_equal(a, b)))
    test_case.assertTrue(np.array_equal(not_equal_out.numpy(), func_not_equal(a, b)))
    test_case.assertTrue(
        np.array_equal(greater_than_out.numpy(), func_greater_than(a, b))
    )
    test_case.assertTrue(
        np.array_equal(greater_equal_out.numpy(), func_greater_equal(a, b))
    )
    test_case.assertTrue(np.array_equal(less_than_out.numpy(), func_less_than(a, b)))
    test_case.assertTrue(np.array_equal(less_equal_out.numpy(), func_less_equal(a, b)))

    flow.clear_default_session()


@flow.unittest.skip_unless_1n1d()
class TestBroadcastLogicalOps(flow.unittest.TestCase):
    def test_broadcast_logical_cpu(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_case"] = [test_case]
        arg_dict["a_shape"] = [(64, 64)]
        arg_dict["b_shape"] = [(1, 64)]
        arg_dict["data_type"] = [
            flow.int32,
            flow.float,
        ]
        arg_dict["device_type"] = ["cpu"]

        for arg in GenArgList(arg_dict):
            if len(arg[1]) < len(arg[2]):
                continue
            GenerateTest(*arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_broadcast_logical_gpu(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_case"] = [test_case]
        arg_dict["a_shape"] = [(64, 64), (64, 64, 64)]
        arg_dict["b_shape"] = [(1, 64), (1, 64, 1)]
        arg_dict["data_type"] = [
            flow.int8,
            flow.int32,
            flow.int64,
            flow.float,
            flow.double,
        ]
        arg_dict["device_type"] = ["gpu"]

        for arg in GenArgList(arg_dict):
            if len(arg[1]) < len(arg[2]):
                continue
            GenerateTest(*arg)


if __name__ == "__main__":
    unittest.main()

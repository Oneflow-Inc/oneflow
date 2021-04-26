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
import os
from collections import OrderedDict

import numpy as np
import oneflow as flow
from test_util import GenArgList
from test_util import type_name_to_flow_type
from test_util import type_name_to_np_type
import oneflow.typing as tp
from oneflow._oneflow_internal.distribute import SplitDistribute, BroadcastDistribute

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def _test_empty_fp16(test_case, device_type, shape, device_count):
    assert device_type in ["gpu"]
    flow.clear_default_session()

    if device_type == "cpu":
        flow.config.cpu_device_num(device_count)
    elif device_type == "gpu":
        flow.config.gpu_device_num(device_count)

    @flow.global_function(function_config=func_config)
    def empty_fp16_job() -> tp.Numpy:
        with flow.scope.placement(device_type, "0:0-%d" % (device_count - 1)):
            out16 = flow.empty(dtype=flow.float16, shape=shape)
            test_case.assertTrue(out16.dtype == flow.float16)
            out32 = flow.cast(out16, flow.float32)
            return out32

    if shape == ():
        np_shape = (1,)
    else:
        np_shape = shape
    of_out = empty_fp16_job()
    test_case.assertTrue(of_out.shape == np_shape)
    test_case.assertTrue(of_out.dtype == np.float32)


def _test_empty(
    test_case, device_type, type_name, shape, device_count, sbp_parallel=None
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()

    if device_type == "cpu":
        flow.config.cpu_device_num(device_count)
    elif device_type == "gpu":
        flow.config.gpu_device_num(device_count)

    flow_type = type_name_to_flow_type[type_name]
    np_type = type_name_to_np_type[type_name]

    @flow.global_function(function_config=func_config)
    def empty_job() -> tp.Numpy:
        with flow.scope.placement(device_type, "0:0-%d" % (device_count - 1)):
            return flow.empty(dtype=flow_type, shape=shape, distribute=sbp_parallel)

    if shape == ():
        np_shape = (1,)
    else:
        np_shape = shape
    of_out = empty_job()
    test_case.assertTrue(of_out.shape == np_shape)
    test_case.assertTrue(of_out.dtype == np_type)


@flow.unittest.skip_unless_1n1d()
class TestEmpty1n1d(flow.unittest.TestCase):
    def test_empty(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["type_name_value"] = [
            "float32",
            "double",
            "int8",
            "int32",
            "int64",
        ]
        arg_dict["shape"] = [(2, 3, 4, 5), (2, 3), (100, 100), (512, 512), ()]
        arg_dict["device_count"] = [1]
        for arg in GenArgList(arg_dict):
            _test_empty(test_case, *arg)

    def test_empty_fp16(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["shape"] = [(2, 3, 4, 5), (2, 3), (100, 100), (512, 512), ()]
        arg_dict["device_count"] = [1]
        for arg in GenArgList(arg_dict):
            _test_empty_fp16(test_case, *arg)


@flow.unittest.skip_unless_1n2d()
class TestEmpty1n2d(flow.unittest.TestCase):
    def test_empty_fp16(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["shape"] = [(2, 3, 4, 5), (2, 3), (100, 100), (512, 512), ()]
        arg_dict["device_count"] = [2]
        for arg in GenArgList(arg_dict):
            _test_empty_fp16(test_case, *arg)

    def test_empty(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["type_name_value"] = [
            "float32",
            "double",
            "int8",
            "int32",
            "int64",
        ]
        arg_dict["shape"] = [(2, 3, 4, 5), (2, 3), (100, 100), (512, 512), ()]
        arg_dict["device_count"] = [2]
        arg_dict["sbp_parallel"] = [flow.distribute.broadcast()]
        for arg in GenArgList(arg_dict):
            _test_empty(test_case, *arg)

    def test_empty_sbp(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["type_name_value"] = [
            "float32",
        ]
        arg_dict["shape"] = [(10, 3, 4, 5)]
        arg_dict["device_count"] = [2]
        arg_dict["sbp_parallel"] = [
            flow.distribute.split(0),
            flow.distribute.broadcast(),
            "S(0)",
            "B",
            "P",
        ]
        for arg in GenArgList(arg_dict):
            _test_empty(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
